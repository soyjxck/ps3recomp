/*
 * ps3recomp - the register-file draw engine's D3D12 backend (Windows)
 *
 * rsx_draw_engine.c (libs/video/rsx_draw_engine.h) decodes the NV4097
 * register file and drives a backend through rsx_draw_backend: resources by
 * handle, pipelines from the decompilers' HLSL plus render state, and one
 * record per draw or clear. This file is that interface over Direct3D 12,
 * written against rsx_metal_backend.m's s_engine_backend, whose record
 * stream and object tables it mirrors so the two can be read side by side.
 *
 * Shape:
 *   - bind_* calls accumulate into s_pending; draw/clear/snapshot calls
 *     append an EngRecord. Nothing touches a command list until a submit.
 *   - Vertices, indices and constants are copied into one UPLOAD-heap arena
 *     per submit (the "stage"); texture rows go through the same arena and a
 *     copy record ordered with the draws.
 *   - A submit opens a command list on one of ENG_FRAMES allocator slots,
 *     walks the records emitting barriers from per-object state tracking,
 *     resolves the occlusion queries and signals a fence. Each slot owns a
 *     region of the shader-visible SRV heap and a sampler heap of its own,
 *     so a slot is reused only once its fence has passed.
 *   - A windowed present keeps at most ENG_INFLIGHT submits queued; every
 *     other submit (query fence, guest reference, headless) waits.
 *
 * Handles are 1-based indices into one object table (textures, colour and
 * depth targets, snapshots and views alike), because the engine binds all of
 * them through bind_textures. Pipelines have their own table.
 *
 * PS3RECOMP_D3D12_HEADLESS=1 renders without a window (tools/rsx_replay);
 * PS3RECOMP_RSX_ENGINE=vtable keeps the old rsx_state path in
 * rsx_d3d12_backend.c instead of this one.
 */
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <direct.h>

#include "rsx_draw_engine.h"
#include "rsx_texture_layout.h"
#include "rsx_d3d12_engine.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

/* COM through C: obj->lpVtbl->Method(obj, ...). */
#define CALL(o, m, ...) ((o)->lpVtbl->m((o), __VA_ARGS__))
#define CALL0(o, m)     ((o)->lpVtbl->m((o)))
#define RELEASE(p)      do { if (p) { (p)->lpVtbl->Release(p); (p) = NULL; } } while (0)

#define ENG_MAX_OBJECTS   4096
#define ENG_MAX_PIPES     4096
#define ENG_MAX_RECORDS   8192
#define ENG_MAX_VIEWS     256
#define ENG_MAX_BLOBS     4096
#define ENG_MAX_SAMPLERS  256
#define ENG_MAX_BUFS      16384
#define ENG_ALIGN         256u
#define ENG_STAGE_MAX     (384u << 20)
#define ENG_STAGE_START   (16u << 20)
#define ENG_FRAMES        4           /* allocator slots */
#define ENG_INFLIGHT      2           /* windowed presents queued at most */
#define ENG_SRV_PER_SLOT  16384u      /* shader-visible SRV descriptors per slot */
#define ENG_SMP_PER_SLOT  2048u       /* the D3D12 sampler-heap maximum */
#define ENG_SRV_TABLE     (RSX_BE_MAX_TEXTURES + RSX_BE_MAX_VERTEX_TEXTURES) /* 20 */
#define ENG_QUERIES_PER_SLOT 16384u
#define ENG_VIS_SLOTS     16384u
#define ENG_MAX_REPORTS   4096
#define ENG_SRV_NULL      ENG_MAX_OBJECTS   /* CPU SRV heap slot for the null texture */
#define ENG_DEPTH_FMT     DXGI_FORMAT_D32_FLOAT_S8X24_UINT
#define ENG_DEPTH_RES_FMT DXGI_FORMAT_R32G8X24_TYPELESS
#define ENG_DEPTH_SRV_FMT DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
#define ENG_SHADER_READ   (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)

typedef enum { OBJ_NONE = 0, OBJ_TEXTURE, OBJ_COLOR, OBJ_DEPTH, OBJ_SNAPSHOT, OBJ_VIEW } EngObjKind;

typedef struct {
    ID3D12Resource* res;        /* NULL for a view: see alias            */
    EngObjKind kind;
    u32 alias;                  /* OBJ_VIEW: the surface handle it views */
    DXGI_FORMAT fmt;
    u32 w, h, mips, faces;      /* host pixels                           */
    u32 gw, gh;                 /* guest pixels; w,h = gw,gh * the scale  */
    D3D12_RESOURCE_STATES state;
    int has_rtv, has_dsv;       /* RTV/DSV written at slot handle-1      */
    int retired;
} EngObj;

/* ---- internal resolution --------------------------------------------------
 *
 * RSX_SCALE=<factor> (1 = the title's own 1280x720; 2 = 2560x1440; 3 = 4K).
 * Render targets, depth targets and their snapshots are created `factor`
 * times the size the guest asked for; viewports, scissors and clear
 * rectangles are scaled to match; everything a shader samples is addressed
 * in normalised coordinates and needs nothing. What crosses back into the
 * guest's world is scaled down: a readback is resampled to the guest size,
 * an occlusion count is divided by factor^2, and a fragment program's WPOS
 * is divided back to guest pixels (a title computes screen UVs from it).
 * Guest textures (uploaded pixels) are never scaled.
 *
 * RSX_DISPLAY=windowed|borderless|fullscreen: a bordered window of
 * RSX_WINDOW=<w>x<h> (default 1280x720); a borderless window the size of
 * the primary display; or exclusive full screen at the display's current
 * mode. (RSX_FULLSCREEN=1 is the old spelling of borderless.) The present blit resamples the surface to
 * the window whatever the two sizes are.
 *
 * RSX_SCALE_MIN=<n> (default 64): a target smaller than n in either
 * dimension stays at the guest size. Drakengard 3 builds its 256x16
 * colour-grading table in a render target, texel by texel from WPOS; scaled,
 * the table came out wrong (its whites 75% grey) and tone mapping then put
 * olive blotches on dark surfaces. So the scale is per target, and so is
 * everything that follows from it, from the target a pass draws into:
 * viewports, scissors and clear rectangles (recorded in guest pixels, scaled
 * at encoding), occlusion counts, depth-snapshot sizes, and WPOS -- a pass
 * into an unscaled target uses a second pipeline state whose fragment program
 * does not divide it (EngPipeline.ps_plain, built on first use). As the
 * Metal engine does (rsx_metal_backend.m, eng_obj_fx). */
static float s_scale = 1.0f;
static u32 s_scale_min = 64;
extern volatile int g_rsx_display_reload;   /* rsx_draw_engine.c */
static u32 s_win_w, s_win_h;                 /* the window and swap chain */
typedef enum { DISP_WINDOWED = 0, DISP_BORDERLESS, DISP_FULLSCREEN } EngDisplayMode;
static EngDisplayMode s_display = DISP_WINDOWED;
static u32 sc_dim(u32 v) { u32 r = (u32)((float)v * s_scale + 0.5f); return v && !r ? 1u : r; }
static u32 sc_pos(u32 v) { return (u32)((float)v * s_scale + 0.5f); }
/* [x, x+w) in guest pixels -> host pixels, as the two ends rounded, so
 * neighbouring rectangles still meet. */
static void sc_rect(u32* x, u32* y, u32* w, u32* h)
{
    const u32 x0 = sc_pos(*x), y0 = sc_pos(*y), x1 = sc_pos(*x + *w), y1 = sc_pos(*y + *h);
    *x = x0; *y = y0; *w = x1 - x0; *h = y1 - y0;
}
/* Whether a target of this guest size is created at the internal resolution. */
static int eng_scales(u32 w, u32 h) { return s_scale != 1.0f && w >= s_scale_min && h >= s_scale_min; }
static void eng_read_scale(void)
{
    const char* m = getenv("RSX_SCALE_MIN");
    if (m && *m) s_scale_min = (u32)atoi(m);
    const char* e = getenv("RSX_SCALE");
    s_scale = 1.0f;
    if (e && *e) {
        const float f = (float)atof(e);
        if (f >= 0.5f && f <= 8.0f) s_scale = f;
        else fprintf(stderr, "[RSX d3d12] RSX_SCALE=%s ignored (0.5 to 8)\n", e);
    }
}

typedef struct {
    ID3DBlob* vs; ID3DBlob* ps;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc;   /* everything but the topology class */
    D3D12_INPUT_ELEMENT_DESC il[RSX_DSP_NUM_VERTEX_ATTR];
    ID3D12PipelineState* pso[3];               /* by topology class: point, line, triangle */
    int failed[3];
    int live;
    /* A fragment program that reads WPOS, at RSX_SCALE != 1: its text as
     * the decompiler gave it (WPOS not divided), and that build's states,
     * for passes into a target kept at the guest size. NULL otherwise. */
    char* ps_plain;
    ID3DBlob* ps1;
    ID3D12PipelineState* pso1[3];
    int failed1[3];
    /* The same geometry with the snapshot-copy pixel shader (see
     * eng_encode_copy_draw), by topology class, for one target format. */
    ID3D12PipelineState* copy_pso[3];
    DXGI_FORMAT copy_fmt[3];
    int copy_failed[3];
} EngPipeline;
static const char kSnapCopyHLSL[] =
    "Texture2D rsx_tex[16] : register(t0);\n"
    "struct PSInput { float4 position : SV_POSITION; };\n"
    "float4 main(PSInput input) : SV_Target0 { return rsx_tex[0].Load(int3((int2)input.position.xy, 0)); }\n";
static ID3DBlob* s_snap_copy_ps;
static u64 s_snap_incr_n, s_snap_full_n, s_snap_copy_draws;   /* RSX_GPU_TIME */
int g_rsx_snap_incr = -1;   /* RSX_SNAP_INCR (see eng_encode_records) */


typedef struct { u64 hash; ID3DBlob* blob; } EngBlob;
typedef struct { u64 key; } EngSampler;   /* the descriptor lives at CPU sampler slot i+1 */
typedef struct { u32 surface, remap, format, view; } EngView;

typedef enum {
    ENG_REC_DRAW, ENG_REC_CLEAR_COLOR, ENG_REC_CLEAR_DS, ENG_REC_DEPTH_RESOLVE,
    ENG_REC_COLOR_COPY, ENG_REC_CLEAR_DS_RECT, ENG_REC_UPLOAD
} EngRecKind;

typedef struct {
    EngRecKind kind;
    u32 rt[RSX_BE_MAX_COLOR_TARGETS];
    u32 nrt;
    u32 depth;
    u32 pipeline;
    rsx_topology topology;
    u32 vb_off, stride, vertex_count;
    u32 ib_off, index_count;
    u32 vs_cb_off, vs_cb_bytes;
    u32 ps_cb_off, ps_cb_bytes;
    u32 tex[RSX_BE_MAX_TEXTURES];
    int samp[RSX_BE_MAX_TEXTURES];
    u32 vtex[RSX_BE_MAX_VERTEX_TEXTURES];
    int vsamp[RSX_BE_MAX_VERTEX_TEXTURES];
    float vp[4];
    u32   sc[4];
    u32   stencil_ref;
    float clear_rgba[4];
    u32   clear_flags;
    float clear_depth;
    u8    clear_stencil;
    u32   resolve_dst;
    u32   resolve_packed;
    u32   vis;
    u32   vbuf;
    u32   clear_rect[4];
    /* ENG_REC_UPLOAD: stage offset, subresource and footprint. */
    u32   up_off, up_sub, up_w, up_h, up_pitch;
} EngRecord;

/* One UPLOAD-heap arena, mapped for its whole life. */
typedef struct {
    ID3D12Resource* res;
    u8* mapped;
    u32 cap;
    u64 fence;        /* the submit that last used it */
    int in_use;       /* handed to a submit that may still be running */
} EngStage;
#define ENG_STAGE_POOL 8


typedef struct { u32 slot, index; } EngVisReport;

/* A submit the GPU may still be running. */
typedef struct {
    u64 fence;
    int stage;                       /* index into s_stage, -1 none */
    u32 q_slot, q_count;             /* its queries, in allocator slot q_slot's region */
    u32* q_vis;                      /* vis counter per query */
    EngVisReport* reports; u32 n_reports;
    int live;
} EngSubmit;
#define ENG_MAX_SUBMITS 16

/* ---- state ---------------------------------------------------------------- */

static int s_active, s_headless, s_ready;
static HWND s_hwnd; static int s_window_closed;
/* Exclusive full screen is dropped by DXGI when the window loses focus and
 * does not come back on its own: set by WM_ACTIVATEAPP, acted on before the
 * next present (SetFullscreenState and the ResizeBuffers it requires must
 * not run inside the message handler). */
static int s_fs_restore;
static u32 s_width, s_height;
static ID3D12Device* s_dev;
static ID3D12CommandQueue* s_queue;
static IDXGISwapChain3* s_swap;
static ID3D12Resource* s_backbuf[3];
static ID3D12DescriptorHeap* s_backbuf_rtv_heap;
static ID3D12Resource* s_offscreen;              /* headless present target */
static ID3D12Fence* s_fence; static HANDLE s_fence_event; static u64 s_fence_value;
static ID3D12CommandAllocator* s_alloc[ENG_FRAMES]; static u64 s_alloc_fence[ENG_FRAMES];
static ID3D12GraphicsCommandList* s_list; static int s_list_open; static u32 s_slot;
static ID3D12DescriptorHeap* s_srv_cpu;          /* ENG_MAX_OBJECTS + 1 */
static ID3D12DescriptorHeap* s_rtv_cpu;
static ID3D12DescriptorHeap* s_dsv_cpu;
static ID3D12DescriptorHeap* s_smp_cpu;          /* ENG_MAX_SAMPLERS + 1 */
static ID3D12DescriptorHeap* s_srv_gpu;          /* ENG_FRAMES * ENG_SRV_PER_SLOT */
static ID3D12DescriptorHeap* s_smp_gpu[ENG_FRAMES];
static u32 s_srv_step, s_rtv_step, s_dsv_step, s_smp_step;
static u32 s_srv_used, s_smp_used;               /* within the current slot's region */
static ID3D12QueryHeap* s_qheap; static ID3D12Resource* s_qread; static u32 s_q_used;
/* RSX_GPU_TIME=1: a timestamp at the start and end of every command list.
 * Every 5 s: submits a second, the GPU's busy time, the mean GPU time of a
 * submit the walker waits on and of one it does not, and how long a list sat
 * in the queue before the GPU started it -- is a synchronous wait GPU work,
 * or latency? */
static ID3D12QueryHeap* s_tsheap; static ID3D12Resource* s_tsread; static u64 s_ts_freq;
static struct { u64 fence; int kind; LARGE_INTEGER cpu_exec; } s_ts_slot[ENG_FRAMES];
static int s_submit_kind;   /* 0 present / other, 1 waited on */
static u64 s_gt_draws, s_gt_bar_calls, s_gt_bars, s_gt_pso, s_gt_queries, s_gt_copies;

/* RSX_GPU_TIME=2: also a timestamp wherever the recorded work changes target
 * (a "pass": the draws into one set of targets, a clear, a copy, a depth
 * resolve, the uploads), and every 5 s the passes that took the most GPU
 * time, by kind, target size and format. Where the GPU time of a frame goes
 * at 4K, which scales differently from the walker's. */
#define ENG_PASS_MAX 1024u
typedef struct { u8 kind; u8 nrt; u16 w, h; u32 fmt; u32 draws; } EngPassDesc;
static EngPassDesc s_pass[ENG_FRAMES][ENG_PASS_MAX]; static u32 s_pass_n[ENG_FRAMES];
static ID3D12QueryHeap* s_pheap; static ID3D12Resource* s_pread;
static ID3D12RootSignature* s_rootsig;
static ID3D12RootSignature* s_helper_rootsig;
static ID3D12PipelineState* s_blit_pso, *s_depth_pso, *s_depth_pack_pso;
static ID3D12Resource* s_null_tex;
static SRWLOCK s_pipe_lock = SRWLOCK_INIT;
static int s_vsync = 1;
/* Windowed and borderless presents go through the compositor, which holds a
 * flip-model swap chain to the display's refresh even at sync interval 0
 * unless the chain allows tearing. With vsync off and the system supporting
 * it (DXGI_FEATURE_PRESENT_ALLOW_TEARING), the chain is created with the
 * flag and presents carry it, so the frame rate is not capped at the
 * refresh. Exclusive full screen does not need it. */
static int s_tearing = 0;

static EngObj s_obj[ENG_MAX_OBJECTS];
static u32 s_obj_count, s_obj_free[ENG_MAX_OBJECTS], s_obj_free_count;
static EngPipeline s_pipe[ENG_MAX_PIPES]; static u32 s_pipe_count;
static EngBlob s_blob[ENG_MAX_BLOBS]; static u32 s_blob_count;
static EngSampler s_samp[ENG_MAX_SAMPLERS]; static u32 s_samp_count;
static EngView s_view[ENG_MAX_VIEWS]; static u32 s_view_count;
static EngRecord s_rec[ENG_MAX_RECORDS]; static u32 s_rec_count, s_dropped;
static EngRecord s_pending;
static ID3D12Resource* s_buf[ENG_MAX_BUFS]; static u32 s_buf_bytes[ENG_MAX_BUFS];
static u32 s_buf_count, s_buf_free[ENG_MAX_BUFS], s_buf_free_count; static u8 s_buf_retired[ENG_MAX_BUFS];
static EngStage s_stage[ENG_STAGE_POOL]; static int s_stage_cur = -1;
static u32 s_stage_used, s_stage_want = ENG_STAGE_START;
static EngSubmit s_sub[ENG_MAX_SUBMITS];
static u32 s_submit_seq;
static u32 s_vs_cb_seq = ~0u, s_vs_cb_off, s_vs_cb_bytes;
static u64 s_vis_count[ENG_VIS_SLOTS]; static u32 s_vis_next, s_vis_cur;
static EngVisReport s_vis_pending[ENG_MAX_REPORTS]; static u32 s_vis_npending;
static u32 s_q_vis[ENG_QUERIES_PER_SLOT];     /* vis counter per query of the list being encoded */
/* Set on a query whose pass drew into a scaled target: its count is in host
 * pixels and is divided by the scale squared when read. */
#define ENG_Q_SCALED 0x80000000u
static u64 eng_q_count(u32 qv, u64 c)
{
    if ((qv & ENG_Q_SCALED) && s_scale != 1.0f) c = (u64)((double)c / ((double)s_scale * (double)s_scale) + 0.5);
    return c;
}
static u32 s_clear_argb;
static u32 s_fallback_depth[8]; static u32 s_fallback_n;
static u32 s_present_center;                  /* last presented centre pixel, headless */
static u32 s_stat_tex, s_stat_snap, s_stat_snap_reused, s_stat_buf, s_stat_rt;   /* created since the last RSX_OBJ_STATS line */

/* Released depth-snapshot textures, kept for the next request of the same
 * size and format: a committed D3D12 resource costs the walker a kernel
 * round trip to create and another to free, and this title asks for a few
 * snapshots a frame. */
#define ENG_SNAP_POOL 64
static struct { ID3D12Resource* res; DXGI_FORMAT fmt; u32 w, h; u64 fence; D3D12_RESOURCE_STATES state; } s_snap_pool[ENG_SNAP_POOL];
static u32 s_snap_pool_n;

/* ---- small helpers -------------------------------------------------------- */

static u64 fnv1a64(const void* data, u32 n, u64 h)
{
    const u8* p = (const u8*)data;
    for (u32 i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static DXGI_FORMAT eng_dxgi(rsx_be_format f)
{
    switch (f) {
    case RSX_BE_FMT_R8:              return DXGI_FORMAT_R8_UNORM;
    case RSX_BE_FMT_R8G8:            return DXGI_FORMAT_R8G8_UNORM;
    case RSX_BE_FMT_R8G8B8A8:        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case RSX_BE_FMT_BC1:             return DXGI_FORMAT_BC1_UNORM;
    case RSX_BE_FMT_BC2:             return DXGI_FORMAT_BC2_UNORM;
    case RSX_BE_FMT_BC3:             return DXGI_FORMAT_BC3_UNORM;
    case RSX_BE_FMT_R16:             return DXGI_FORMAT_R16_UNORM;
    case RSX_BE_FMT_R16G16:          return DXGI_FORMAT_R16G16_UNORM;
    case RSX_BE_FMT_R16G16F:         return DXGI_FORMAT_R16G16_FLOAT;
    case RSX_BE_FMT_R16G16B16A16F:   return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case RSX_BE_FMT_R32F:            return DXGI_FORMAT_R32_FLOAT;
    case RSX_BE_FMT_R32G32B32A32F:   return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:                         return DXGI_FORMAT_R8_UNORM;
    }
}

static u32 eng_bpp(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8_UNORM: return 1;
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R16_UNORM: return 2;
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R32_FLOAT: return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    default: return 4;
    }
}

/* The crossbar's selectors arrive in A,R,G,B order; RSX_REMAP_ZERO/ONE are
 * 4 and 5, which are D3D12's FORCE_VALUE_0/1. */
static UINT eng_mapping(u32 remap, u32 rsx_fmt)
{
    u8 sel[4];
    rsx_texture_component_remap(remap, rsx_fmt & 0x9Fu, sel);
    return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(sel[1], sel[2], sel[3], sel[0]);
}

static D3D12_COMPARISON_FUNC gcm_cmp(u32 f)
{
    switch (f) {
    case 0x0200: return D3D12_COMPARISON_FUNC_NEVER;
    case 0x0201: return D3D12_COMPARISON_FUNC_LESS;
    case 0x0202: return D3D12_COMPARISON_FUNC_EQUAL;
    case 0x0203: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    case 0x0204: return D3D12_COMPARISON_FUNC_GREATER;
    case 0x0205: return D3D12_COMPARISON_FUNC_NOT_EQUAL;
    case 0x0206: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    default:     return D3D12_COMPARISON_FUNC_ALWAYS;
    }
}
static D3D12_BLEND gcm_blend_factor(u32 f, int alpha)
{
    switch (f & 0xFFFFu) {
    case 0x0000: return D3D12_BLEND_ZERO;
    case 0x0001: return D3D12_BLEND_ONE;
    case 0x0300: return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
    case 0x0301: return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
    case 0x0302: return D3D12_BLEND_SRC_ALPHA;
    case 0x0303: return D3D12_BLEND_INV_SRC_ALPHA;
    case 0x0304: return D3D12_BLEND_DEST_ALPHA;
    case 0x0305: return D3D12_BLEND_INV_DEST_ALPHA;
    case 0x0306: return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
    case 0x0307: return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
    case 0x0308: return D3D12_BLEND_SRC_ALPHA_SAT;
    case 0x8001: return D3D12_BLEND_BLEND_FACTOR;
    case 0x8002: return D3D12_BLEND_INV_BLEND_FACTOR;
    case 0x8003: return D3D12_BLEND_BLEND_FACTOR;
    case 0x8004: return D3D12_BLEND_INV_BLEND_FACTOR;
    default:     return D3D12_BLEND_ONE;
    }
}
static D3D12_BLEND_OP gcm_blend_op(u32 e)
{
    switch (e & 0xFFFFu) {
    case 0x8007: return D3D12_BLEND_OP_MIN;
    case 0x8008: return D3D12_BLEND_OP_MAX;
    case 0x800A: return D3D12_BLEND_OP_SUBTRACT;
    case 0x800B: return D3D12_BLEND_OP_REV_SUBTRACT;
    default:     return D3D12_BLEND_OP_ADD;
    }
}
static D3D12_STENCIL_OP gcm_stencil_op(u32 op)
{
    switch (op) {
    case 0x0000: return D3D12_STENCIL_OP_ZERO;
    case 0x1E01: return D3D12_STENCIL_OP_REPLACE;
    case 0x1E02: return D3D12_STENCIL_OP_INCR_SAT;
    case 0x1E03: return D3D12_STENCIL_OP_DECR_SAT;
    case 0x150A: return D3D12_STENCIL_OP_INVERT;
    case 0x8507: return D3D12_STENCIL_OP_INCR;
    case 0x8508: return D3D12_STENCIL_OP_DECR;
    default:     return D3D12_STENCIL_OP_KEEP;
    }
}
static D3D12_TEXTURE_ADDRESS_MODE gcm_wrap(u32 w)
{
    switch (w & 0xFu) {
    case 1:  return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case 2:  return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case 4:  return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case 6: case 7: case 8: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}
static D3D_PRIMITIVE_TOPOLOGY topo_d3d(rsx_topology t)
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS:         return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case RSX_TOPOLOGY_LINES:          return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case RSX_TOPOLOGY_LINE_STRIP:     return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case RSX_TOPOLOGY_TRIANGLE_STRIP: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    default:                          return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }
}
static int topo_class(rsx_topology t)   /* index into EngPipeline.pso */
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS: return 0;
    case RSX_TOPOLOGY_LINES: case RSX_TOPOLOGY_LINE_STRIP: return 1;
    default: return 2;
    }
}

static D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(ID3D12DescriptorHeap* heap, u32 step, u32 slot)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    CALL(heap, GetCPUDescriptorHandleForHeapStart, &h);
    h.ptr += (SIZE_T)slot * step;
    return h;
}
static D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(ID3D12DescriptorHeap* heap, u32 step, u32 slot)
{
    D3D12_GPU_DESCRIPTOR_HANDLE h;
    CALL(heap, GetGPUDescriptorHandleForHeapStart, &h);
    h.ptr += (UINT64)slot * step;
    return h;
}
static D3D12_CPU_DESCRIPTOR_HANDLE obj_srv(u32 handle) { return cpu_handle(s_srv_cpu, s_srv_step, handle ? handle - 1 : ENG_SRV_NULL); }
static D3D12_CPU_DESCRIPTOR_HANDLE obj_rtv(u32 handle) { return cpu_handle(s_rtv_cpu, s_rtv_step, handle - 1); }
static D3D12_CPU_DESCRIPTOR_HANDLE obj_dsv(u32 handle) { return cpu_handle(s_dsv_cpu, s_dsv_step, handle - 1); }

static EngObj* eng_obj(u32 handle)
{
    if (!handle || handle > s_obj_count) return NULL;
    EngObj* o = &s_obj[handle - 1];
    return o->kind ? o : NULL;
}
/* The object that owns the memory: a view resolves to its surface. */
static EngObj* eng_owner(u32 handle)
{
    EngObj* o = eng_obj(handle);
    if (o && o->kind == OBJ_VIEW) o = eng_obj(o->alias);
    return o && o->res ? o : NULL;
}
/* Whether a target (or the surface a view aliases) is at the internal
 * resolution: then its pass's guest-pixel coordinates are scaled. */
static int eng_obj_scaled(u32 handle)
{
    const EngObj* o = eng_owner(handle);
    return o && o->gw && o->gh && (o->w != o->gw || o->h != o->gh);
}

static ID3D12Resource* make_buffer(D3D12_HEAP_TYPE type, u64 bytes, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = type;
    D3D12_RESOURCE_DESC bd = {0};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bytes; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = NULL;
    if (FAILED(CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &bd, st, NULL,
                    &IID_ID3D12Resource, (void**)&r)))
        return NULL;
    return r;
}

static ID3D12Resource* make_texture(DXGI_FORMAT fmt, u32 w, u32 h, u32 mips, u32 faces,
                                    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES st,
                                    const D3D12_CLEAR_VALUE* cv)
{
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = (UINT16)(faces ? faces : 1);
    td.MipLevels = (UINT16)(mips ? mips : 1); td.Format = fmt; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; td.Flags = flags;
    ID3D12Resource* r = NULL;
    HRESULT hr = CALL(s_dev, CreateCommittedResource, &hp, D3D12_HEAP_FLAG_NONE, &td, st, cv,
                      &IID_ID3D12Resource, (void**)&r);
    if (FAILED(hr)) {
        static int n = 0;
        if (n++ < 16) fprintf(stderr, "[rsx engine/d3d12] texture %ux%u fmt %d failed: 0x%08lX\n", w, h, (int)fmt, (long)hr);
        return NULL;
    }
    return r;
}

/* Bounded fence wait: a device removal leaves the fence unsignaled forever,
 * and an INFINITE wait here would freeze the whole emulation (see
 * rsx_d3d12_backend.c's wait_for_gpu). */
/* RSX_SYNC_STATS=1: every 5 s, the waits for the GPU that actually blocked,
 * by the source line that waited, with their count and time -- the walker's
 * stalls on the real GPU, which the render thread waits behind. */
static int s_sync_stats = -1;
static struct { int line; unsigned long long n; double ms; } s_sync_rec[32];
static void sync_stats_add(int line, double ms)
{
    int i;
    for (i = 0; i < 32 && s_sync_rec[i].line && s_sync_rec[i].line != line; i++) {}
    if (i == 32) return;
    s_sync_rec[i].line = line; s_sync_rec[i].n++; s_sync_rec[i].ms += ms;
    static ULONGLONG last;
    const ULONGLONG now = GetTickCount64();
    if (!last) last = now;
    if (now - last >= 5000) {
        const double secs = (now - last) / 1000.0;
        last = now;
        fprintf(stderr, "[sync-stats] GPU waits that blocked, per second:");
        for (int k = 0; k < 32 && s_sync_rec[k].line; k++) {
            if (s_sync_rec[k].n)
                fprintf(stderr, " line %d %.0f/s %.1f ms/s;", s_sync_rec[k].line,
                        s_sync_rec[k].n / secs, s_sync_rec[k].ms / secs);
            s_sync_rec[k].n = 0; s_sync_rec[k].ms = 0;
        }
        fputc('\n', stderr);
    }
}

extern double g_rsx_frame_gpu_wait_ms;   /* rsx_draw_engine.c: blocking GPU waits since the last present */
static void fence_wait_at(u64 v, int line)
{
    if (!v || CALL0(s_fence, GetCompletedValue) >= v) return;
    if (s_sync_stats < 0) s_sync_stats = getenv("RSX_SYNC_STATS") ? 1 : 0;
    static int hl = -1;
    if (hl < 0) hl = getenv("RSX_HITCH_LOG") ? 1 : 0;
    LARGE_INTEGER qf, q0, q1;
    if (s_sync_stats || hl) { QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&q0); }
    CALL(s_fence, SetEventOnCompletion, v, s_fence_event);
    for (int tries = 0; tries < 5; tries++) {
        if (WaitForSingleObject(s_fence_event, 2000) != WAIT_TIMEOUT) {
            if (s_sync_stats || hl) {
                QueryPerformanceCounter(&q1);
                const double ms = (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)qf.QuadPart;
                if (s_sync_stats) sync_stats_add(line, ms);
                g_rsx_frame_gpu_wait_ms += ms;
            }
            return;
        }
        HRESULT rr = CALL0(s_dev, GetDeviceRemovedReason);
        fprintf(stderr, "[rsx engine/d3d12] fence %llu stuck %ds (completed %llu, removed=0x%08lX)\n",
                (unsigned long long)v, 2 * (tries + 1),
                (unsigned long long)CALL0(s_fence, GetCompletedValue), (long)rr);
        if (rr != S_OK) { s_ready = 0; return; }
    }
}
#define fence_wait(v) fence_wait_at((v), __LINE__)
static int fence_done(u64 v) { return !v || CALL0(s_fence, GetCompletedValue) >= v; }
static u64 fence_signal(void)
{
    const u64 v = ++s_fence_value;
    CALL(s_queue, Signal, s_fence, v);
    return v;
}

/* ---- object table ---------------------------------------------------------- */

static u32 eng_obj_add(ID3D12Resource* res, EngObjKind kind, DXGI_FORMAT fmt, u32 w, u32 h,
                       u32 mips, u32 faces, D3D12_RESOURCE_STATES st)
{
    if (!res && kind != OBJ_VIEW) return 0;
    u32 slot;
    if (s_obj_free_count) slot = s_obj_free[--s_obj_free_count];
    else {
        if (s_obj_count >= ENG_MAX_OBJECTS) {
            static unsigned long n = 0;
            if (n++ % 10000 == 0)
                fprintf(stderr, "[rsx engine/d3d12] object table full (%u): texture creation FAILS (#%lu)\n", s_obj_count, n);
            if (res) res->lpVtbl->Release(res);
            return 0;
        }
        slot = s_obj_count++;
    }
    EngObj* o = &s_obj[slot];
    memset(o, 0, sizeof *o);
    o->res = res; o->kind = kind; o->fmt = fmt; o->w = w; o->h = h;
    o->gw = w; o->gh = h;       /* a scaled object sets its guest size after */
    o->mips = mips ? mips : 1; o->faces = faces ? faces : 1; o->state = st;
    return slot + 1;
}

/* Retired objects are released only once no submitted list can still name
 * them: every submit the GPU may be running must have completed. Records
 * resolve handles at encode time, so nothing is recycled while records wait. */
static u32 s_retired[ENG_MAX_OBJECTS]; static u32 s_retired_count;
static u64 s_retired_fence[ENG_MAX_OBJECTS];

static void eng_collect_retired(void)
{
    u32 k = 0;
    for (u32 i = 0; i < s_retired_count; i++) {
        const u32 h = s_retired[i];
        if (s_rec_count || !fence_done(s_retired_fence[i])) { s_retired[k] = h; s_retired_fence[k] = s_retired_fence[i]; k++; continue; }
        EngObj* o = &s_obj[h - 1];
        if (o->kind == OBJ_SNAPSHOT && o->has_rtv && s_snap_pool_n < ENG_SNAP_POOL) {
            s_snap_pool[s_snap_pool_n].res = o->res; s_snap_pool[s_snap_pool_n].fmt = o->fmt;
            s_snap_pool[s_snap_pool_n].w = o->w; s_snap_pool[s_snap_pool_n].h = o->h;
            s_snap_pool[s_snap_pool_n].fence = s_retired_fence[i];
            s_snap_pool[s_snap_pool_n].state = o->state;
            s_snap_pool_n++;
            o->res = NULL;
        }
        RELEASE(o->res);
        memset(o, 0, sizeof *o);
        s_obj_free[s_obj_free_count++] = h - 1;
    }
    s_retired_count = k;
}

static void eng_obj_release(void* user, u32 handle)
{
    (void)user;
    EngObj* o = eng_obj(handle);
    if (!o || o->retired) return;
    o->retired = 1;
    /* Any view cut from this object stops being valid with it. */
    for (u32 i = 0; i < s_view_count; i++)
        if (s_view[i].surface == handle) {
            eng_obj_release(NULL, s_view[i].view);
            s_view[i] = s_view[--s_view_count];
            i--;
        }
    if (s_retired_count < ENG_MAX_OBJECTS) {
        s_retired[s_retired_count] = handle;
        s_retired_fence[s_retired_count] = s_fence_value;   /* the last signalled submit */
        s_retired_count++;
    }
}

/* Write the object's SRV at its CPU slot. */
static void eng_write_srv(u32 handle, ID3D12Resource* res, DXGI_FORMAT fmt, u32 mips, u32 faces, UINT mapping)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
    sd.Format = fmt;
    sd.Shader4ComponentMapping = mapping;
    if (faces == 6) {
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        sd.TextureCube.MipLevels = mips;
    } else {
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = mips;
    }
    CALL(s_dev, CreateShaderResourceView, res, &sd, obj_srv(handle));
}

/* ---- staging --------------------------------------------------------------- */

static void eng_submit(u32 present_surface, int wait);

static int eng_stage_pick(void)
{
    /* A free arena at least as large as the largest a submit grew to. */
    int best = -1;
    for (int i = 0; i < ENG_STAGE_POOL; i++) {
        EngStage* s = &s_stage[i];
        if (s->in_use && fence_done(s->fence)) s->in_use = 0;
        if (s->in_use) continue;
        if (s->res && s->cap >= s_stage_want && (best < 0 || s->cap < s_stage[best].cap)) best = i;
    }
    if (best >= 0) return best;
    for (int i = 0; i < ENG_STAGE_POOL; i++) if (!s_stage[i].in_use && !s_stage[i].res) { best = i; break; }
    if (best < 0) {   /* every slot is in use or too small: replace the smallest free one */
        for (int i = 0; i < ENG_STAGE_POOL; i++)
            if (!s_stage[i].in_use && (best < 0 || s_stage[i].cap < s_stage[best].cap)) best = i;
        if (best < 0) return -1;
        if (s_stage[best].res) { CALL(s_stage[best].res, Unmap, 0, NULL); RELEASE(s_stage[best].res); }
        s_stage[best].mapped = NULL; s_stage[best].cap = 0;
    }
    EngStage* s = &s_stage[best];
    s->res = make_buffer(D3D12_HEAP_TYPE_UPLOAD, s_stage_want, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!s->res) return -1;
    D3D12_RANGE nr = {0, 0};
    if (FAILED(CALL(s->res, Map, 0, &nr, (void**)&s->mapped))) { RELEASE(s->res); return -1; }
    s->cap = s_stage_want;
    return best;
}

static int eng_stage_reserve(u32 bytes, u32* out_off)
{
    const u32 start = (s_stage_used + ENG_ALIGN - 1u) & ~(ENG_ALIGN - 1u);
    if ((u64)start + bytes > ENG_STAGE_MAX) {
        /* Submit and wait rather than drop the rest of the frame. */
        eng_submit(0, 1);
        return eng_stage_reserve(bytes, out_off);
    }
    if (s_stage_cur < 0) {
        s_stage_cur = eng_stage_pick();
        if (s_stage_cur < 0) return 0;
        s_stage_used = 0;
    }
    EngStage* s = &s_stage[s_stage_cur];
    if (start + bytes > s->cap) {
        /* Grow: nothing has been submitted from this arena, so the staged
         * bytes move over and the old buffer is let go. */
        u32 cap = s->cap ? s->cap : ENG_STAGE_START;
        while (start + bytes > cap) cap *= 2u;
        ID3D12Resource* n = make_buffer(D3D12_HEAP_TYPE_UPLOAD, cap, D3D12_RESOURCE_STATE_GENERIC_READ);
        if (!n) return 0;
        u8* nm = NULL; D3D12_RANGE nr = {0, 0};
        if (FAILED(CALL(n, Map, 0, &nr, (void**)&nm))) { RELEASE(n); return 0; }
        if (s_stage_used) memcpy(nm, s->mapped, s_stage_used);
        CALL(s->res, Unmap, 0, NULL); RELEASE(s->res);
        s->res = n; s->mapped = nm; s->cap = cap;
        if (cap > s_stage_want) s_stage_want = cap;
    }
    *out_off = start;
    s_stage_used = start + bytes;
    return 1;
}

static int eng_stage_copy(const void* src, u32 bytes, u32* out_off)
{
    if (!bytes) { *out_off = 0; return 1; }
    if (!eng_stage_reserve(bytes, out_off)) return 0;
    memcpy(s_stage[s_stage_cur].mapped + *out_off, src, bytes);
    return 1;
}

/* ---- buffers the engine keeps across frames (vertex cache) ------------------ */

/* Retired vertex buffers are kept and reused, by size class (powers of two
 * from 64 KiB). Creating a committed resource is a kernel allocation
 * (NtGdiDdDDICreateAllocation) and releasing one a kernel free, each a
 * millisecond or more on the thread that walks the FIFO; Drakengard 3's
 * vertex cache turns over a few hundred of these a minute, and the slow
 * frames at 60 fps caught the walker inside those calls. The pool is bounded;
 * past it, buffers are released as before. */
#define ENG_BUF_POOL        256
#define ENG_BUF_POOL_BYTES  ((u64)512 << 20)
static struct { ID3D12Resource* res; u32 cap; } s_buf_pool[ENG_BUF_POOL];
static u32 s_buf_pool_n; static u64 s_buf_pool_bytes;
static u32 s_buf_cap[ENG_MAX_BUFS];
static u32 buf_class(u32 bytes) { u32 c = 65536u; while (c < bytes) c <<= 1; return c; }
int g_eng_buf_pool = 1;   /* RSX_BUF_POOL=0 turns the pool off; a host may switch it in a run */

static u32 eng_buffer_wrap(void* user, void* data, u32 bytes)
{
    (void)user;
    if (!s_dev || !data || !bytes) return 0;
    u32 slot;
    if (s_buf_free_count) slot = s_buf_free[s_buf_free_count - 1];
    else if (s_buf_count < ENG_MAX_BUFS) slot = s_buf_count;
    else return 0;
    /* D3D12 cannot read the engine's memory in place; it is copied into an
     * upload-heap buffer and, ownership being ours, freed here. */
    const u32 cap = buf_class(bytes);
    ID3D12Resource* b = NULL;
    for (u32 i = 0; i < s_buf_pool_n; i++) {
        if (s_buf_pool[i].cap != cap) continue;
        b = s_buf_pool[i].res;
        s_buf_pool_bytes -= cap;
        s_buf_pool[i] = s_buf_pool[--s_buf_pool_n];
        break;
    }
    if (!b) b = make_buffer(D3D12_HEAP_TYPE_UPLOAD, cap, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!b) return 0;
    s_buf_cap[slot] = cap;
    void* m = NULL; D3D12_RANGE nr = {0, 0};
    if (FAILED(CALL(b, Map, 0, &nr, &m))) { RELEASE(b); return 0; }
    memcpy(m, data, bytes);
    CALL(b, Unmap, 0, NULL);
    _aligned_free(data);
    s_stat_buf++;
    if (s_buf_free_count) s_buf_free_count--; else s_buf_count++;
    s_buf[slot] = b; s_buf_bytes[slot] = bytes; s_buf_retired[slot] = 0;
    return slot + 1;
}

static u64 s_buf_retired_fence[ENG_MAX_BUFS];
static void eng_collect_retired_buffers(void)
{
    if (s_rec_count) return;
    for (u32 i = 0; i < s_buf_count; i++) {
        if (!s_buf_retired[i] || !fence_done(s_buf_retired_fence[i])) continue;
        s_buf_retired[i] = 0;
        if (g_eng_buf_pool && s_buf_pool_n < ENG_BUF_POOL && s_buf_pool_bytes + s_buf_cap[i] <= ENG_BUF_POOL_BYTES) {
            s_buf_pool[s_buf_pool_n].res = s_buf[i]; s_buf_pool[s_buf_pool_n].cap = s_buf_cap[i];
            s_buf_pool_n++; s_buf_pool_bytes += s_buf_cap[i];
            s_buf[i] = NULL;
        } else {
            RELEASE(s_buf[i]);
        }
        s_buf_free[s_buf_free_count++] = i;
    }
}
static void eng_buffer_release(void* user, u32 h)
{
    (void)user;
    if (!h || h > s_buf_count || !s_buf[h - 1] || s_buf_retired[h - 1]) return;
    s_buf_retired[h - 1] = 1;
    s_buf_retired_fence[h - 1] = s_fence_value;
}

/* ---- window ---------------------------------------------------------------- */

static LRESULT CALLBACK eng_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_CLOSE:   s_window_closed = 1; DestroyWindow(hwnd); return 0;
    case WM_ACTIVATEAPP:
        if (wp && s_display == DISP_FULLSCREEN) s_fs_restore = 1;
        break;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) { s_window_closed = 1; DestroyWindow(hwnd); }
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

static HWND eng_create_window(u32 width, u32 height, const char* title)
{
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = eng_wndproc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    /* The executable's icon resource (id 101 in the port's app.rc), when it
     * has one; the default application icon otherwise. */
    wc.hIcon = LoadIconA(GetModuleHandle(NULL), MAKEINTRESOURCEA(101));
    wc.hIconSm = wc.hIcon;
    wc.lpszClassName = "ps3recomp_d3d12_engine";
    RegisterClassExA(&wc);
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    if (s_display != DISP_WINDOWED) {
        /* A popup over the display: for borderless that is the whole of it
         * (no mode switch, so alt-tab and the flip model behave as in a
         * window); for exclusive full screen DXGI takes it from here. */
        return CreateWindowExA(0, "ps3recomp_d3d12_engine", title ? title : "ps3recomp (D3D12)",
                               WS_POPUP | WS_VISIBLE, 0, 0, (int)width, (int)height,
                               NULL, NULL, GetModuleHandle(NULL), NULL);
    }
    RECT wr = {0, 0, (LONG)width, (LONG)height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExA(0, "ps3recomp_d3d12_engine", title ? title : "ps3recomp (D3D12)",
                           WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                           wr.right - wr.left, wr.bottom - wr.top,
                           NULL, NULL, GetModuleHandle(NULL), NULL);
}

/* ---- helper shaders: present blit, depth resolve, depth pack --------------- */

static const char kHelperHLSL[] =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut vs_main(uint vid : SV_VertexID) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
    "}\n"
    "Texture2D<float4> src : register(t0);\n"
    "SamplerState smp : register(s0);\n"
    "float4 ps_blit(VSOut i) : SV_Target { return src.Sample(smp, i.uv); }\n"
    /* Depth copies are texel for texel from the top-left corner: a copy
     * smaller than its depth target is the region the title's texture covers. */
    "Texture2D<float> dsrc : register(t0);\n"
    "float ps_depth(VSOut i) : SV_Target { return dsrc.Load(int3(i.pos.xy, 0)); }\n"
    /* The depth as the RSX's D24S8 word reads through an A8R8G8B8 texture:
     * A = depth[23:16], R = depth[15:8], G = depth[7:0], B = stencil (0). */
    "float4 ps_pack(VSOut i) : SV_Target {\n"
    "    uint v = uint(saturate(dsrc.Load(int3(i.pos.xy, 0))) * 16777215.0 + 0.5);\n"
    "    return float4(float((v >> 8) & 255u), float(v & 255u), 0.0, float((v >> 16) & 255u)) / 255.0;\n"
    "}\n";

static ID3DBlob* eng_compile(const char* src, u32 len, const char* entry, const char* target, const char* what)
{
    ID3DBlob* b = NULL; ID3DBlob* e = NULL;
    HRESULT hr = D3DCompile(src, len, what, NULL, NULL, entry, target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &b, &e);
    if (FAILED(hr)) {
        static int n = 0;
        if (n++ < 32)
            fprintf(stderr, "[rsx engine/d3d12] %s compile failed: %.900s\n", what,
                    e ? (const char*)CALL0(e, GetBufferPointer) : "no diagnostic");
        RELEASE(e);
        return NULL;
    }
    RELEASE(e);
    return b;
}

static ID3D12PipelineState* eng_helper_pso(ID3DBlob* vs, ID3DBlob* ps, DXGI_FORMAT rt)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_helper_rootsig;
    pd.VS.pShaderBytecode = CALL0(vs, GetBufferPointer); pd.VS.BytecodeLength = CALL0(vs, GetBufferSize);
    pd.PS.pShaderBytecode = CALL0(ps, GetBufferPointer); pd.PS.BytecodeLength = CALL0(ps, GetBufferSize);
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1; pd.RTVFormats[0] = rt;
    pd.SampleDesc.Count = 1;
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&pso);
    if (FAILED(hr)) fprintf(stderr, "[rsx engine/d3d12] helper pipeline failed: 0x%08lX\n", (long)hr);
    return pso;
}

static int eng_make_root_signatures(void)
{
    /* The draw root signature, matching the decompilers' register plan:
     *   b0 VS constants (VPConst), b1 PS constants (PSConstants),
     *   t0-t15 / s0-s15 fragment units, t16-t19 / s0-s3 vertex units. */
    D3D12_DESCRIPTOR_RANGE r_ps_srv = {0}, r_ps_smp = {0}, r_vs_srv = {0}, r_vs_smp = {0};
    r_ps_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;     r_ps_srv.NumDescriptors = RSX_BE_MAX_TEXTURES;
    r_ps_smp.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; r_ps_smp.NumDescriptors = RSX_BE_MAX_TEXTURES;
    r_vs_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;     r_vs_srv.NumDescriptors = RSX_BE_MAX_VERTEX_TEXTURES; r_vs_srv.BaseShaderRegister = 16;
    r_vs_smp.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; r_vs_smp.NumDescriptors = RSX_BE_MAX_VERTEX_TEXTURES;
    D3D12_ROOT_PARAMETER p[6] = {0};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor.ShaderRegister = 0; p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[1].DescriptorTable.NumDescriptorRanges = 1; p[1].DescriptorTable.pDescriptorRanges = &r_ps_srv; p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[2].DescriptorTable.NumDescriptorRanges = 1; p[2].DescriptorTable.pDescriptorRanges = &r_ps_smp; p[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[3].DescriptorTable.NumDescriptorRanges = 1; p[3].DescriptorTable.pDescriptorRanges = &r_vs_srv; p[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[4].Descriptor.ShaderRegister = 1; p[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[5].DescriptorTable.NumDescriptorRanges = 1; p[5].DescriptorTable.pDescriptorRanges = &r_vs_smp; p[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC rd = {0};
    rd.NumParameters = 6; rd.pParameters = p;
    rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob* sig = NULL; ID3DBlob* err = NULL;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
        fprintf(stderr, "[rsx engine/d3d12] root signature: %s\n", err ? (const char*)CALL0(err, GetBufferPointer) : "?");
        RELEASE(err); return -1;
    }
    HRESULT hr = CALL(s_dev, CreateRootSignature, 0, CALL0(sig, GetBufferPointer), CALL0(sig, GetBufferSize),
                      &IID_ID3D12RootSignature, (void**)&s_rootsig);
    RELEASE(sig);
    if (FAILED(hr)) return -1;

    /* The helper passes: one texture, one point/clamp sampler. */
    D3D12_DESCRIPTOR_RANGE hr_srv = {0};
    hr_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; hr_srv.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER hp[1] = {0};
    hp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; hp[0].DescriptorTable.NumDescriptorRanges = 1;
    hp[0].DescriptorTable.pDescriptorRanges = &hr_srv; hp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC ss = {0};
    ss.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.MaxLOD = D3D12_FLOAT32_MAX; ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC hd = {0};
    hd.NumParameters = 1; hd.pParameters = hp; hd.NumStaticSamplers = 1; hd.pStaticSamplers = &ss;
    if (FAILED(D3D12SerializeRootSignature(&hd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) { RELEASE(err); return -1; }
    hr = CALL(s_dev, CreateRootSignature, 0, CALL0(sig, GetBufferPointer), CALL0(sig, GetBufferSize),
              &IID_ID3D12RootSignature, (void**)&s_helper_rootsig);
    RELEASE(sig);
    return FAILED(hr) ? -1 : 0;
}

static ID3D12DescriptorHeap* eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE type, u32 n, int visible)
{
    D3D12_DESCRIPTOR_HEAP_DESC d = {0};
    d.Type = type; d.NumDescriptors = n;
    d.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ID3D12DescriptorHeap* h = NULL;
    if (FAILED(CALL(s_dev, CreateDescriptorHeap, &d, &IID_ID3D12DescriptorHeap, (void**)&h))) return NULL;
    return h;
}

/* ---- device ---------------------------------------------------------------- */

/* The swap chain over s_hwnd at s_win_w x s_win_h for s_display: flip
 * model, three buffers; tearing allowed whenever the system supports it
 * outside exclusive full screen, so v-sync can be switched off while
 * running. At start-up and again when the display settings change. */
static int eng_swapchain_create(IDXGIFactory4* factory)
{
    HRESULT hr;
    DXGI_SWAP_CHAIN_DESC1 sd = {0};
    sd.Width = s_win_w; sd.Height = s_win_h; sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 3; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    s_tearing = 0;
    if (s_display != DISP_FULLSCREEN) {
        IDXGIFactory5* f5 = NULL;
        if (SUCCEEDED(CALL(factory, QueryInterface, &IID_IDXGIFactory5, (void**)&f5))) {
            BOOL allow = FALSE;
            if (SUCCEEDED(CALL(f5, CheckFeatureSupport, DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof allow)) && allow)
                s_tearing = 1;
            RELEASE(f5);
        }
        if (s_tearing) sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    }
    IDXGISwapChain1* sc1 = NULL;
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC fsd = {0};
    fsd.Windowed = TRUE;
    if (s_display == DISP_FULLSCREEN) {
        /* Exclusive: the swap chain owns the output, at the mode its
         * size names. ALLOW_MODE_SWITCH lets that be a different mode
         * from the desktop's (RSX_WINDOW). */
        sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        fsd.Windowed = FALSE;
        fsd.RefreshRate.Numerator = 0; fsd.RefreshRate.Denominator = 0;
        fsd.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
        fsd.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    }
    hr = CALL(factory, CreateSwapChainForHwnd, (IUnknown*)s_queue, s_hwnd, &sd, &fsd, NULL, &sc1);
    if (FAILED(hr) && s_display == DISP_FULLSCREEN) {
        /* The output refused the mode: fall back to a borderless window
         * of the same size rather than no window at all. */
        fprintf(stderr, "[rsx engine/d3d12] exclusive full screen refused (0x%08lX); borderless instead\n", (long)hr);
        s_display = DISP_BORDERLESS;
        sd.Flags &= ~(UINT)DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        {
            IDXGIFactory5* f5 = NULL; BOOL allow = FALSE;
            if (SUCCEEDED(CALL(factory, QueryInterface, &IID_IDXGIFactory5, (void**)&f5))) {
                if (SUCCEEDED(CALL(f5, CheckFeatureSupport, DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof allow)) && allow)
                    s_tearing = 1;
                RELEASE(f5);
            }
            if (s_tearing) sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        }
        hr = CALL(factory, CreateSwapChainForHwnd, (IUnknown*)s_queue, s_hwnd, &sd, NULL, NULL, &sc1);
    }
    if (FAILED(hr)) { fprintf(stderr, "[rsx engine/d3d12] swap chain failed: 0x%08lX\n", (long)hr); return -1; }
    CALL(factory, MakeWindowAssociation, s_hwnd, DXGI_MWA_NO_ALT_ENTER);
    hr = CALL(sc1, QueryInterface, &IID_IDXGISwapChain3, (void**)&s_swap);
    RELEASE(sc1);
    if (FAILED(hr)) return -1;
    if (!s_backbuf_rtv_heap) s_backbuf_rtv_heap = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, 0);
    if (!s_backbuf_rtv_heap) return -1;
    s_rtv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (u32 i = 0; i < 3; i++) {
        if (FAILED(CALL(s_swap, GetBuffer, i, &IID_ID3D12Resource, (void**)&s_backbuf[i]))) return -1;
        CALL(s_dev, CreateRenderTargetView, s_backbuf[i], NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, i));
    }
    return 0;
}

static int eng_init_device(u32 width, u32 height)
{
    HRESULT hr;
    if (getenv("D3D12_DBG")) {
        ID3D12Debug* dbg = NULL;
        if (SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&dbg)) && dbg) {
            CALL0(dbg, EnableDebugLayer); RELEASE(dbg);
            fprintf(stderr, "[rsx engine/d3d12] debug layer enabled\n");
        }
    }
    IDXGIFactory4* factory = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory))) return -1;
    {
        IDXGIFactory6* f6 = NULL;
        if (SUCCEEDED(CALL(factory, QueryInterface, &IID_IDXGIFactory6, (void**)&f6))) {
            DXGI_GPU_PREFERENCE pref = getenv("CELLMARK_IGPU") ? DXGI_GPU_PREFERENCE_MINIMUM_POWER
                                                                : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
            IDXGIAdapter1* ad = NULL;
            for (UINT i = 0; CALL(f6, EnumAdapterByGpuPreference, i, pref, &IID_IDXGIAdapter1, (void**)&ad) != DXGI_ERROR_NOT_FOUND; i++) {
                DXGI_ADAPTER_DESC1 desc; CALL(ad, GetDesc1, &desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                    SUCCEEDED(D3D12CreateDevice((IUnknown*)ad, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&s_dev))) {
                    fprintf(stderr, "[rsx engine/d3d12] adapter: %ls (%llu MB)\n", desc.Description,
                            (unsigned long long)(desc.DedicatedVideoMemory >> 20));
                    RELEASE(ad); break;
                }
                RELEASE(ad);
            }
            RELEASE(f6);
        }
    }
    if (!s_dev && FAILED(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&s_dev))) {
        RELEASE(factory); return -1;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {0}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(CALL(s_dev, CreateCommandQueue, &qd, &IID_ID3D12CommandQueue, (void**)&s_queue))) { RELEASE(factory); return -1; }

    if (!s_headless && eng_swapchain_create(factory) != 0) { RELEASE(factory); return -1; }
    RELEASE(factory);

    if (FAILED(CALL(s_dev, CreateFence, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&s_fence))) return -1;
    s_fence_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    for (u32 i = 0; i < ENG_FRAMES; i++)
        if (FAILED(CALL(s_dev, CreateCommandAllocator, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&s_alloc[i]))) return -1;
    if (FAILED(CALL(s_dev, CreateCommandList, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_alloc[0], NULL,
                    &IID_ID3D12GraphicsCommandList, (void**)&s_list))) return -1;
    CALL0(s_list, Close); s_list_open = 0;

    s_srv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s_rtv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    s_dsv_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    s_smp_step = CALL(s_dev, GetDescriptorHandleIncrementSize, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    s_srv_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ENG_MAX_OBJECTS + 1, 0);
    s_rtv_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, ENG_MAX_OBJECTS, 0);
    s_dsv_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, ENG_MAX_OBJECTS, 0);
    s_smp_cpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENG_MAX_SAMPLERS + 1, 0);
    s_srv_gpu = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ENG_FRAMES * ENG_SRV_PER_SLOT, 1);
    if (!s_srv_cpu || !s_rtv_cpu || !s_dsv_cpu || !s_smp_cpu || !s_srv_gpu) return -1;
    for (u32 i = 0; i < ENG_FRAMES; i++)
        if (!(s_smp_gpu[i] = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENG_SMP_PER_SLOT, 1))) return -1;

    /* Sampler slot 0: the default for an unbound unit. */
    {
        D3D12_SAMPLER_DESC sd = {0};
        sd.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.MaxLOD = D3D12_FLOAT32_MAX; sd.MaxAnisotropy = 1;
        CALL(s_dev, CreateSampler, &sd, cpu_handle(s_smp_cpu, s_smp_step, 0));
    }
    /* The null texture: 1x1 transparent black at CPU SRV slot ENG_SRV_NULL. */
    s_null_tex = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, D3D12_RESOURCE_FLAG_NONE, ENG_SHADER_READ, NULL);
    if (!s_null_tex) return -1;
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Texture2D.MipLevels = 1;
        CALL(s_dev, CreateShaderResourceView, s_null_tex, &sd, obj_srv(0));
    }
    if (!getenv("RSX_NO_QUERIES")) {
        D3D12_QUERY_HEAP_DESC qh = {0};
        qh.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION; qh.Count = ENG_FRAMES * ENG_QUERIES_PER_SLOT;
        if (FAILED(CALL(s_dev, CreateQueryHeap, &qh, &IID_ID3D12QueryHeap, (void**)&s_qheap))) s_qheap = NULL;
        if (s_qheap) s_qread = make_buffer(D3D12_HEAP_TYPE_READBACK, (u64)ENG_FRAMES * ENG_QUERIES_PER_SLOT * 8u, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s_qread) { RELEASE(s_qheap); }
    }
    if (getenv("RSX_GPU_TIME")) {
        D3D12_QUERY_HEAP_DESC th = {0};
        th.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; th.Count = ENG_FRAMES * 2;
        if (SUCCEEDED(CALL(s_dev, CreateQueryHeap, &th, &IID_ID3D12QueryHeap, (void**)&s_tsheap)))
            s_tsread = make_buffer(D3D12_HEAP_TYPE_READBACK, (u64)ENG_FRAMES * 16u, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s_tsread || FAILED(CALL(s_queue, GetTimestampFrequency, &s_ts_freq)) || !s_ts_freq) { RELEASE(s_tsheap); RELEASE(s_tsread); }
        if (s_tsheap && atoi(getenv("RSX_GPU_TIME")) >= 2) {
            D3D12_QUERY_HEAP_DESC ph = {0};
            ph.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; ph.Count = ENG_FRAMES * (ENG_PASS_MAX + 1u);
            if (SUCCEEDED(CALL(s_dev, CreateQueryHeap, &ph, &IID_ID3D12QueryHeap, (void**)&s_pheap)))
                s_pread = make_buffer(D3D12_HEAP_TYPE_READBACK, (u64)ENG_FRAMES * (ENG_PASS_MAX + 1u) * 8u, D3D12_RESOURCE_STATE_COPY_DEST);
            if (!s_pread) RELEASE(s_pheap);
        }
    }
    if (eng_make_root_signatures() != 0) return -1;
    s_snap_copy_ps = eng_compile(kSnapCopyHLSL, sizeof kSnapCopyHLSL - 1, "main", "ps_5_0", "snapshot copy ps");
    {
        ID3DBlob* vs = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "vs_main", "vs_5_0", "helper vs");
        ID3DBlob* pb = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_blit", "ps_5_0", "helper blit");
        ID3DBlob* pd = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_depth", "ps_5_0", "helper depth");
        ID3DBlob* pp = eng_compile(kHelperHLSL, sizeof kHelperHLSL - 1, "ps_pack", "ps_5_0", "helper pack");
        if (vs && pb) s_blit_pso = eng_helper_pso(vs, pb, DXGI_FORMAT_R8G8B8A8_UNORM);
        if (vs && pd) s_depth_pso = eng_helper_pso(vs, pd, DXGI_FORMAT_R32_FLOAT);
        if (vs && pp) s_depth_pack_pso = eng_helper_pso(vs, pp, DXGI_FORMAT_R8G8B8A8_UNORM);
        RELEASE(vs); RELEASE(pb); RELEASE(pd); RELEASE(pp);
        if (!s_blit_pso) return -1;
    }
    if (s_headless) {
        s_offscreen = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, 1,
                                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, NULL);
        if (!s_offscreen) return -1;
        s_backbuf_rtv_heap = eng_make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, 0);
        CALL(s_dev, CreateRenderTargetView, s_offscreen, NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, 0));
    }
    return 0;
}

/* ---- the engine backend: lifecycle ----------------------------------------- */

static int eng_init(void* user, u32 width, u32 height)
{
    (void)user; (void)width; (void)height;
    return s_dev ? 0 : -1;
}

static void eng_shutdown(void* user)
{
    (void)user;
    if (s_queue && s_fence) fence_wait(fence_signal());
    for (u32 i = 0; i < s_obj_count; i++) { RELEASE(s_obj[i].res); memset(&s_obj[i], 0, sizeof s_obj[i]); }
    s_obj_count = s_obj_free_count = s_retired_count = 0;
    for (u32 i = 0; i < s_pipe_count; i++) {
        RELEASE(s_pipe[i].vs); RELEASE(s_pipe[i].ps);
        for (int k = 0; k < 3; k++) RELEASE(s_pipe[i].pso[k]);
    }
    s_pipe_count = 0;
    for (u32 i = 0; i < s_blob_count; i++) RELEASE(s_blob[i].blob);
    s_blob_count = s_samp_count = s_view_count = s_rec_count = 0;
    for (u32 i = 0; i < s_buf_count; i++) RELEASE(s_buf[i]);
    for (u32 i = 0; i < s_buf_pool_n; i++) RELEASE(s_buf_pool[i].res);
    s_buf_pool_n = 0; s_buf_pool_bytes = 0;
    s_buf_count = s_buf_free_count = 0; memset(s_buf_retired, 0, sizeof s_buf_retired);
    for (int i = 0; i < ENG_STAGE_POOL; i++) {
        if (s_stage[i].res) { CALL(s_stage[i].res, Unmap, 0, NULL); RELEASE(s_stage[i].res); }
        memset(&s_stage[i], 0, sizeof s_stage[i]);
    }
    s_stage_cur = -1; s_stage_used = 0;
    for (int i = 0; i < ENG_MAX_SUBMITS; i++) { free(s_sub[i].q_vis); free(s_sub[i].reports); memset(&s_sub[i], 0, sizeof s_sub[i]); }
    s_vs_cb_seq = ~0u; s_vis_npending = 0; s_fallback_n = 0;
    for (u32 i = 0; i < s_snap_pool_n; i++) RELEASE(s_snap_pool[i].res);
    s_snap_pool_n = 0;
    s_active = 0;
}

/* ---- resources ------------------------------------------------------------- */

static u32 eng_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                              u32 mips, u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)user;
    if (!s_dev || !w || !h) return 0;
    const DXGI_FORMAT df = eng_dxgi(fmt);
    ID3D12Resource* t = make_texture(df, w, h, mips, faces, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, NULL);
    s_stat_tex++;
    const u32 handle = eng_obj_add(t, OBJ_TEXTURE, df, w, h, mips, faces, D3D12_RESOURCE_STATE_COPY_DEST);
    if (handle) eng_write_srv(handle, t, df, mips ? mips : 1, faces, eng_mapping(remap, rsx_fmt));
    return handle;
}

/* Rows go into the stage; the copy is a record, ordered with the draws. */
static void eng_upload_rows(u32 handle, u32 sub, u32 w, u32 h, const void* src, u32 row_bytes, u32 rows)
{
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    const u32 pitch = (row_bytes + 255u) & ~255u;
    u32 off;
    if (!eng_stage_reserve(pitch * rows, &off)) return;
    /* The reserve may have submitted, so the pointer is taken after it. */
    u8* dst = s_stage[s_stage_cur].mapped + off;
    if (pitch == row_bytes) memcpy(dst, src, (size_t)row_bytes * rows);
    else for (u32 y = 0; y < rows; y++) memcpy(dst + (size_t)y * pitch, (const u8*)src + (size_t)y * row_bytes, row_bytes);
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_UPLOAD;
    r->depth = handle;
    r->up_off = off; r->up_sub = sub; r->up_w = w; r->up_h = h; r->up_pitch = pitch;
}

static void eng_texture_upload(void* user, u32 handle, u32 face, u32 mip,
                               u32 w, u32 h, const void* src, u32 row_bytes, u32 rows)
{
    (void)user;
    EngObj* o = eng_obj(handle);
    if (!o || !o->res || !src || !row_bytes || !rows) return;
    eng_upload_rows(handle, face * o->mips + mip, w, h, src, row_bytes, rows);
}

static u32 eng_color_target_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                   const void* seed, u32 seed_row_bytes)
{
    (void)user;
    if (!s_dev || !w || !h) return 0;
    const DXGI_FORMAT df = eng_dxgi(fmt);
    const int scaled = eng_scales(w, h);
    const u32 sw = scaled ? sc_dim(w) : w, sh = scaled ? sc_dim(h) : h;
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = df;
    ID3D12Resource* t = make_texture(df, sw, sh, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                     D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
    const u32 handle = eng_obj_add(t, OBJ_COLOR, df, sw, sh, 1, 1, D3D12_RESOURCE_STATE_RENDER_TARGET);
    s_stat_rt++;
    if (!handle) return 0;
    s_obj[handle - 1].gw = w; s_obj[handle - 1].gh = h;
    eng_write_srv(handle, t, df, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    CALL(s_dev, CreateRenderTargetView, t, NULL, obj_rtv(handle));
    s_obj[handle - 1].has_rtv = 1;
    if (seed && seed_row_bytes) {
        if (sw == w && sh == h) eng_upload_rows(handle, 0, w, h, seed, seed_row_bytes, h);
        else {
            /* The guest's bytes, resampled (nearest) to the scaled target. */
            const u32 bpp = eng_bpp(df);
            u8* big = (u8*)malloc((size_t)sw * bpp * sh);
            if (big) {
                for (u32 y = 0; y < sh; y++) {
                    const u8* srow = (const u8*)seed + (size_t)((u64)y * h / sh) * seed_row_bytes;
                    u8* drow = big + (size_t)y * sw * bpp;
                    for (u32 x = 0; x < sw; x++)
                        memcpy(drow + (size_t)x * bpp, srow + (size_t)((u64)x * w / sw) * bpp, bpp);
                }
                eng_upload_rows(handle, 0, sw, sh, big, sw * bpp, sh);
                free(big);
            }
        }
    }
    return handle;
}

static u32 eng_surface_view(void* user, u32 surface, u32 remap, u32 rsx_format)
{
    (void)user;
    EngObj* o = eng_obj(surface);
    if (!o || !o->res) return 0;
    for (u32 i = 0; i < s_view_count; i++)
        if (s_view[i].surface == surface && s_view[i].remap == remap && s_view[i].format == rsx_format)
            return s_view[i].view;
    if (s_view_count >= ENG_MAX_VIEWS) return 0;
    const u32 handle = eng_obj_add(NULL, OBJ_VIEW, o->fmt, o->w, o->h, 1, 1, o->state);
    if (!handle) return 0;
    s_obj[handle - 1].alias = surface;
    eng_write_srv(handle, o->res, o->fmt, 1, 1, eng_mapping(remap, rsx_format));
    s_view[s_view_count].surface = surface; s_view[s_view_count].remap = remap;
    s_view[s_view_count].format = rsx_format; s_view[s_view_count].view = handle;
    s_view_count++;
    return handle;
}

static u32 eng_depth_target_create_host(u32 w, u32 h, u32 gw, u32 gh)
{
    if (!s_dev || !w || !h) return 0;
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = ENG_DEPTH_FMT; cv.DepthStencil.Depth = 1.0f;
    ID3D12Resource* t = make_texture(ENG_DEPTH_RES_FMT, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                                     D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv);
    const u32 handle = eng_obj_add(t, OBJ_DEPTH, ENG_DEPTH_FMT, w, h, 1, 1, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    if (!handle) return 0;
    s_obj[handle - 1].gw = gw; s_obj[handle - 1].gh = gh;
    D3D12_DEPTH_STENCIL_VIEW_DESC dd = {0};
    dd.Format = ENG_DEPTH_FMT; dd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    CALL(s_dev, CreateDepthStencilView, t, &dd, obj_dsv(handle));
    s_obj[handle - 1].has_dsv = 1;
    /* Sampled by the resolve passes as its depth plane. */
    eng_write_srv(handle, t, ENG_DEPTH_SRV_FMT, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    return handle;
}
static u32 eng_depth_target_create(void* user, u32 w, u32 h)
{
    (void)user;
    if (!eng_scales(w, h)) return eng_depth_target_create_host(w, h, w, h);
    return eng_depth_target_create_host(sc_dim(w), sc_dim(h), w, h);
}

/* A depth target for a draw pass whose guest never declared a zeta: every
 * pipeline declares the depth format, so a pass always needs one bound. */
static u32 eng_fallback_depth(u32 w, u32 h)
{
    for (u32 i = 0; i < s_fallback_n; i++) {
        EngObj* o = eng_obj(s_fallback_depth[i]);
        if (o && o->w == w && o->h == h) return s_fallback_depth[i];
    }
    const u32 d = eng_depth_target_create_host(w, h, w, h);   /* w, h are host pixels here */
    if (d && s_fallback_n < 8) s_fallback_depth[s_fallback_n++] = d;
    return d;
}

/* Copy of a colour target as it stands at this point in the stream. One copy
 * texture per source is kept and refreshed by each request. */
static u32 eng_color_snapshot(void* user, u32 surface)
{
    (void)user;
    EngObj* t = eng_obj(surface);
    if (!t || !t->res) return 0;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return 0; }
    static struct { u32 surface, snap; } pool[32]; static u32 npool;
    u32 dst = 0, slot = npool;
    for (u32 i = 0; i < npool; i++) if (pool[i].surface == surface) { slot = i; break; }
    if (slot < npool) {
        EngObj* d = eng_obj(pool[slot].snap);
        if (d && d->res && d->w == t->w && d->h == t->h && d->fmt == t->fmt) dst = pool[slot].snap;
    }
    if (!dst) {
        /* A render target as well, so the snapshot can be brought up to date
         * by drawing into it (eng_encode_copy_draw) rather than copied whole. */
        ID3D12Resource* r = make_texture(t->fmt, t->w, t->h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                         D3D12_RESOURCE_STATE_COPY_DEST, NULL);
        dst = eng_obj_add(r, OBJ_SNAPSHOT, t->fmt, t->w, t->h, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!dst) return 0;
        s_obj[dst - 1].gw = t->gw; s_obj[dst - 1].gh = t->gh;
        eng_write_srv(dst, r, t->fmt, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
        CALL(s_dev, CreateRenderTargetView, r, NULL, obj_rtv(dst));
        s_obj[dst - 1].has_rtv = 1;
        if (slot == npool) { if (npool >= 32) return 0; npool++; }
        pool[slot].surface = surface; pool[slot].snap = dst;
    }
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_COLOR_COPY;
    r->depth = surface;
    r->resolve_dst = dst;
    return dst;
}

static u32 eng_depth_snapshot_common(u32 depth, u32 w, u32 h, int packed)
{
    EngObj* z = eng_obj(depth);
    if (!z || !z->res || !(packed ? s_depth_pack_pso : s_depth_pso)) return 0;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return 0; }
    const u32 gw = w, gh = h;
    if (eng_obj_scaled(depth)) { w = sc_dim(w); h = sc_dim(h); }
    const DXGI_FORMAT df = packed ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R32_FLOAT;
    ID3D12Resource* r = NULL;
    D3D12_RESOURCE_STATES st = D3D12_RESOURCE_STATE_RENDER_TARGET;
    for (u32 i = 0; i < s_snap_pool_n; i++) {
        if (s_snap_pool[i].fmt != df || s_snap_pool[i].w != w || s_snap_pool[i].h != h) continue;
        r = s_snap_pool[i].res;
        st = s_snap_pool[i].state;
        s_snap_pool[i] = s_snap_pool[--s_snap_pool_n];
        s_stat_snap_reused++;
        break;
    }
    if (!r) {
        D3D12_CLEAR_VALUE cv = {0}; cv.Format = df;
        r = make_texture(df, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                         D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
        s_stat_snap++;
    }
    const u32 dst = eng_obj_add(r, OBJ_SNAPSHOT, df, w, h, 1, 1, st);
    if (!dst) return 0;
    s_obj[dst - 1].gw = gw; s_obj[dst - 1].gh = gh;
    eng_write_srv(dst, r, df, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    CALL(s_dev, CreateRenderTargetView, r, NULL, obj_rtv(dst));
    s_obj[dst - 1].has_rtv = 1;
    EngRecord* rec = &s_rec[s_rec_count++];
    memset(rec, 0, sizeof *rec);
    rec->kind = ENG_REC_DEPTH_RESOLVE;
    rec->depth = depth;
    rec->resolve_dst = dst;
    rec->resolve_packed = packed;
    return dst;
}
static u32 eng_depth_snapshot(void* user, u32 depth, u32 w, u32 h)
{ (void)user; return eng_depth_snapshot_common(depth, w, h, 0); }
static u32 eng_depth_snapshot_rgba8(void* user, u32 depth, u32 w, u32 h)
{ (void)user; return eng_depth_snapshot_common(depth, w, h, 1); }

/* ---- occlusion queries ----------------------------------------------------- */

static u32 eng_query_begin(void* user)
{
    (void)user;
    if (!s_qheap) return 0;
    const u32 slot = s_vis_next++ % ENG_VIS_SLOTS;
    s_vis_count[slot] = 0;
    return slot + 1;
}
static void eng_query_set(void* user, u32 query) { (void)user; s_vis_cur = query; }
static void eng_query_report(void* user, u32 query, u32 report_index)
{
    (void)user;
    if (!query || !s_qheap) return;
    if (s_vis_npending >= ENG_MAX_REPORTS) return;
    s_vis_pending[s_vis_npending].slot = query - 1;
    s_vis_pending[s_vis_npending].index = report_index;
    s_vis_npending++;
}

/* ---- pipelines ------------------------------------------------------------- */

static void dump_shader(const char* name, const void* text, size_t n)
{
    static const char* dir = (const char*)1;
    if (dir == (const char*)1) { dir = getenv("PS3RECOMP_SHADER_DUMP"); if (dir && !*dir) dir = NULL; }
    if (!dir) return;
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(text, 1, n, f);
    fclose(f);
}

/* Compiled DXBC, keyed on the HLSL the decompilers emitted: one program is
 * compiled once however many pipeline variants reference it.
 * PS3RECOMP_DXBC_CACHE=<dir> keeps the bytecode on disk between runs. */
static ID3DBlob* eng_shader(const char* hlsl, int stage, const char* what)
{
    const u32 len = (u32)strlen(hlsl);
    const u64 hash = fnv1a64(hlsl, len, 1469598103934665603ull);
    for (u32 i = 0; i < s_blob_count; i++)
        if (s_blob[i].hash == hash) return s_blob[i].blob;
    if (s_blob_count >= ENG_MAX_BLOBS) return NULL;

    char name[96];
    snprintf(name, sizeof name, "%s_%016llx.hlsl", what, (unsigned long long)hash);
    dump_shader(name, hlsl, len);

    static const char* cache_dir = (const char*)1;
    if (cache_dir == (const char*)1) {
        cache_dir = getenv("PS3RECOMP_DXBC_CACHE");
        if (cache_dir && *cache_dir) _mkdir(cache_dir); else cache_dir = NULL;
    }
    char cpath[1024] = "";
    ID3DBlob* blob = NULL;
    if (cache_dir) {
        const u64 key = fnv1a64("dxbc-v1", 7, hash ^ (u64)(stage + 1) * 0x9E3779B97F4A7C15ull);
        snprintf(cpath, sizeof cpath, "%s/%s_%016llx.dxbc", cache_dir, what, (unsigned long long)key);
        FILE* cf = fopen(cpath, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END); long n = ftell(cf); fseek(cf, 0, SEEK_SET);
            if (n > 0 && SUCCEEDED(D3DCreateBlob((SIZE_T)n, &blob))) {
                if (fread(CALL0(blob, GetBufferPointer), 1, (size_t)n, cf) != (size_t)n) RELEASE(blob);
            }
            fclose(cf);
        }
    }
    if (!blob) {
        snprintf(name, sizeof name, "%s %016llx", what, (unsigned long long)hash);
        blob = eng_compile(hlsl, len, "main", stage == 0 ? "vs_5_0" : "ps_5_0", name);
        if (blob && cpath[0]) {
            char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", cpath);
            FILE* cf = fopen(tmp, "wb");
            if (cf) {
                fwrite(CALL0(blob, GetBufferPointer), 1, CALL0(blob, GetBufferSize), cf);
                fclose(cf);
                remove(cpath); rename(tmp, cpath);
            }
        }
    }
    s_blob[s_blob_count].hash = hash;
    s_blob[s_blob_count].blob = blob;
    s_blob_count++;
    return blob;
}

/* A fragment program at the current internal resolution. */
static ID3DBlob* eng_fp_blob(const char* ps_hlsl)
{
    /* With the internal resolution raised, a fragment program's WPOS (the
     * decompiler's `input.position`) arrives in host pixels; the title
     * computes screen UVs and offsets from it in its own. Divide it back:
     * every use of the input is wrapped, and the text is what the shader
     * cache is keyed on, so the two builds never collide. */
    ID3DBlob* ps = NULL;
    const char* wp = (s_scale != 1.0f) ? strstr(ps_hlsl, "input.position") : NULL;
    if (wp) {
        const char* needle = "input.position";
        const char* repl = "(input.position*RSX_WPOS)";
        const size_t nl = strlen(needle), rl = strlen(repl), src_len = strlen(ps_hlsl);
        size_t n = 0;
        for (const char* q = wp; (q = strstr(q, needle)) != NULL; q += nl) n++;
        char head[96];
        snprintf(head, sizeof head, "static const float4 RSX_WPOS = float4(%.8f, %.8f, 1.0, 1.0);\n",
                 1.0 / (double)s_scale, 1.0 / (double)s_scale);
        char* text = (char*)malloc(src_len + n * (rl - nl) + strlen(head) + 1);
        if (text) {
            char* d = text + sprintf(text, "%s", head);
            for (const char* q = ps_hlsl;;) {
                const char* hit = strstr(q, needle);
                if (!hit) { strcpy(d, q); break; }
                memcpy(d, q, (size_t)(hit - q)); d += hit - q;
                memcpy(d, repl, rl); d += rl;
                q = hit + nl;
            }
            ps = eng_shader(text, 1, "fp");
            free(text);
        }
    }
    if (!ps) ps = eng_shader(ps_hlsl, 1, "fp");
    return ps;
}

static u32 eng_pipeline_create_locked(const char* vs_hlsl, const char* ps_hlsl,
                                      const rsx_be_render_state* rs,
                                      const rsx_vertex_layout_plan* layout,
                                      u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    if (!s_dev || !vertex_stride) return 0;
    if (!rt_count) rt_count = 1;
    if (rt_count > RSX_BE_MAX_COLOR_TARGETS) rt_count = RSX_BE_MAX_COLOR_TARGETS;
    if (s_pipe_count >= ENG_MAX_PIPES) return 0;
    ID3DBlob* vs = eng_shader(vs_hlsl, 0, "vp");
    if (!vs) return 0;
    ID3DBlob* ps = NULL;
    const int wp = strstr(ps_hlsl, "input.position") != NULL;
    ps = eng_fp_blob(ps_hlsl);
    if (!ps) return 0;

    EngPipeline* p = &s_pipe[s_pipe_count];
    memset(p, 0, sizeof *p);
    CALL0(vs, AddRef); CALL0(ps, AddRef);
    p->vs = vs; p->ps = ps;
    if (wp) p->ps_plain = _strdup(ps_hlsl);   /* at any scale: a rescale rebuilds from it */
    /* Input slot i carries attribute attrs[i] at i*16: the layout SLOT, as
     * rsx_metal_backend.m's eng_pipeline_create explains. */
    for (u32 slot = 0; slot < layout->count && slot < RSX_DSP_NUM_VERTEX_ATTR; slot++) {
        D3D12_INPUT_ELEMENT_DESC* e = &p->il[slot];
        e->SemanticName = "ATTR"; e->SemanticIndex = layout->attrs[slot];
        e->Format = DXGI_FORMAT_R32G32B32A32_FLOAT; e->InputSlot = 0;
        e->AlignedByteOffset = slot * 16u;
        e->InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC* pd = &p->desc;
    pd->pRootSignature = s_rootsig;
    pd->VS.pShaderBytecode = CALL0(vs, GetBufferPointer); pd->VS.BytecodeLength = CALL0(vs, GetBufferSize);
    pd->PS.pShaderBytecode = CALL0(ps, GetBufferPointer); pd->PS.BytecodeLength = CALL0(ps, GetBufferSize);
    pd->InputLayout.pInputElementDescs = layout->count ? p->il : NULL;
    pd->InputLayout.NumElements = layout->count;
    pd->SampleMask = UINT_MAX;
    pd->SampleDesc.Count = 1;
    pd->RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd->RasterizerState.DepthClipEnable = TRUE;
    /* CULL_FACE FRONT=0x0404 BACK=0x0405 FRONT_AND_BACK=0x0408 (front here,
     * as in the Metal engine); FRONT_FACE CW=0x0900 CCW=0x0901. */
    pd->RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    if (rs->cull_enable && rs->cull_face)
        pd->RasterizerState.CullMode = (rs->cull_face == 0x0404u || rs->cull_face == 0x0408u) ? D3D12_CULL_MODE_FRONT
                                     : (rs->cull_face == 0x0405u ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE);
    pd->RasterizerState.FrontCounterClockwise = (rs->front_face == 0x0901u) ? TRUE : FALSE;
    /* nv40 COLOR_MASK byte layout: B=[0:7] G=[8:15] R=[16:23] A=[24:31]. */
    UINT8 wm = 0;
    if ((rs->color_mask >>  0) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_BLUE;
    if ((rs->color_mask >>  8) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_GREEN;
    if ((rs->color_mask >> 16) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_RED;
    if ((rs->color_mask >> 24) & 0xFF) wm |= D3D12_COLOR_WRITE_ENABLE_ALPHA;
    D3D12_RENDER_TARGET_BLEND_DESC b = {0};
    b.RenderTargetWriteMask = wm;
    if (rs->blend_enable) {
        b.BlendEnable    = TRUE;
        b.SrcBlend       = gcm_blend_factor(rs->sf_rgb, 0);
        b.DestBlend      = gcm_blend_factor(rs->df_rgb, 0);
        b.BlendOp        = gcm_blend_op(rs->eq_rgb);
        b.SrcBlendAlpha  = gcm_blend_factor(rs->sf_a, 1);
        b.DestBlendAlpha = gcm_blend_factor(rs->df_a, 1);
        b.BlendOpAlpha   = gcm_blend_op(rs->eq_a);
    } else {
        b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_ZERO; b.BlendOp = D3D12_BLEND_OP_ADD;
        b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_ZERO; b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    }
    /* Every attachment of an MRT set takes target A's blend and colour mask. */
    pd->NumRenderTargets = rt_count;
    for (u32 r = 0; r < rt_count; r++) {
        pd->RTVFormats[r] = eng_dxgi(rt_fmt);
        pd->BlendState.RenderTarget[r] = b;
    }
    pd->DSVFormat = ENG_DEPTH_FMT;
    pd->DepthStencilState.DepthEnable = rs->depth_test ? TRUE : FALSE;
    pd->DepthStencilState.DepthWriteMask = (rs->depth_test && rs->depth_write) ? D3D12_DEPTH_WRITE_MASK_ALL
                                                                               : D3D12_DEPTH_WRITE_MASK_ZERO;
    pd->DepthStencilState.DepthFunc = rs->depth_test ? gcm_cmp(rs->depth_func) : D3D12_COMPARISON_FUNC_ALWAYS;
    if (rs->stencil_enable) {
        pd->DepthStencilState.StencilEnable    = TRUE;
        pd->DepthStencilState.StencilReadMask  = (UINT8)(rs->s_func_mask & 0xFFu);
        pd->DepthStencilState.StencilWriteMask = (UINT8)(rs->s_write_mask & 0xFFu);
        pd->DepthStencilState.FrontFace.StencilFunc        = gcm_cmp(rs->s_func);
        pd->DepthStencilState.FrontFace.StencilFailOp      = gcm_stencil_op(rs->s_fail);
        pd->DepthStencilState.FrontFace.StencilDepthFailOp = gcm_stencil_op(rs->s_zfail);
        pd->DepthStencilState.FrontFace.StencilPassOp      = gcm_stencil_op(rs->s_zpass);
        if (rs->stencil_two_sided) {
            pd->DepthStencilState.BackFace.StencilFunc        = gcm_cmp(rs->bs_func);
            pd->DepthStencilState.BackFace.StencilFailOp      = gcm_stencil_op(rs->bs_fail);
            pd->DepthStencilState.BackFace.StencilDepthFailOp = gcm_stencil_op(rs->bs_zfail);
            pd->DepthStencilState.BackFace.StencilPassOp      = gcm_stencil_op(rs->bs_zpass);
        } else {
            /* Two-sided off: nv40 applies the front state to both faces. */
            pd->DepthStencilState.BackFace = pd->DepthStencilState.FrontFace;
        }
    }
    p->live = 1;
    return ++s_pipe_count;
}

/* The engine may build pipelines on a worker thread as well as the walker;
 * the blob cache and the table are shared, so one at a time. */
static ID3D12PipelineState* eng_pso(u32 pipeline, int cls);   /* below */
static u32 eng_pipeline_create(void* user, const char* vs_hlsl, const char* ps_hlsl,
                               const rsx_be_render_state* rs,
                               const rsx_vertex_layout_plan* layout,
                               u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    (void)user;
    AcquireSRWLockExclusive(&s_pipe_lock);
    const u32 h = eng_pipeline_create_locked(vs_hlsl, ps_hlsl, rs, layout, vertex_stride, rt_fmt, rt_count);
    ReleaseSRWLockExclusive(&s_pipe_lock);
    /* The pipeline state for triangles too, here: this runs on the draw
     * engine's build thread (RSX_ASYNC_SHADERS), and a state object is the
     * driver's compile -- milliseconds, tens of them for a shader it has not
     * cached -- which on the walker, at the pipeline's first draw, was a
     * hitch the first time anything new came on screen. Points and lines,
     * rare, are still built on first use. RSX_PSO_PREWARM=0: all on first use. */
    { static int warm = -1;
      if (warm < 0) { const char* e = getenv("RSX_PSO_PREWARM"); warm = !(e && e[0] == '0'); }
      if (h && warm) eng_pso(h, 2); }
    return h;
}

static void eng_pipeline_release(void* user, u32 pipeline)
{
    (void)user;
    AcquireSRWLockExclusive(&s_pipe_lock);
    if (pipeline && pipeline <= s_pipe_count) {
        /* The PSOs stay until shutdown: a submitted list may still name them,
         * and a released pipeline is one the engine evicted, which is rare. */
        s_pipe[pipeline - 1].live = 0;
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
}

/* The PSO for a pipeline and a topology class, built on first use: D3D12
 * bakes the class into the state object where Metal does not. */
extern double g_rsx_frame_pso_ms;   /* rsx_draw_engine.c: pipeline states built on the walker this frame */
static DWORD s_walker_tid;
static ID3D12PipelineState* eng_pso(u32 pipeline, int cls)
{
    if (!pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->vs || p->failed[cls]) return NULL;
    if (p->pso[cls]) return p->pso[cls];
    LARGE_INTEGER pq0, pq1, pqf; QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0);
    AcquireSRWLockExclusive(&s_pipe_lock);
    if (!p->pso[cls] && !p->failed[cls]) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
        pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
        pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                 : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                            : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        HRESULT hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&p->pso[cls]);
        if (FAILED(hr)) {
            static int n = 0;
            if (n++ < 16) fprintf(stderr, "[rsx engine/d3d12] pipeline state failed: 0x%08lX (removed 0x%08lX)\n",
                                  (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason));
            p->failed[cls] = 1; p->pso[cls] = NULL;
        }
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    QueryPerformanceCounter(&pq1);
    const double pso_ms = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
    const int on_walker = GetCurrentThreadId() == s_walker_tid;
    if (on_walker) g_rsx_frame_pso_ms += pso_ms;
    /* RSX_PSO_LOG=1: every pipeline state built, how long, and where. */
    { static int lg = -1; if (lg < 0) lg = getenv("RSX_PSO_LOG") ? 1 : 0;
      if (lg) fprintf(stderr, "[pso] pipeline %u class %d: %.2f ms on the %s\n", pipeline, cls, pso_ms,
                      on_walker ? "walker" : "build thread"); }
    return p->pso[cls];
}

/* eng_pso, or for a pass into a target kept at the guest size the build whose
 * fragment program leaves WPOS undivided (a pipeline that does not read WPOS
 * has one build for both). */
static ID3D12PipelineState* eng_pso_for(u32 pipeline, int cls, int unscaled)
{
    if (!unscaled || !pipeline || pipeline > s_pipe_count) return eng_pso(pipeline, cls);
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->ps_plain || s_scale == 1.0f) return eng_pso(pipeline, cls);
    if (p->pso1[cls]) return p->pso1[cls];
    if (!p->vs || p->failed1[cls]) return NULL;
    LARGE_INTEGER pq0, pq1, pqf; QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0);
    AcquireSRWLockExclusive(&s_pipe_lock);
    if (!p->pso1[cls] && !p->failed1[cls]) {
        if (!p->ps1) { p->ps1 = eng_shader(p->ps_plain, 1, "fp"); if (p->ps1) CALL0(p->ps1, AddRef); }
        HRESULT hr = E_FAIL;
        if (p->ps1) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
            pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
            pd.PS.pShaderBytecode = CALL0(p->ps1, GetBufferPointer); pd.PS.BytecodeLength = CALL0(p->ps1, GetBufferSize);
            pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                     : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                                : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&p->pso1[cls]);
        }
        if (FAILED(hr)) { p->failed1[cls] = 1; p->pso1[cls] = NULL; }
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    QueryPerformanceCounter(&pq1);
    const double pso_ms = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
    if (GetCurrentThreadId() == s_walker_tid) g_rsx_frame_pso_ms += pso_ms;
    { static int lg = -1; if (lg < 0) lg = getenv("RSX_PSO_LOG") ? 1 : 0;
      if (lg) fprintf(stderr, "[pso] pipeline %u class %d, WPOS undivided: %.2f ms\n", pipeline, cls, pso_ms); }
    return p->pso1[cls];
}

/* ---- per-draw binding ------------------------------------------------------ */

/* RSX_ANISO=<1..16> (default 16): anisotropic filtering for the textures the
 * title filters linearly and mipmaps -- the world's ground and walls, which
 * the RSX's own trilinear blurs at a glancing angle. Only for a guest
 * texture with a real mip chain bound to a linear, mipmapped sampler: render
 * targets, their views and snapshots (shadow maps, packed depth, post-process
 * inputs) carry data that spreading samples corrupts -- 16x over them put red
 * and cyan speckle on the ground and the cloth. 1 is off. */
int g_rsx_aniso;   /* 0 until read from RSX_ANISO; the Graphics Settings page sets it live */
static int eng_aniso(void)
{
    int a = g_rsx_aniso;
    if (a <= 0) {
        const char* e = getenv("RSX_ANISO");
        a = (e && *e) ? atoi(e) : 16;
        g_rsx_aniso = a;
    }
    return a < 1 ? 1 : a > 16 ? 16 : a;
}
int g_eng_aniso_on = 1;   /* the host A/B (DOD3_AB=aniso) flips it */

/* A guest texture with a mip chain (not a target, view or snapshot). */
static int eng_aniso_tex(u32 handle)
{
    const EngObj* o = eng_obj(handle);
    return o && o->kind == OBJ_TEXTURE && o->mips > 1;
}
static int eng_sampler_slot(const rsx_be_sampler_desc* d, int aniso_ok)
{
    aniso_ok = aniso_ok && g_eng_aniso_on && eng_aniso() > 1 && d->min_linear && d->mag_linear && d->mip_present;
    const u64 key = ((u64)(aniso_ok ? eng_aniso() : 0) << 56) | (u64)d->min_linear | ((u64)d->mag_linear << 1)
                  | ((u64)d->mip_linear << 2) | ((u64)d->mip_present << 3)
                  | ((u64)d->wrap_s << 4) | ((u64)d->wrap_t << 8) | ((u64)d->wrap_r << 12)
                  | ((u64)(u32)(d->min_lod * 256.0f) << 16)
                  | ((u64)(u32)(d->max_lod * 256.0f) << 32);
    for (u32 i = 0; i < s_samp_count; i++)
        if (s_samp[i].key == key) return (int)i;
    if (s_samp_count >= ENG_MAX_SAMPLERS) return -1;
    D3D12_SAMPLER_DESC sd = {0};
    const D3D12_FILTER_TYPE mnf = d->min_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mgf = d->mag_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mpf = d->mip_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    sd.Filter = D3D12_ENCODE_BASIC_FILTER(mnf, mgf, mpf, D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    sd.AddressU = gcm_wrap(d->wrap_s); sd.AddressV = gcm_wrap(d->wrap_t); sd.AddressW = gcm_wrap(d->wrap_r);
    sd.MinLOD = d->min_lod;
    sd.MaxLOD = d->mip_present ? d->max_lod : 0.0f;
    if (sd.MaxLOD < sd.MinLOD) sd.MaxLOD = sd.MinLOD;
    sd.MaxAnisotropy = 1;
    if (aniso_ok) {
        sd.Filter = D3D12_FILTER_ANISOTROPIC;
        sd.MaxAnisotropy = (UINT)eng_aniso();
    }
    sd.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    /* Border: transparent black, as the Metal engine's. */
    const int slot = (int)s_samp_count++;
    CALL(s_dev, CreateSampler, &sd, cpu_handle(s_smp_cpu, s_smp_step, (u32)slot + 1));
    s_samp[slot].key = key;
    return slot;
}

static void eng_bind_targets(void* user, const u32* surfaces, u32 count, u32 depth)
{
    (void)user;
    if (count > RSX_BE_MAX_COLOR_TARGETS) count = RSX_BE_MAX_COLOR_TARGETS;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++)
        s_pending.rt[i] = (i < count) ? surfaces[i] : 0;
    s_pending.nrt = count;
    s_pending.depth = depth;
}
static void eng_bind_pipeline(void* user, u32 pipeline) { (void)user; s_pending.pipeline = pipeline; }

static void eng_bind_vs_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!eng_stage_copy(data, bytes, &s_pending.vs_cb_off)) bytes = 0;
    s_pending.vs_cb_bytes = bytes;
    s_vs_cb_seq = s_submit_seq;        /* read after the copy: it may have submitted */
    s_vs_cb_off = s_pending.vs_cb_off;
    s_vs_cb_bytes = bytes;
}
static int eng_reuse_vs_constants(void* user)
{
    (void)user;
    if (s_vs_cb_seq != s_submit_seq || !s_vs_cb_bytes) return 0;
    s_pending.vs_cb_off = s_vs_cb_off;
    s_pending.vs_cb_bytes = s_vs_cb_bytes;
    return 1;
}
static void eng_bind_ps_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!eng_stage_copy(data, bytes, &s_pending.ps_cb_off)) bytes = 0;
    s_pending.ps_cb_bytes = bytes;
}
static void eng_bind_textures(void* user, const u32* textures, const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
        s_pending.tex[u]  = ((mask >> u) & 1u) ? textures[u] : 0;
        s_pending.samp[u] = ((mask >> u) & 1u) ? eng_sampler_slot(&samplers[u], eng_aniso_tex(textures[u])) : -1;
    }
}
static void eng_bind_vertex_textures(void* user, const u32* textures, const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        s_pending.vtex[u]  = ((mask >> u) & 1u) ? textures[u] : 0;
        s_pending.vsamp[u] = ((mask >> u) & 1u) ? eng_sampler_slot(&samplers[u], 0) : -1;
    }
}
/* Viewport and scissor are kept in guest pixels; eng_vp_sc scales them for
 * the target the draw lands in. */
static void eng_set_viewport(void* user, float x, float y, float w, float h)
{ (void)user; s_pending.vp[0] = x; s_pending.vp[1] = y; s_pending.vp[2] = w; s_pending.vp[3] = h; }
static void eng_set_scissor(void* user, u32 x, u32 y, u32 w, u32 h)
{ (void)user; s_pending.sc[0] = x; s_pending.sc[1] = y; s_pending.sc[2] = w; s_pending.sc[3] = h; }
static void eng_set_stencil_ref(void* user, u32 ref) { (void)user; s_pending.stencil_ref = ref; }

static void eng_draw(void* user, rsx_topology topology, const void* vertices, u32 vertex_count,
                     u32 stride, const u32* indices, u32 index_count)
{
    (void)user;
    if (!vertex_count || !stride) return;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    u32 vb_off = 0, ib_off = 0;
    if (!eng_stage_copy(vertices, vertex_count * stride, &vb_off)) return;
    if (indices && index_count && !eng_stage_copy(indices, index_count * (u32)sizeof(u32), &ib_off)) return;
    EngRecord* r = &s_rec[s_rec_count++];
    *r = s_pending;
    r->kind = ENG_REC_DRAW; r->topology = topology; r->vbuf = 0;
    r->vb_off = vb_off; r->stride = stride; r->vertex_count = vertex_count;
    r->ib_off = ib_off; r->index_count = indices ? index_count : 0;
    r->vis = s_vis_cur;
}
static void eng_draw_buffer(void* user, rsx_topology topology, u32 buffer, u32 vb_off,
                            u32 vertex_count, u32 stride, u32 ib_off, u32 index_count)
{
    (void)user;
    if (!vertex_count || !stride || !buffer || buffer > s_buf_count || !s_buf[buffer - 1]) return;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    EngRecord* r = &s_rec[s_rec_count++];
    *r = s_pending;
    r->kind = ENG_REC_DRAW; r->topology = topology; r->vbuf = buffer;
    r->vb_off = vb_off; r->stride = stride; r->vertex_count = vertex_count;
    r->ib_off = ib_off; r->index_count = index_count;
    r->vis = s_vis_cur;
}

static void eng_clear_color(void* user, u32 surface, const float rgba[4])
{
    (void)user;
    s_clear_argb = ((u32)(rgba[3] * 255.0f + 0.5f) << 24) | ((u32)(rgba[0] * 255.0f + 0.5f) << 16) |
                   ((u32)(rgba[1] * 255.0f + 0.5f) <<  8) |  (u32)(rgba[2] * 255.0f + 0.5f);
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_COLOR; r->rt[0] = surface; r->nrt = 1;
    memcpy(r->clear_rgba, rgba, sizeof r->clear_rgba);
}
static void eng_clear_depth_stencil(void* user, u32 depth, u32 flags, float depth_value, u8 stencil)
{
    (void)user;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_DS; r->depth = depth; r->clear_flags = flags;
    r->clear_depth = depth_value; r->clear_stencil = stencil;
}
static void eng_clear_depth_stencil_rect(void* user, u32 depth, u32 flags, float depth_value, u8 stencil,
                                         u32 x, u32 y, u32 w, u32 h)
{
    (void)user;
    if (!w || !h) return;
    if (s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    if (eng_obj_scaled(depth)) sc_rect(&x, &y, &w, &h);
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_CLEAR_DS_RECT; r->depth = depth; r->clear_flags = flags;
    r->clear_depth = depth_value; r->clear_stencil = stencil;
    r->clear_rect[0] = x; r->clear_rect[1] = y; r->clear_rect[2] = w; r->clear_rect[3] = h;
}

/* ---- encoding -------------------------------------------------------------- */

static D3D12_RESOURCE_BARRIER s_bar[32]; static u32 s_nbar;
static void bar_flush(void)
{
    if (s_nbar) { CALL(s_list, ResourceBarrier, s_nbar, s_bar); s_gt_bar_calls++; s_gt_bars += s_nbar; s_nbar = 0; }
}
static void res_transition(ID3D12Resource* res, D3D12_RESOURCE_STATES* state, D3D12_RESOURCE_STATES to)
{
    if (*state == to) return;
    if (s_nbar == 32) bar_flush();
    D3D12_RESOURCE_BARRIER* b = &s_bar[s_nbar++];
    memset(b, 0, sizeof *b);
    b->Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b->Transition.pResource = res;
    b->Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b->Transition.StateBefore = *state;
    b->Transition.StateAfter = to;
    *state = to;
}
static void obj_transition(u32 handle, D3D12_RESOURCE_STATES to)
{
    EngObj* o = eng_owner(handle);
    if (o) res_transition(o->res, &o->state, to);
}

/* What the open list has bound, so a run of similar draws rebinds little. */
static ID3D12PipelineState* s_cur_pso; static int s_cur_helper_rs;
static u32 s_cur_rt[RSX_BE_MAX_COLOR_TARGETS], s_cur_nrt, s_cur_depth; static int s_cur_targets_valid;
static D3D_PRIMITIVE_TOPOLOGY s_cur_topo;
static D3D12_CPU_DESCRIPTOR_HANDLE s_srv_last_src[ENG_SRV_TABLE]; static D3D12_GPU_DESCRIPTOR_HANDLE s_srv_last_gpu; static int s_srv_last_valid;
typedef struct { int slots[ENG_SRV_TABLE]; D3D12_GPU_DESCRIPTOR_HANDLE gpu; } EngSmpTable;
static EngSmpTable s_smp_tab[ENG_SMP_PER_SLOT / ENG_SRV_TABLE]; static u32 s_smp_ntab;
static ID3D12Resource* s_zero_cb;            /* bound where a draw stages no constants */
static D3D12_RESOURCE_STATES s_offscreen_state = D3D12_RESOURCE_STATE_RENDER_TARGET;

static void eng_reset_bind_state(void)
{
    s_cur_pso = NULL; s_cur_helper_rs = 0; s_cur_targets_valid = 0;
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    s_srv_last_valid = 0; s_smp_ntab = 0;
    s_srv_used = s_smp_used = 0; s_q_used = 0;
}

static const char* pass_kind_name(int k)
{
    static const char* const n[] = { "draws", "clear colour", "clear depth", "copy", "depth resolve", "uploads" };
    return k >= 0 && k < 6 ? n[k] : "?";
}
static void eng_pass_collect(u32 slot, int report)
{
    enum { N = 256 };
    static struct { u8 kind, nrt; u16 w, h; u32 fmt; double ms; u64 n, draws; } st[N];
    static u32 nst;
    if (s_pheap && s_pass_n[slot]) {
        const u32 base = slot * (ENG_PASS_MAX + 1u), np = s_pass_n[slot];
        u64* p = NULL; D3D12_RANGE rg = { base * 8u, (base + np + 1u) * 8u };
        if (SUCCEEDED(CALL(s_pread, Map, 0, &rg, (void**)&p))) {
            for (u32 i = 0; i < np; i++) {
                const u64 a = p[base + i], b = p[base + i + 1];
                const double ms = b > a ? (double)(b - a) * 1000.0 / (double)s_ts_freq : 0.0;
                const EngPassDesc* d = &s_pass[slot][i];
                u32 k;
                for (k = 0; k < nst; k++)
                    if (st[k].kind == d->kind && st[k].w == d->w && st[k].h == d->h && st[k].fmt == d->fmt && st[k].nrt == d->nrt) break;
                if (k == nst) { if (nst == N) continue; nst++; st[k].kind = d->kind; st[k].w = d->w; st[k].h = d->h; st[k].fmt = d->fmt; st[k].nrt = d->nrt; st[k].ms = 0; st[k].n = st[k].draws = 0; }
                st[k].ms += ms; st[k].n++; st[k].draws += d->draws;
            }
            D3D12_RANGE wr = {0, 0}; CALL(s_pread, Unmap, 0, &wr);
        }
        s_pass_n[slot] = 0;
    }
    if (report && nst) {
        double tot = 0; for (u32 k = 0; k < nst; k++) tot += st[k].ms;
        fprintf(stderr, "[gpu-pass] %.1f ms of passes in 5 s; the largest:\n", tot);
        for (int shown = 0; shown < 12; shown++) {
            int best = -1;
            for (u32 k = 0; k < nst; k++) if (st[k].n && (best < 0 || st[k].ms > st[best].ms)) best = (int)k;
            if (best < 0 || st[best].ms <= 0) break;
            fprintf(stderr, "[gpu-pass]   %5.1f%% %8.1f ms  %-13s %4ux%-4u fmt %3u x%u  %6llu passes, %7llu draws\n",
                    tot > 0 ? 100.0 * st[best].ms / tot : 0.0, st[best].ms, pass_kind_name(st[best].kind),
                    st[best].w, st[best].h, st[best].fmt, st[best].nrt,
                    (unsigned long long)st[best].n, (unsigned long long)st[best].draws);
            st[best].n = 0;   /* shown */
        }
        nst = 0;
    }
}

static void eng_gpu_time_collect(u32 slot)
{
    static double busy_ms, wait_ms, other_ms, queue_ms; static unsigned n, n_wait, n_other;
    static LARGE_INTEGER last; LARGE_INTEGER qf, now;
    if (!s_tsread || !s_ts_slot[slot].fence) return;
    QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&now);
    u64* p = NULL; D3D12_RANGE rg = { slot * 16u, slot * 16u + 16u };
    if (SUCCEEDED(CALL(s_tsread, Map, 0, &rg, (void**)&p))) {
        const u64 t0 = p[slot * 2], t1 = p[slot * 2 + 1];
        D3D12_RANGE wr = {0, 0}; CALL(s_tsread, Unmap, 0, &wr);
        const double gpu_ms = t1 > t0 ? (double)(t1 - t0) * 1000.0 / (double)s_ts_freq : 0.0;
        /* Where the GPU clock was when the CPU executed the list: one
         * calibration a submit is cheap next to the submit itself. */
        u64 g_now = 0, c_now = 0;
        if (SUCCEEDED(CALL(s_queue, GetClockCalibration, &g_now, &c_now))) {
            const double cpu_ago_ms = (double)((long long)c_now - s_ts_slot[slot].cpu_exec.QuadPart) * 1000.0 / (double)qf.QuadPart;
            const double gpu_ago_ms = (double)((long long)g_now - (long long)t0) * 1000.0 / (double)s_ts_freq;
            const double q = cpu_ago_ms - gpu_ago_ms;   /* exec -> GPU start */
            if (q > 0 && q < 1000) queue_ms += q;
        }
        busy_ms += gpu_ms; n++;
        eng_pass_collect(slot, 0);
        if (s_ts_slot[slot].kind) { wait_ms += gpu_ms; n_wait++; } else { other_ms += gpu_ms; n_other++; }
    }
    s_ts_slot[slot].fence = 0;
    if (!last.QuadPart) last = now;
    const double secs = (double)(now.QuadPart - last.QuadPart) / (double)qf.QuadPart;
    if (secs >= 5.0) {
        fprintf(stderr, "[gpu-time] %.0f submits/s, GPU busy %.1f ms/s; waited-on submits %.0f/s, %.2f ms GPU each; others %.0f/s, %.2f ms each; exec->GPU start %.2f ms mean\n",
                n / secs, busy_ms / secs, n_wait / secs, n_wait ? wait_ms / n_wait : 0.0,
                n_other / secs, n_other ? other_ms / n_other : 0.0, n ? queue_ms / n : 0.0);
        fprintf(stderr, "[gpu-time] per second: %.0f draws, %.0f PSO changes, %.0f ResourceBarrier calls (%.0f transitions), %.0f occlusion queries, %.0f texture uploads\n",
                s_gt_draws / secs, s_gt_pso / secs, s_gt_bar_calls / secs, s_gt_bars / secs, s_gt_queries / secs, s_gt_copies / secs);
        fprintf(stderr, "[gpu-time] own-target snapshots: %.0f/s brought up to date by %.0f copy draws/s, %.0f/s copied whole\n",
                s_snap_incr_n / secs, s_snap_copy_draws / secs, s_snap_full_n / secs);
        s_snap_incr_n = s_snap_full_n = s_snap_copy_draws = 0;
        s_gt_draws = s_gt_pso = s_gt_bar_calls = s_gt_bars = s_gt_queries = s_gt_copies = 0;
        eng_pass_collect(slot, 1);
        busy_ms = wait_ms = other_ms = queue_ms = 0; n = n_wait = n_other = 0; last = now;
    }
}

static void eng_list_begin(void)
{
    eng_gpu_time_collect(s_slot);
    CALL0(s_alloc[s_slot], Reset);
    CALL(s_list, Reset, s_alloc[s_slot], NULL);
    s_list_open = 1;
    if (s_tsheap) CALL(s_list, EndQuery, s_tsheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * 2u);
    ID3D12DescriptorHeap* heaps[2] = { s_srv_gpu, s_smp_gpu[s_slot] };
    CALL(s_list, SetDescriptorHeaps, 2, heaps);
    CALL(s_list, SetGraphicsRootSignature, s_rootsig);
    eng_reset_bind_state();
}

static void eng_use_main_rootsig(void)
{
    if (!s_cur_helper_rs) return;
    CALL(s_list, SetGraphicsRootSignature, s_rootsig);
    s_cur_helper_rs = 0; s_cur_pso = NULL; s_cur_targets_valid = 0;
}
static void eng_use_helper_rootsig(void)
{
    if (s_cur_helper_rs) return;
    CALL(s_list, SetGraphicsRootSignature, s_helper_rootsig);
    s_cur_helper_rs = 1; s_cur_pso = NULL; s_cur_targets_valid = 0;
}

/* The list's descriptor rings ran dry mid-frame: submit what is encoded,
 * wait, and continue on the same slot with fresh rings. */
static void eng_submit_list(u64* out_fence);
static void eng_hard_flush(void)
{
    u64 f = 0;
    eng_submit_list(&f);
    fence_wait(f);
    /* The queries of the partial list, synchronously. */
    if (s_q_used && s_qread) {
        D3D12_RANGE rg = { (SIZE_T)s_slot * ENG_QUERIES_PER_SLOT * 8u, ((SIZE_T)s_slot * ENG_QUERIES_PER_SLOT + s_q_used) * 8u };
        u64* p = NULL;
        if (SUCCEEDED(CALL(s_qread, Map, 0, &rg, (void**)&p))) {
            for (u32 i = 0; i < s_q_used; i++) s_vis_count[s_q_vis[i] & ~ENG_Q_SCALED] += eng_q_count(s_q_vis[i], p[s_slot * ENG_QUERIES_PER_SLOT + i]);
            D3D12_RANGE wr = {0, 0}; CALL(s_qread, Unmap, 0, &wr);
        }
    }
    static unsigned n = 0;
    if (n++ < 8) fprintf(stderr, "[rsx engine/d3d12] descriptor ring full: mid-frame flush (#%u)\n", n);
    eng_list_begin();
}

static D3D12_GPU_DESCRIPTOR_HANDLE eng_srv_table(const D3D12_CPU_DESCRIPTOR_HANDLE* src, u32 n)
{
    if (s_srv_used + n > ENG_SRV_PER_SLOT) { eng_hard_flush(); }
    const u32 base = s_slot * ENG_SRV_PER_SLOT + s_srv_used;
    s_srv_used += n;
    D3D12_CPU_DESCRIPTOR_HANDLE dst = cpu_handle(s_srv_gpu, s_srv_step, base);
    UINT dst_size = n;
    CALL(s_dev, CopyDescriptors, 1, &dst, &dst_size, n, src, NULL, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return gpu_handle(s_srv_gpu, s_srv_step, base);
}

static D3D12_GPU_DESCRIPTOR_HANDLE eng_smp_table(const int* slots)
{
    for (u32 i = 0; i < s_smp_ntab; i++)
        if (memcmp(s_smp_tab[i].slots, slots, sizeof(int) * ENG_SRV_TABLE) == 0) return s_smp_tab[i].gpu;
    if (s_smp_used + ENG_SRV_TABLE > ENG_SMP_PER_SLOT) { eng_hard_flush(); }
    const u32 base = s_smp_used;
    s_smp_used += ENG_SRV_TABLE;
    D3D12_CPU_DESCRIPTOR_HANDLE src[ENG_SRV_TABLE];
    for (u32 i = 0; i < ENG_SRV_TABLE; i++) src[i] = cpu_handle(s_smp_cpu, s_smp_step, slots[i] >= 0 ? (u32)slots[i] + 1 : 0);
    D3D12_CPU_DESCRIPTOR_HANDLE dst = cpu_handle(s_smp_gpu[s_slot], s_smp_step, base);
    UINT dst_size = ENG_SRV_TABLE;
    CALL(s_dev, CopyDescriptors, 1, &dst, &dst_size, ENG_SRV_TABLE, src, NULL, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    EngSmpTable* t = &s_smp_tab[s_smp_ntab++];
    memcpy(t->slots, slots, sizeof t->slots);
    t->gpu = gpu_handle(s_smp_gpu[s_slot], s_smp_step, base);
    return t->gpu;
}

/* A helper pass over a whole colour target: dst bound, src sampled at t0. */
static void eng_fullscreen_pass(ID3D12PipelineState* pso, D3D12_CPU_DESCRIPTOR_HANDLE rtv, u32 w, u32 h,
                                D3D12_CPU_DESCRIPTOR_HANDLE src_srv)
{
    eng_use_helper_rootsig();
    CALL(s_list, OMSetRenderTargets, 1, &rtv, FALSE, NULL);
    D3D12_VIEWPORT vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    D3D12_RECT sc = { 0, 0, (LONG)w, (LONG)h };
    CALL(s_list, RSSetViewports, 1, &vp);
    CALL(s_list, RSSetScissorRects, 1, &sc);
    CALL(s_list, SetPipelineState, pso);
    CALL(s_list, IASetPrimitiveTopology, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    CALL(s_list, SetGraphicsRootDescriptorTable, 0, eng_srv_table(&src_srv, 1));
    CALL(s_list, DrawInstanced, 3, 1, 0, 0);
    s_cur_pso = NULL; s_cur_targets_valid = 0;
}

/* A record's viewport and scissor (guest pixels) for a target of tw x th
 * host pixels, scaled if the target is. Empty ones cover the target. */
static void eng_vp_sc(const EngRecord* r, int scaled, u32 tw, u32 th, D3D12_VIEWPORT* vp, D3D12_RECT* sc)
{
    const float f = scaled ? s_scale : 1.0f;
    vp->TopLeftX = r->vp[0] * f; vp->TopLeftY = r->vp[1] * f;
    vp->Width = r->vp[2] * f;    vp->Height = r->vp[3] * f;
    vp->MinDepth = 0.0f; vp->MaxDepth = 1.0f;
    if (vp->Width <= 0.0f || vp->Height <= 0.0f) { vp->TopLeftX = vp->TopLeftY = 0.0f; vp->Width = (float)tw; vp->Height = (float)th; }
    u32 x = r->sc[0], y = r->sc[1], w = r->sc[2], h = r->sc[3];
    if (w && h && scaled) sc_rect(&x, &y, &w, &h);
    if (w && h) { sc->left = (LONG)x; sc->top = (LONG)y; sc->right = (LONG)(x + w); sc->bottom = (LONG)(y + h); }
    else { sc->left = 0; sc->top = 0; sc->right = (LONG)tw; sc->bottom = (LONG)th; }
}

static ID3D12PipelineState* eng_pso_for(u32 pipeline, int cls, int unscaled);   /* below */

static void eng_encode_upload(const EngRecord* r, ID3D12Resource* stage)
{
    EngObj* o = eng_owner(r->depth);
    if (!o || !stage) return;
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_COPY_DEST);
    bar_flush();
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = o->res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = r->up_sub;
    src.pResource = stage; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = r->up_off;
    src.PlacedFootprint.Footprint.Format = o->fmt;
    u32 w = r->up_w, h = r->up_h;
    if (o->fmt == DXGI_FORMAT_BC1_UNORM || o->fmt == DXGI_FORMAT_BC2_UNORM || o->fmt == DXGI_FORMAT_BC3_UNORM) {
        w = (w + 3u) & ~3u; h = (h + 3u) & ~3u;
    }
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = r->up_pitch;
    CALL(s_list, CopyTextureRegion, &dst, 0, 0, 0, &src, NULL);
    s_gt_copies++;
}

static void eng_encode_draw(const EngRecord* r, ID3D12Resource* stage, D3D12_GPU_VIRTUAL_ADDRESS stage_va)
{
    /* Attachments: target A's size rules, as the Metal engine. */
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[RSX_BE_MAX_COLOR_TARGETS];
    u32 nrt = 0, tw = 0, th = 0;
    for (u32 k = 0; k < r->nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) {
        EngObj* o = eng_owner(r->rt[k]);
        if (!o || !o->has_rtv) break;
        if (nrt && (o->w != tw || o->h != th)) break;
        if (!nrt) { tw = o->w; th = o->h; }
        rtv[nrt++] = obj_rtv(r->rt[k]);
    }
    if (nrt != r->nrt) return;   /* the pipeline declares r->nrt attachments */
    u32 depth = r->depth;
    EngObj* z = eng_owner(depth);
    if (z && !z->has_dsv) { z = NULL; depth = 0; }
    if (!nrt && !z) return;
    if (!z) { depth = eng_fallback_depth(tw, th); z = eng_owner(depth); if (!z) return; }
    if (!nrt) { tw = z->w; th = z->h; }
    /* The pass's scale is its first colour target's, else its depth's. */
    const int scaled = eng_obj_scaled(nrt ? r->rt[0] : depth);
    ID3D12PipelineState* pso = eng_pso_for(r->pipeline, topo_class(r->topology), !scaled);
    if (!pso) return;

    for (u32 k = 0; k < nrt; k++) obj_transition(r->rt[k], D3D12_RESOURCE_STATE_RENDER_TARGET);
    obj_transition(depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);

    /* Textures, with a unit that names one of this draw's own targets bound
     * to the null texture rather than left in a read/write hazard. */
    D3D12_CPU_DESCRIPTOR_HANDLE srv[ENG_SRV_TABLE];
    for (u32 u = 0; u < ENG_SRV_TABLE; u++) {
        const u32 h = u < RSX_BE_MAX_TEXTURES ? r->tex[u] : r->vtex[u - RSX_BE_MAX_TEXTURES];
        EngObj* o = eng_owner(h);
        int hazard = 0;
        if (o) {
            for (u32 k = 0; k < nrt; k++) if (eng_owner(r->rt[k]) == o) hazard = 1;
            if (z == o) hazard = 1;
        }
        if (!o || hazard) { srv[u] = obj_srv(0); continue; }
        obj_transition(h, ENG_SHADER_READ);
        srv[u] = obj_srv(h);
    }
    bar_flush();
    eng_use_main_rootsig();

    int same_targets = s_cur_targets_valid && s_cur_nrt == nrt && s_cur_depth == depth;
    for (u32 k = 0; same_targets && k < nrt; k++) same_targets = s_cur_rt[k] == r->rt[k];
    if (!same_targets) {
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = obj_dsv(depth);
        CALL(s_list, OMSetRenderTargets, nrt, nrt ? rtv : NULL, FALSE, &dsv);
        for (u32 k = 0; k < RSX_BE_MAX_COLOR_TARGETS; k++) s_cur_rt[k] = k < nrt ? r->rt[k] : 0;
        s_cur_nrt = nrt; s_cur_depth = depth; s_cur_targets_valid = 1;
    }
    if (pso != s_cur_pso) { CALL(s_list, SetPipelineState, pso); s_cur_pso = pso; s_gt_pso++; }
    const D3D_PRIMITIVE_TOPOLOGY topo = topo_d3d(r->topology);
    if (topo != s_cur_topo) { CALL(s_list, IASetPrimitiveTopology, topo); s_cur_topo = topo; }

    D3D12_VIEWPORT vp; D3D12_RECT sc;
    eng_vp_sc(r, scaled, tw, th, &vp, &sc);
    CALL(s_list, RSSetViewports, 1, &vp);
    CALL(s_list, RSSetScissorRects, 1, &sc);
    CALL(s_list, OMSetStencilRef, r->stencil_ref);

    ID3D12Resource* vres = r->vbuf ? s_buf[r->vbuf - 1] : stage;
    if (!vres) return;
    const D3D12_GPU_VIRTUAL_ADDRESS vva = r->vbuf ? CALL0(vres, GetGPUVirtualAddress) : stage_va;
    const u32 vbytes = r->vbuf ? s_buf_bytes[r->vbuf - 1] : s_stage_used;
    D3D12_VERTEX_BUFFER_VIEW vbv = { vva + r->vb_off, r->vb_off < vbytes ? vbytes - r->vb_off : 0, r->stride };
    if (vbv.SizeInBytes > (u64)r->vertex_count * r->stride && !r->index_count) vbv.SizeInBytes = r->vertex_count * r->stride;
    CALL(s_list, IASetVertexBuffers, 0, 1, &vbv);
    if (r->index_count) {
        D3D12_INDEX_BUFFER_VIEW ibv = { vva + r->ib_off, r->index_count * 4u, DXGI_FORMAT_R32_UINT };
        CALL(s_list, IASetIndexBuffer, &ibv);
    }
    const D3D12_GPU_VIRTUAL_ADDRESS zero_va = CALL0(s_zero_cb, GetGPUVirtualAddress);
    CALL(s_list, SetGraphicsRootConstantBufferView, 0, r->vs_cb_bytes ? stage_va + r->vs_cb_off : zero_va);
    CALL(s_list, SetGraphicsRootConstantBufferView, 4, r->ps_cb_bytes ? stage_va + r->ps_cb_off : zero_va);

    D3D12_GPU_DESCRIPTOR_HANDLE st;
    if (s_srv_last_valid && memcmp(s_srv_last_src, srv, sizeof srv) == 0) st = s_srv_last_gpu;
    else {
        st = eng_srv_table(srv, ENG_SRV_TABLE);
        memcpy(s_srv_last_src, srv, sizeof srv); s_srv_last_gpu = st; s_srv_last_valid = 1;
        /* A hard flush inside eng_srv_table rebound everything. */
        if (s_cur_pso != pso) {
            CALL(s_list, SetPipelineState, pso); s_cur_pso = pso;
            D3D12_CPU_DESCRIPTOR_HANDLE dsv = obj_dsv(depth);
            CALL(s_list, OMSetRenderTargets, nrt, nrt ? rtv : NULL, FALSE, &dsv);
            for (u32 k = 0; k < RSX_BE_MAX_COLOR_TARGETS; k++) s_cur_rt[k] = k < nrt ? r->rt[k] : 0;
            s_cur_nrt = nrt; s_cur_depth = depth; s_cur_targets_valid = 1;
            CALL(s_list, IASetPrimitiveTopology, topo); s_cur_topo = topo;
            CALL(s_list, RSSetViewports, 1, &vp); CALL(s_list, RSSetScissorRects, 1, &sc);
            CALL(s_list, OMSetStencilRef, r->stencil_ref);
            CALL(s_list, IASetVertexBuffers, 0, 1, &vbv);
            if (r->index_count) { D3D12_INDEX_BUFFER_VIEW ibv = { vva + r->ib_off, r->index_count * 4u, DXGI_FORMAT_R32_UINT }; CALL(s_list, IASetIndexBuffer, &ibv); }
            CALL(s_list, SetGraphicsRootConstantBufferView, 0, r->vs_cb_bytes ? stage_va + r->vs_cb_off : zero_va);
            CALL(s_list, SetGraphicsRootConstantBufferView, 4, r->ps_cb_bytes ? stage_va + r->ps_cb_off : zero_va);
        }
    }
    D3D12_GPU_DESCRIPTOR_HANDLE vt = st; vt.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_srv_step;
    CALL(s_list, SetGraphicsRootDescriptorTable, 1, st);
    CALL(s_list, SetGraphicsRootDescriptorTable, 3, vt);
    int smp[ENG_SRV_TABLE];
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) smp[u] = r->samp[u];
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) smp[RSX_BE_MAX_TEXTURES + u] = r->vsamp[u];
    D3D12_GPU_DESCRIPTOR_HANDLE sm = eng_smp_table(smp);
    D3D12_GPU_DESCRIPTOR_HANDLE vsm = sm; vsm.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_smp_step;
    CALL(s_list, SetGraphicsRootDescriptorTable, 2, sm);
    CALL(s_list, SetGraphicsRootDescriptorTable, 5, vsm);

    const int query = s_qheap && r->vis && s_q_used < ENG_QUERIES_PER_SLOT;
    const u32 qi = s_slot * ENG_QUERIES_PER_SLOT + s_q_used;
    if (query) { CALL(s_list, BeginQuery, s_qheap, D3D12_QUERY_TYPE_OCCLUSION, qi); s_gt_queries++; }
    s_gt_draws++;
    if (r->index_count) CALL(s_list, DrawIndexedInstanced, r->index_count, 1, 0, 0, 0);
    else                CALL(s_list, DrawInstanced, r->vertex_count, 1, 0, 0);
    if (query) { CALL(s_list, EndQuery, s_qheap, D3D12_QUERY_TYPE_OCCLUSION, qi); s_q_vis[s_q_used++] = (r->vis - 1) | (scaled ? ENG_Q_SCALED : 0u); }
}

/* ---- incremental own-target snapshots --------------------------------------
 *
 * A draw that samples the colour target it renders into samples a snapshot
 * of it (eng_color_snapshot), refreshed at that point in the stream by a full
 * copy -- ~0.1 ms of GPU for a 4K RGBA16F target, and Drakengard 3's heat-haze
 * particles ask for ~95 a frame. Between two refreshes the target only
 * changes where the draws into it in between wrote, and those are recorded:
 * draw each of them again, same vertex shader and geometry, viewport and
 * scissor, with a pixel shader that copies the target's pixel into the
 * snapshot, and the snapshot equals the target everywhere again -- exactly,
 * at the cost of the draws' own footprints. Culling and depth are off in the
 * copy, so it covers at least every pixel the draw could have written; more
 * is harmless, those pixels are copies of an unchanged target. Anything else
 * that writes the target (a clear, a copy, an upload, a resolve, a draw with
 * it as a second attachment), more than 16 draws, or a refresh with no
 * earlier one in the same submit takes the full copy. RSX_SNAP_INCR=0: full
 * copies always. */

static ID3D12PipelineState* eng_copy_pso_build(u32 pipeline, int cls, DXGI_FORMAT fmt)
{
    if (!s_snap_copy_ps || !pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->vs || p->copy_failed[cls]) return NULL;
    if (p->copy_pso[cls] && p->copy_fmt[cls] == fmt) return p->copy_pso[cls];
    ID3D12PipelineState* pso = NULL;
    AcquireSRWLockExclusive(&s_pipe_lock);
    if (p->copy_pso[cls] && p->copy_fmt[cls] == fmt) pso = p->copy_pso[cls];
    else if (!p->copy_failed[cls]) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = p->desc;
        pd.InputLayout.pInputElementDescs = pd.InputLayout.NumElements ? p->il : NULL;
        pd.PS.pShaderBytecode = CALL0(s_snap_copy_ps, GetBufferPointer);
        pd.PS.BytecodeLength = CALL0(s_snap_copy_ps, GetBufferSize);
        pd.NumRenderTargets = 1;
        for (u32 r = 0; r < 8; r++) pd.RTVFormats[r] = DXGI_FORMAT_UNKNOWN;
        pd.RTVFormats[0] = fmt;
        memset(&pd.BlendState, 0, sizeof pd.BlendState);
        D3D12_RENDER_TARGET_BLEND_DESC* b = &pd.BlendState.RenderTarget[0];
        b->SrcBlend = D3D12_BLEND_ONE; b->DestBlend = D3D12_BLEND_ZERO; b->BlendOp = D3D12_BLEND_OP_ADD;
        b->SrcBlendAlpha = D3D12_BLEND_ONE; b->DestBlendAlpha = D3D12_BLEND_ZERO; b->BlendOpAlpha = D3D12_BLEND_OP_ADD;
        b->LogicOp = D3D12_LOGIC_OP_NOOP;
        b->RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        memset(&pd.DepthStencilState, 0, sizeof pd.DepthStencilState);
        pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        const D3D12_DEPTH_STENCILOP_DESC keep = { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                                  D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS };
        pd.DepthStencilState.FrontFace = keep; pd.DepthStencilState.BackFace = keep;
        pd.DSVFormat = DXGI_FORMAT_UNKNOWN;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                 : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                            : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        HRESULT hr = CALL(s_dev, CreateGraphicsPipelineState, &pd, &IID_ID3D12PipelineState, (void**)&pso);
        if (FAILED(hr)) {
            static int n = 0;
            if (n++ < 8) fprintf(stderr, "[rsx engine/d3d12] snapshot-copy pipeline failed: 0x%08lX\n", (long)hr);
            p->copy_failed[cls] = 1; pso = NULL;
        } else {
            /* A format change leaves the old one to any list still naming it. */
            p->copy_pso[cls] = pso; p->copy_fmt[cls] = fmt;
        }
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    return pso;
}

/* Copy pipelines are built on a thread of their own: a pipeline state is
 * milliseconds of driver compile, and on the walker each first use was a
 * hitch. Until one is ready its refreshes take the full copy. With
 * RSX_ASYNC_SHADERS=0 (the replays) they are built in place. */
static SRWLOCK s_cp_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_cp_cv = CONDITION_VARIABLE_INIT;
static struct { u32 pipeline; int cls; DXGI_FORMAT fmt; } s_cp_req[256];
static u32 s_cp_head, s_cp_tail;      /* ring of requests */
static int s_cp_thread_started;
static DWORD WINAPI eng_copy_pso_worker(LPVOID arg)
{
    (void)arg;
    for (;;) {
        AcquireSRWLockExclusive(&s_cp_lock);
        while (s_cp_head == s_cp_tail) SleepConditionVariableSRW(&s_cp_cv, &s_cp_lock, INFINITE, 0);
        const u32 i = s_cp_head++ % 256u;
        const u32 pl = s_cp_req[i].pipeline; const int cls = s_cp_req[i].cls; const DXGI_FORMAT fmt = s_cp_req[i].fmt;
        ReleaseSRWLockExclusive(&s_cp_lock);
        eng_copy_pso_build(pl, cls, fmt);
    }
    return 0;
}
static ID3D12PipelineState* eng_copy_pso(u32 pipeline, int cls, DXGI_FORMAT fmt)
{
    if (!s_snap_copy_ps || !pipeline || pipeline > s_pipe_count) return NULL;
    EngPipeline* p = &s_pipe[pipeline - 1];
    if (!p->vs || p->copy_failed[cls]) return NULL;
    if (p->copy_pso[cls] && p->copy_fmt[cls] == fmt) return p->copy_pso[cls];
    static int async = -1;
    if (async < 0) { const char* e = getenv("RSX_ASYNC_SHADERS"); async = !(e && e[0] == '0'); }
    if (!async) return eng_copy_pso_build(pipeline, cls, fmt);
    /* Ask once per (pipeline, class, format): requests[] remembers. */
    static struct { u32 pipeline; int cls; DXGI_FORMAT fmt; } asked[1024];
    static u32 nasked;
    for (u32 i = 0; i < nasked; i++)
        if (asked[i].pipeline == pipeline && asked[i].cls == cls && asked[i].fmt == fmt) return NULL;
    if (nasked < 1024) { asked[nasked].pipeline = pipeline; asked[nasked].cls = cls; asked[nasked].fmt = fmt; nasked++; }
    AcquireSRWLockExclusive(&s_cp_lock);
    if (!s_cp_thread_started) {
        HANDLE th = CreateThread(NULL, 1u << 20, eng_copy_pso_worker, NULL, 0, NULL);
        if (th) { SetThreadDescription(th, L"rsx copy pipelines"); CloseHandle(th); s_cp_thread_started = 1; }
    }
    if (s_cp_thread_started && s_cp_tail - s_cp_head < 256u) {
        const u32 i = s_cp_tail++ % 256u;
        s_cp_req[i].pipeline = pipeline; s_cp_req[i].cls = cls; s_cp_req[i].fmt = fmt;
        WakeConditionVariable(&s_cp_cv);
    }
    ReleaseSRWLockExclusive(&s_cp_lock);
    return NULL;
}

/* Draw r's geometry into snapshot S with the copy shader reading target T. */
static int eng_encode_copy_draw(const EngRecord* r, u32 T, u32 S, ID3D12Resource* stage, D3D12_GPU_VIRTUAL_ADDRESS stage_va)
{
    EngObj* t = eng_owner(T); EngObj* sn = eng_owner(S);
    if (!t || !sn || !sn->has_rtv) return 0;
    ID3D12PipelineState* pso = eng_copy_pso(r->pipeline, topo_class(r->topology), sn->fmt);
    if (!pso) return 0;
    ID3D12Resource* vres = r->vbuf ? s_buf[r->vbuf - 1] : stage;
    if (!vres) return 0;
    /* Tables first: a full descriptor ring flushes the list and rebinds. */
    D3D12_CPU_DESCRIPTOR_HANDLE srv[ENG_SRV_TABLE];
    for (u32 u = 0; u < ENG_SRV_TABLE; u++) srv[u] = obj_srv(0);
    srv[0] = obj_srv(T);
    u32 vt_h[RSX_BE_MAX_VERTEX_TEXTURES] = {0};
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        EngObj* o = eng_owner(r->vtex[u]);
        if (o && o != t && o != sn) { srv[RSX_BE_MAX_TEXTURES + u] = obj_srv(r->vtex[u]); vt_h[u] = r->vtex[u]; }
    }
    const D3D12_GPU_DESCRIPTOR_HANDLE st = eng_srv_table(srv, ENG_SRV_TABLE);
    int smp[ENG_SRV_TABLE];
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) smp[u] = r->samp[u];
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) smp[RSX_BE_MAX_TEXTURES + u] = r->vsamp[u];
    const D3D12_GPU_DESCRIPTOR_HANDLE sm = eng_smp_table(smp);

    res_transition(t->res, &t->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    res_transition(sn->res, &sn->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) if (vt_h[u]) obj_transition(vt_h[u], ENG_SHADER_READ);
    bar_flush();
    eng_use_main_rootsig();
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = obj_rtv(S);
    CALL(s_list, OMSetRenderTargets, 1, &rtv, FALSE, NULL);
    CALL(s_list, SetPipelineState, pso);
    CALL(s_list, IASetPrimitiveTopology, topo_d3d(r->topology));
    D3D12_VIEWPORT vp; D3D12_RECT sc;
    eng_vp_sc(r, eng_obj_scaled(T), t->w, t->h, &vp, &sc);
    CALL(s_list, RSSetViewports, 1, &vp);
    CALL(s_list, RSSetScissorRects, 1, &sc);
    const D3D12_GPU_VIRTUAL_ADDRESS vva = r->vbuf ? CALL0(vres, GetGPUVirtualAddress) : stage_va;
    const u32 vbytes = r->vbuf ? s_buf_bytes[r->vbuf - 1] : s_stage_used;
    D3D12_VERTEX_BUFFER_VIEW vbv = { vva + r->vb_off, r->vb_off < vbytes ? vbytes - r->vb_off : 0, r->stride };
    if (vbv.SizeInBytes > (u64)r->vertex_count * r->stride && !r->index_count) vbv.SizeInBytes = r->vertex_count * r->stride;
    CALL(s_list, IASetVertexBuffers, 0, 1, &vbv);
    if (r->index_count) {
        D3D12_INDEX_BUFFER_VIEW ibv = { vva + r->ib_off, r->index_count * 4u, DXGI_FORMAT_R32_UINT };
        CALL(s_list, IASetIndexBuffer, &ibv);
    }
    const D3D12_GPU_VIRTUAL_ADDRESS zero_va = CALL0(s_zero_cb, GetGPUVirtualAddress);
    CALL(s_list, SetGraphicsRootConstantBufferView, 0, r->vs_cb_bytes ? stage_va + r->vs_cb_off : zero_va);
    CALL(s_list, SetGraphicsRootConstantBufferView, 4, zero_va);
    D3D12_GPU_DESCRIPTOR_HANDLE vt = st; vt.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_srv_step;
    D3D12_GPU_DESCRIPTOR_HANDLE vsm = sm; vsm.ptr += (UINT64)RSX_BE_MAX_TEXTURES * s_smp_step;
    CALL(s_list, SetGraphicsRootDescriptorTable, 1, st);
    CALL(s_list, SetGraphicsRootDescriptorTable, 3, vt);
    CALL(s_list, SetGraphicsRootDescriptorTable, 2, sm);
    CALL(s_list, SetGraphicsRootDescriptorTable, 5, vsm);
    if (r->index_count) CALL(s_list, DrawIndexedInstanced, r->index_count, 1, 0, 0, 0);
    else                CALL(s_list, DrawInstanced, r->vertex_count, 1, 0, 0);
    /* Everything the draw path caches is stale now. */
    s_cur_pso = NULL; s_cur_targets_valid = 0; s_srv_last_valid = 0;
    s_cur_topo = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    s_snap_copy_draws++;
    return 1;
}

static int src_ok_for_sync(u32 T, u32 S)
{
    EngObj* t = eng_owner(T); EngObj* sn = eng_owner(S);
    return t && sn && sn->has_rtv && t->w == sn->w && t->h == sn->h && t->fmt == sn->fmt;
}

static void eng_pass_mark(const EngRecord* r, u64* cur_key)
{
    int kind; u32 h1, h2 = 0;
    switch (r->kind) {
    case ENG_REC_DRAW:          kind = 0; h1 = r->nrt ? r->rt[0] : 0; h2 = r->depth; break;
    case ENG_REC_CLEAR_COLOR:   kind = 1; h1 = r->rt[0]; break;
    case ENG_REC_CLEAR_DS:
    case ENG_REC_CLEAR_DS_RECT: kind = 2; h1 = r->depth; break;
    case ENG_REC_COLOR_COPY:    kind = 3; h1 = r->resolve_dst; break;
    case ENG_REC_DEPTH_RESOLVE: kind = 4; h1 = r->resolve_dst; break;
    default:                    kind = 5; h1 = 0; break;
    }
    const u64 key = ((u64)kind << 56) | ((u64)(h1 & 0xFFFFFFu) << 24) | (h2 & 0xFFFFFFu);
    u32* n = &s_pass_n[s_slot];
    if (key != *cur_key && *n < ENG_PASS_MAX) {
        CALL(s_list, EndQuery, s_pheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * (ENG_PASS_MAX + 1u) + *n);
        EngPassDesc* d = &s_pass[s_slot][*n];
        memset(d, 0, sizeof *d);
        d->kind = (u8)kind; d->nrt = (u8)(kind == 0 ? r->nrt : 0);
        EngObj* o = eng_owner(h1 ? h1 : h2);
        if (o) { d->w = (u16)o->w; d->h = (u16)o->h; d->fmt = (u32)o->fmt; }
        (*n)++;
        *cur_key = key;
    }
    if (kind == 0 && *n) s_pass[s_slot][*n - 1].draws++;
}

static void eng_encode_records(ID3D12Resource* stage, D3D12_GPU_VIRTUAL_ADDRESS stage_va)
{
    u64 pass_key = ~0ull;
    /* g_rsx_snap_incr: -1 until read from RSX_SNAP_INCR; the host A/B (DOD3_AB=snapincr) flips it. */
    if (g_rsx_snap_incr < 0) { const char* e = getenv("RSX_SNAP_INCR"); g_rsx_snap_incr = !(e && e[0] == '0'); }
    const int incr = g_rsx_snap_incr;
    /* Per target with a snapshot: whether the snapshot was brought up to date
     * earlier in this submit, and the draws into the target since. */
    enum { SY_MAX = 8, SY_DRAWS = 16 };
    struct { u32 surf, snap, n; int valid; u32 idx[SY_DRAWS]; } sy[SY_MAX];
    u32 nsy = 0;
    for (u32 i = 0; i < s_rec_count; i++) {
        const EngRecord* r = &s_rec[i];
        if (s_pheap) eng_pass_mark(r, &pass_key);
        if (r->kind == ENG_REC_COLOR_COPY) {
            const u32 T = r->depth, S = r->resolve_dst;
            u32 k = 0;
            while (k < nsy && sy[k].surf != T) k++;
            int done = 0;
            EngObj* sobj = eng_owner(S);
            { static int dbg = -1; if (dbg < 0) dbg = getenv("RSX_SNAP_DEBUG") ? 1 : 0;
              static unsigned nd = 0;
              if (dbg && nd++ < 60)
                  fprintf(stderr, "[snap-debug] rec %u T=%u S=%u: entry %s valid %d snap-match %d rtv %d pending %u\n",
                          i, T, S, k < nsy ? "yes" : "no", k < nsy ? sy[k].valid : -1, k < nsy ? (sy[k].snap == S) : -1,
                          sobj ? sobj->has_rtv : -1, k < nsy ? sy[k].n : 0); }
            int ready = 1;
            if (incr && k < nsy && sy[k].valid && sy[k].snap == S && sobj && sobj->has_rtv) {
                /* Every copy pipeline ready, or the full copy this time. */
                for (u32 j = 0; j < sy[k].n; j++)
                    if (!eng_copy_pso(s_rec[sy[k].idx[j]].pipeline, topo_class(s_rec[sy[k].idx[j]].topology), sobj->fmt)) { ready = 0; break; }
            }
            if (incr && ready && k < nsy && sy[k].valid && sy[k].snap == S && sobj && sobj->has_rtv) {
                done = 1;
                for (u32 j = 0; j < sy[k].n; j++)
                    if (!eng_encode_copy_draw(&s_rec[sy[k].idx[j]], T, S, stage, stage_va)) { done = 0; break; }
            }
            if (done) {
                s_snap_incr_n++;
                static int said = 0;
                if (said++ < 2)
                    fprintf(stderr, "[rsx engine/d3d12] own-target snapshot brought up to date by %u copy draw(s) instead of a full copy\n",
                            sy[k].n);
            }
            else {
                EngObj* src = eng_owner(T); EngObj* dst = sobj;
                if (src && dst) {
                    res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_COPY_DEST);
                    bar_flush();
                    CALL(s_list, CopyResource, dst->res, src->res);
                    s_snap_full_n++;
                }
            }
            if (k == nsy && nsy < SY_MAX) nsy++;
            if (k < nsy) { sy[k].surf = T; sy[k].snap = S; sy[k].n = 0; sy[k].valid = (src_ok_for_sync(T, S)); }
            continue;
        }
        /* Writes to a tracked target: a draw as attachment A is replayable,
         * anything else is not. */
        for (u32 k = 0; k < nsy; k++) {
            if (!sy[k].valid) continue;
            const u32 T = sy[k].surf, S = sy[k].snap;
            switch (r->kind) {
            case ENG_REC_DRAW:
                for (u32 a = 0; a < r->nrt && a < RSX_BE_MAX_COLOR_TARGETS; a++) {
                    if (r->rt[a] == S) sy[k].valid = 0;
                    if (r->rt[a] != T) continue;
                    if (a == 0 && sy[k].n < SY_DRAWS) sy[k].idx[sy[k].n++] = i;
                    else sy[k].valid = 0;
                }
                break;
            case ENG_REC_CLEAR_COLOR:
                if (r->rt[0] == T || r->rt[0] == S) sy[k].valid = 0;
                break;
            case ENG_REC_DEPTH_RESOLVE:
                if (r->resolve_dst == T || r->resolve_dst == S) sy[k].valid = 0;
                break;
            case ENG_REC_UPLOAD:
                if (r->depth == T || r->depth == S) sy[k].valid = 0;
                break;
            default:
                break;
            }
        }
        switch (r->kind) {
        case ENG_REC_DRAW:
            eng_encode_draw(r, stage, stage_va);
            break;
        case ENG_REC_UPLOAD:
            eng_encode_upload(r, stage);
            break;
        case ENG_REC_CLEAR_COLOR: {
            EngObj* o = eng_owner(r->rt[0]);
            if (!o || !o->has_rtv) break;
            obj_transition(r->rt[0], D3D12_RESOURCE_STATE_RENDER_TARGET);
            bar_flush();
            CALL(s_list, ClearRenderTargetView, obj_rtv(r->rt[0]), r->clear_rgba, 0, NULL);
            break; }
        case ENG_REC_CLEAR_DS:
        case ENG_REC_CLEAR_DS_RECT: {
            EngObj* z = eng_owner(r->depth);
            if (!z || !z->has_dsv) break;
            D3D12_CLEAR_FLAGS f = 0;
            if (r->clear_flags & RSX_BE_CLEAR_DEPTH)   f |= D3D12_CLEAR_FLAG_DEPTH;
            if (r->clear_flags & RSX_BE_CLEAR_STENCIL) f |= D3D12_CLEAR_FLAG_STENCIL;
            if (!f) break;
            obj_transition(r->depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            bar_flush();
            if (r->kind == ENG_REC_CLEAR_DS_RECT) {
                u32 x = r->clear_rect[0], y = r->clear_rect[1], w = r->clear_rect[2], h = r->clear_rect[3];
                if (x >= z->w || y >= z->h) break;
                if (x + w > z->w) w = z->w - x;
                if (y + h > z->h) h = z->h - y;
                D3D12_RECT rc = { (LONG)x, (LONG)y, (LONG)(x + w), (LONG)(y + h) };
                CALL(s_list, ClearDepthStencilView, obj_dsv(r->depth), f, r->clear_depth, r->clear_stencil, 1, &rc);
            } else {
                CALL(s_list, ClearDepthStencilView, obj_dsv(r->depth), f, r->clear_depth, r->clear_stencil, 0, NULL);
            }
            break; }
        case ENG_REC_COLOR_COPY: {
            EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
            if (!src || !dst) break;
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_COPY_DEST);
            bar_flush();
            CALL(s_list, CopyResource, dst->res, src->res);
            break; }
        case ENG_REC_DEPTH_RESOLVE: {
            EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
            ID3D12PipelineState* pso = r->resolve_packed ? s_depth_pack_pso : s_depth_pso;
            if (!src || !dst || !dst->has_rtv || !pso) break;
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            res_transition(dst->res, &dst->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
            bar_flush();
            eng_fullscreen_pass(pso, obj_rtv(r->resolve_dst), dst->w, dst->h, obj_srv(r->depth));
            break; }
        }
    }
    /* Close the last pass. */
    if (s_pheap && s_pass_n[s_slot])
        CALL(s_list, EndQuery, s_pheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * (ENG_PASS_MAX + 1u) + s_pass_n[s_slot]);
}

/* ---- submit ---------------------------------------------------------------- */

static ID3D12Resource* s_cur_stage_res; static D3D12_GPU_VIRTUAL_ADDRESS s_cur_stage_va;

/* Close and execute the open list; signal its fence. Rings and queries of
 * the list are the caller's to account for. */
static void eng_submit_list(u64* out_fence)
{
    bar_flush();
    if (s_q_used && s_qread)
        CALL(s_list, ResolveQueryData, s_qheap, D3D12_QUERY_TYPE_OCCLUSION, s_slot * ENG_QUERIES_PER_SLOT, s_q_used,
             s_qread, (UINT64)s_slot * ENG_QUERIES_PER_SLOT * 8u);
    if (s_pheap && s_pass_n[s_slot])
        CALL(s_list, ResolveQueryData, s_pheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * (ENG_PASS_MAX + 1u), s_pass_n[s_slot] + 1u,
             s_pread, (UINT64)s_slot * (ENG_PASS_MAX + 1u) * 8u);
    if (s_tsheap) {
        CALL(s_list, EndQuery, s_tsheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * 2u + 1u);
        CALL(s_list, ResolveQueryData, s_tsheap, D3D12_QUERY_TYPE_TIMESTAMP, s_slot * 2u, 2, s_tsread, (UINT64)s_slot * 16u);
    }
    HRESULT hr = CALL0(s_list, Close);
    s_list_open = 0;
    if (FAILED(hr)) {
        fprintf(stderr, "[rsx engine/d3d12] command list Close failed: 0x%08lX (removed 0x%08lX)\n",
                (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason));
        s_ready = 0; *out_fence = 0; return;
    }
    ID3D12CommandList* lists[1] = { (ID3D12CommandList*)s_list };
    if (s_tsheap) QueryPerformanceCounter(&s_ts_slot[s_slot].cpu_exec);
    CALL(s_queue, ExecuteCommandLists, 1, lists);
    *out_fence = fence_signal();
    s_alloc_fence[s_slot] = *out_fence;
    if (s_tsheap) { s_ts_slot[s_slot].fence = *out_fence; s_ts_slot[s_slot].kind = s_submit_kind; }
}

/* A completed submit: its query counts and the reports waiting on them. */
static void eng_poll_submits(void)
{
    for (;;) {
        /* Oldest first: a report rides on the submit open when the walker met
         * it, and the queries it counts may be in an earlier one, whose counts
         * have to be in before it reads them. In table order a later submit in
         * a lower slot was delivered first. */
        EngSubmit* s = NULL;
        for (int i = 0; i < ENG_MAX_SUBMITS; i++)
            if (s_sub[i].live && (!s || s_sub[i].fence < s->fence)) s = &s_sub[i];
        if (!s || !fence_done(s->fence)) break;
        if (s->q_count && s_qread) {
            D3D12_RANGE rg = { (SIZE_T)s->q_slot * ENG_QUERIES_PER_SLOT * 8u, ((SIZE_T)s->q_slot * ENG_QUERIES_PER_SLOT + s->q_count) * 8u };
            u64* p = NULL;
            if (SUCCEEDED(CALL(s_qread, Map, 0, &rg, (void**)&p))) {
                for (u32 k = 0; k < s->q_count; k++) s_vis_count[s->q_vis[k] & ~ENG_Q_SCALED] += eng_q_count(s->q_vis[k], p[s->q_slot * ENG_QUERIES_PER_SLOT + k]);
                D3D12_RANGE wr = {0, 0}; CALL(s_qread, Unmap, 0, &wr);
            }
        }
        for (u32 k = 0; k < s->n_reports; k++) {
            rsx_draw_engine_query_result(s->reports[k].index, s_vis_count[s->reports[k].slot]);
        }
        free(s->q_vis); free(s->reports);
        memset(s, 0, sizeof *s);
    }
}

static void eng_dump_frame(u32 surface);

static void eng_swapchain_release(void)
{
    if (!s_swap) return;
    if (s_display == DISP_FULLSCREEN) CALL(s_swap, SetFullscreenState, FALSE, NULL);
    for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
    RELEASE(s_swap);
}

/* RSX_DISPLAY / RSX_FULLSCREEN / RSX_WINDOW -> s_display and the window
 * size: a windowed window of RSX_WINDOW (else the base size), the others
 * over the display at its current mode. */
static void eng_read_display(void)
{
    const char* dm = getenv("RSX_DISPLAY");
    const char* fs = getenv("RSX_FULLSCREEN");
    s_display = DISP_WINDOWED;
    if (dm && *dm) {
        if (!_stricmp(dm, "borderless")) s_display = DISP_BORDERLESS;
        else if (!_stricmp(dm, "fullscreen") || !_stricmp(dm, "exclusive")) s_display = DISP_FULLSCREEN;
        else if (_stricmp(dm, "windowed") && _stricmp(dm, "window"))
            fprintf(stderr, "[RSX d3d12] RSX_DISPLAY=%s ignored (windowed, borderless or fullscreen)\n", dm);
    } else if (fs && *fs && *fs != '0') s_display = DISP_BORDERLESS;
    const char* ws = getenv("RSX_WINDOW");
    unsigned ww = 0, wh = 0;
    const int have_ws = ws && *ws && sscanf(ws, "%ux%u", &ww, &wh) == 2 && ww >= 320 && wh >= 240 && ww <= 16384 && wh <= 16384;
    if (ws && *ws && !have_ws) fprintf(stderr, "[RSX d3d12] RSX_WINDOW=%s ignored (want <w>x<h>)\n", ws);
    s_win_w = s_width; s_win_h = s_height;
    if (s_display == DISP_WINDOWED) { if (have_ws) { s_win_w = ww; s_win_h = wh; } }
    else { s_win_w = (u32)GetSystemMetrics(SM_CXSCREEN); s_win_h = (u32)GetSystemMetrics(SM_CYSCREEN); }
}

static int eng_read_vsync(void)
{
    const char* v = getenv("RSX_VSYNC");
    return (v && *v) ? (atoi(v) ? 1 : 0) : 1;
}

/* g_rsx_display_reload (rsx_draw_engine.c): the settings page changed
 * RSX_VSYNC / RSX_DISPLAY / RSX_WINDOW / RSX_SCALE. Picked up by the message pump, on
 * the thread that owns the window and presents. V-sync alone is the next
 * present's interval; a new mode or size drains the GPU, restyles the
 * window and makes a new swap chain (a frame or two of black). */
static void eng_rescale(void);   /* below */
static void eng_display_reload(void)
{
    g_rsx_display_reload = 0;
    const EngDisplayMode od = s_display;
    const u32 ow = s_win_w, oh = s_win_h;
    const int ov = s_vsync;
    const float os = s_scale;
    s_vsync = eng_read_vsync();
    eng_read_display();
    eng_read_scale();
    if (s_scale != os) eng_rescale();
    if (s_vsync != ov) fprintf(stderr, "[RSX d3d12] vsync %d\n", s_vsync);
    if (s_display == od && s_win_w == ow && s_win_h == oh) return;
    fence_wait(fence_signal());
    const EngDisplayMode nd = s_display;
    s_display = od;
    eng_swapchain_release();
    s_display = nd;
    if (s_display == DISP_WINDOWED) {
        RECT wr = {0, 0, (LONG)s_win_w, (LONG)s_win_h};
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
        const int w = wr.right - wr.left, h = wr.bottom - wr.top;
        const int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2, y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
        SetWindowLongPtrA(s_hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
        SetWindowPos(s_hwnd, HWND_NOTOPMOST, x > 0 ? x : 0, y > 0 ? y : 0, w, h, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    } else {
        SetWindowLongPtrA(s_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(s_hwnd, HWND_TOP, 0, 0, (int)s_win_w, (int)s_win_h, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    }
    IDXGIFactory4* factory = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory)) || eng_swapchain_create(factory) != 0)
        fprintf(stderr, "[RSX d3d12] could not make a swap chain for the new display mode\n");
    RELEASE(factory);
    fprintf(stderr, "[RSX d3d12] display: %s %ux%u, vsync %d\n",
            s_display == DISP_FULLSCREEN ? "full screen" : s_display == DISP_BORDERLESS ? "borderless" : "windowed",
            s_win_w, s_win_h, s_vsync);
}

/* RSX_SCALE changed while running (the reload below). With the GPU idle:
 * every colour and depth target at the internal resolution is made again at
 * the new one under the same handle (the views cut from it pointed at the
 * new resource), the pipelines whose fragment program reads WPOS get that
 * program rebuilt for the new divisor, and the pools of host-sized
 * scratch targets are emptied. The targets come back empty: the title draws
 * nearly all of them every frame, so this is a frame or two of black. */
static void eng_rescale(void)
{
    fence_wait(fence_signal());
    AcquireSRWLockExclusive(&s_pipe_lock);
    u32 np = 0;
    for (u32 i = 0; i < s_pipe_count; i++) {
        EngPipeline* p = &s_pipe[i];
        if (!p->ps_plain || !p->vs) continue;
        ID3DBlob* ps = eng_fp_blob(p->ps_plain);
        if (!ps) continue;
        CALL0(ps, AddRef);
        RELEASE(p->ps);
        p->ps = ps;
        p->desc.PS.pShaderBytecode = CALL0(ps, GetBufferPointer);
        p->desc.PS.BytecodeLength = CALL0(ps, GetBufferSize);
        for (int c = 0; c < 3; c++) { RELEASE(p->pso[c]); p->failed[c] = 0; }
        np++;
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    for (u32 i = 0; i < s_fallback_n; i++) eng_obj_release(NULL, s_fallback_depth[i]);
    s_fallback_n = 0;
    for (u32 i = 0; i < s_snap_pool_n; i++) RELEASE(s_snap_pool[i].res);
    s_snap_pool_n = 0;
    u32 nt = 0;
    for (u32 i = 0; i < s_obj_count; i++) {
        EngObj* o = &s_obj[i];
        const u32 h = i + 1;
        if (o->retired || !o->res || !o->gw || !o->gh) continue;
        if (o->kind != OBJ_COLOR && o->kind != OBJ_DEPTH) continue;
        const int was = o->w != o->gw || o->h != o->gh;
        const int now = eng_scales(o->gw, o->gh);
        if (!was && !now) continue;
        const u32 nw = now ? sc_dim(o->gw) : o->gw, nh = now ? sc_dim(o->gh) : o->gh;
        if (nw == o->w && nh == o->h) continue;
        RELEASE(o->res);
        D3D12_CLEAR_VALUE cv = {0};
        if (o->kind == OBJ_COLOR) {
            cv.Format = o->fmt;
            o->res = make_texture(o->fmt, nw, nh, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                  D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
            o->state = D3D12_RESOURCE_STATE_RENDER_TARGET;
            if (!o->res) continue;
            eng_write_srv(h, o->res, o->fmt, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
            CALL(s_dev, CreateRenderTargetView, o->res, NULL, obj_rtv(h));
        } else {
            cv.Format = ENG_DEPTH_FMT; cv.DepthStencil.Depth = 1.0f;
            o->res = make_texture(ENG_DEPTH_RES_FMT, nw, nh, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                                  D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv);
            o->state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            if (!o->res) continue;
            D3D12_DEPTH_STENCIL_VIEW_DESC dd = {0};
            dd.Format = ENG_DEPTH_FMT; dd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            CALL(s_dev, CreateDepthStencilView, o->res, &dd, obj_dsv(h));
            eng_write_srv(h, o->res, ENG_DEPTH_SRV_FMT, 1, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
        }
        o->w = nw; o->h = nh;
        for (u32 v = 0; v < s_view_count; v++)
            if (s_view[v].surface == h) {
                eng_write_srv(s_view[v].view, o->res, o->fmt, 1, 1, eng_mapping(s_view[v].remap, s_view[v].format));
                EngObj* vo = eng_obj(s_view[v].view);
                if (vo) { vo->w = nw; vo->h = nh; }
            }
        nt++;
    }
    fprintf(stderr, "[RSX d3d12] internal resolution x%.2f (%ux%u): %u targets and %u programs rebuilt\n",
            s_scale, sc_dim(s_width), sc_dim(s_height), nt, np);
}

/* Back in exclusive full screen after a focus loss. The GPU is drained first
 * (the back buffers are released and recreated), so this costs a frame. */
static void eng_fullscreen_restore(void)
{
    s_fs_restore = 0;
    if (!s_swap || s_headless) return;
    BOOL fs = FALSE;
    if (FAILED(CALL(s_swap, GetFullscreenState, &fs, NULL)) || fs) return;
    fence_wait(fence_signal());
    if (FAILED(CALL(s_swap, SetFullscreenState, TRUE, NULL))) {
        fprintf(stderr, "[rsx engine/d3d12] could not re-enter full screen; staying windowed\n");
        return;
    }
    for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
    HRESULT hr = CALL(s_swap, ResizeBuffers, 0, 0, 0, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);
    if (FAILED(hr)) { fprintf(stderr, "[rsx engine/d3d12] ResizeBuffers after full screen failed: 0x%08lX\n", (long)hr); return; }
    for (u32 i = 0; i < 3; i++) {
        if (FAILED(CALL(s_swap, GetBuffer, i, &IID_ID3D12Resource, (void**)&s_backbuf[i]))) return;
        CALL(s_dev, CreateRenderTargetView, s_backbuf[i], NULL, cpu_handle(s_backbuf_rtv_heap, s_rtv_step, i));
    }
}

/* s_walker_tid (above eng_pso): the thread that encodes -- PSO time there is walker time. */
static void eng_submit(u32 present_surface, int wait)
{
    if (!s_walker_tid) s_walker_tid = GetCurrentThreadId();
    if (!s_ready) { s_rec_count = 0; s_stage_used = 0; s_stage_cur = -1; s_vis_npending = 0; return; }
    if (s_fs_restore && present_surface) eng_fullscreen_restore();
    if (!s_rec_count && !present_surface) return;
    eng_poll_submits();

    s_slot = s_submit_seq % ENG_FRAMES;
    fence_wait(s_alloc_fence[s_slot]);
    eng_list_begin();

    ID3D12Resource* stage = s_stage_cur >= 0 ? s_stage[s_stage_cur].res : NULL;
    const D3D12_GPU_VIRTUAL_ADDRESS stage_va = stage ? CALL0(stage, GetGPUVirtualAddress) : 0;
    eng_encode_records(stage, stage_va);

    u32 bbi = 0; int windowed = 0;
    if (present_surface) {
        EngObj* src = eng_owner(present_surface);
        if (src && s_blit_pso) {
            ID3D12Resource* dst; D3D12_CPU_DESCRIPTOR_HANDLE rtv;
            D3D12_RESOURCE_STATES before;
            if (s_headless) { dst = s_offscreen; rtv = cpu_handle(s_backbuf_rtv_heap, s_rtv_step, 0); before = s_offscreen_state; }
            else {
                bbi = CALL0(s_swap, GetCurrentBackBufferIndex);
                dst = s_backbuf[bbi]; rtv = cpu_handle(s_backbuf_rtv_heap, s_rtv_step, bbi);
                before = D3D12_RESOURCE_STATE_PRESENT; windowed = 1;
            }
            D3D12_RESOURCE_STATES st = before;
            res_transition(dst, &st, D3D12_RESOURCE_STATE_RENDER_TARGET);
            res_transition(src->res, &src->state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            bar_flush();
            eng_fullscreen_pass(s_blit_pso, rtv, s_win_w, s_win_h, obj_srv(present_surface));
            if (windowed) res_transition(dst, &st, D3D12_RESOURCE_STATE_PRESENT);
            else s_offscreen_state = st;
            bar_flush();
        }
    }

    /* The submit record: what completion has to deliver. */
    EngSubmit* sub = NULL;
    for (int i = 0; i < ENG_MAX_SUBMITS && !sub; i++) if (!s_sub[i].live) sub = &s_sub[i];
    if (!sub) {   /* every slot in flight: wait for the oldest */
        int oldest = 0;
        for (int i = 1; i < ENG_MAX_SUBMITS; i++) if (s_sub[i].fence < s_sub[oldest].fence) oldest = i;
        fence_wait(s_sub[oldest].fence);
        eng_poll_submits();
        sub = &s_sub[oldest];
    }
    memset(sub, 0, sizeof *sub);
    sub->q_slot = s_slot; sub->q_count = s_q_used;
    if (s_q_used) { sub->q_vis = (u32*)malloc(s_q_used * sizeof(u32)); if (sub->q_vis) memcpy(sub->q_vis, s_q_vis, s_q_used * sizeof(u32)); else sub->q_count = 0; }
    if (s_vis_npending) {
        sub->reports = (EngVisReport*)malloc(s_vis_npending * sizeof(EngVisReport));
        if (sub->reports) { memcpy(sub->reports, s_vis_pending, s_vis_npending * sizeof(EngVisReport)); sub->n_reports = s_vis_npending; }
        s_vis_npending = 0;
    }
    u64 f = 0;
    s_submit_kind = wait;
    eng_submit_list(&f);
    s_submit_kind = 0;
    sub->fence = f; sub->live = 1; sub->stage = s_stage_cur;
    if (s_stage_cur >= 0) { s_stage[s_stage_cur].fence = f; s_stage[s_stage_cur].in_use = 1; }
    s_stage_cur = -1; s_stage_used = 0;
    s_rec_count = 0;
    s_submit_seq++;

    if (windowed) {
        /* RSX_HITCH_LOG=<ms>: a Present or a pacing wait that took more than
         * a third of that, with its time -- whether a long frame was spent
         * here, in the display path, or before the walker had the frame. */
        static double hl = -1.0;
        if (hl < 0.0) { const char* e = getenv("RSX_HITCH_LOG"); hl = e ? atof(e) : 0.0; }
        LARGE_INTEGER pq0, pq1, pq2, pqf;
        if (hl > 0.0) { QueryPerformanceFrequency(&pqf); QueryPerformanceCounter(&pq0); }
        HRESULT hr = CALL(s_swap, Present, (UINT)s_vsync,
                          (!s_vsync && s_tearing && s_display != DISP_FULLSCREEN) ? DXGI_PRESENT_ALLOW_TEARING : 0);
        if (FAILED(hr)) {
            static int n = 0;
            if (n++ < 8) fprintf(stderr, "[rsx engine/d3d12] Present failed: 0x%08lX (removed 0x%08lX)\n",
                                 (long)hr, (long)CALL0(s_dev, GetDeviceRemovedReason));
        }
        if (hl > 0.0) QueryPerformanceCounter(&pq1);
        /* Pace: at most ENG_INFLIGHT presents queued. */
        if (f > ENG_INFLIGHT) fence_wait(f - ENG_INFLIGHT);
        if (hl > 0.0) {
            QueryPerformanceCounter(&pq2);
            const double pres = (double)(pq1.QuadPart - pq0.QuadPart) * 1000.0 / (double)pqf.QuadPart;
            const double pace = (double)(pq2.QuadPart - pq1.QuadPart) * 1000.0 / (double)pqf.QuadPart;
            if (pres > hl / 3.0 || pace > hl / 3.0)
                fprintf(stderr, "[hitch-present] submit %llu: Present %.1f ms, pacing wait %.1f ms\n",
                        (unsigned long long)f, pres, pace);
        }
    }
    if (wait || (present_surface && s_headless)) fence_wait(f);
    if (s_dropped) {
        fprintf(stderr, "[rsx engine/d3d12] dropped %u record(s) (cap %d)\n", s_dropped, ENG_MAX_RECORDS);
        s_dropped = 0;
    }
    eng_poll_submits();
    eng_collect_retired();
    eng_collect_retired_buffers();
}

static void eng_submit_and_wait(void* user, u32 reason)
{
    (void)user; (void)reason;
    eng_submit(0, 1);
}

/* ---- readback and present -------------------------------------------------- */

static ID3D12CommandAllocator* s_aux_alloc;
static ID3D12Resource* s_readback; static u32 s_readback_cap;

/* Diagnostics only (present log, trace triggers, surface dumps, frame grabs):
 * wait for the GPU so the bytes are the frame just submitted. */
static void eng_readback(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h, void* out, u32 out_pitch)
{
    (void)user;
    EngObj* o = eng_owner(surface);
    if (!o || !out || !out_pitch || !w || !h || !s_ready) return;
    /* The caller's rectangle is in guest pixels; the texture may be larger.
     * Read the matching host rectangle and resample it (nearest) on the way
     * out, so the caller sees the surface at the size it asked for. */
    const u32 gw = w, gh = h;
    const int scaled = o->gw && o->gh && (o->gw != o->w || o->gh != o->h);
    if (scaled) {
        if (x + w > o->gw || y + h > o->gh) return;
        const u32 x0 = (u32)((u64)x * o->w / o->gw), x1 = (u32)((u64)(x + w) * o->w / o->gw);
        const u32 y0 = (u32)((u64)y * o->h / o->gh), y1 = (u32)((u64)(y + h) * o->h / o->gh);
        x = x0; y = y0; w = x1 > x0 ? x1 - x0 : 1u; h = y1 > y0 ? y1 - y0 : 1u;
    }
    if (x + w > o->w || y + h > o->h) return;
    const u32 bpp = eng_bpp(o->fmt);
    const u32 pitch = (w * bpp + 255u) & ~255u;
    const u32 need = pitch * h;
    if (need > s_readback_cap) {
        RELEASE(s_readback);
        s_readback = make_buffer(D3D12_HEAP_TYPE_READBACK, need, D3D12_RESOURCE_STATE_COPY_DEST);
        s_readback_cap = s_readback ? need : 0;
        if (!s_readback) return;
    }
    if (!s_aux_alloc && FAILED(CALL(s_dev, CreateCommandAllocator, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&s_aux_alloc)))
        return;
    /* Everything submitted so far has to have run, including whatever list
     * last used the aux allocator. */
    fence_wait(s_fence_value);
    CALL0(s_aux_alloc, Reset);
    CALL(s_list, Reset, s_aux_alloc, NULL);
    res_transition(o->res, &o->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    bar_flush();
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    src.pResource = o->res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
    dst.pResource = s_readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = o->fmt;
    dst.PlacedFootprint.Footprint.Width = w; dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = pitch;
    D3D12_BOX box = { x, y, 0, x + w, y + h, 1 };
    CALL(s_list, CopyTextureRegion, &dst, 0, 0, 0, &src, &box);
    CALL0(s_list, Close);
    ID3D12CommandList* lists[1] = { (ID3D12CommandList*)s_list };
    CALL(s_queue, ExecuteCommandLists, 1, lists);
    fence_wait(fence_signal());
    u8* p = NULL; D3D12_RANGE rg = { 0, need };
    if (SUCCEEDED(CALL(s_readback, Map, 0, &rg, (void**)&p))) {
        if (!scaled) {
            for (u32 r = 0; r < h; r++) memcpy((u8*)out + (size_t)r * out_pitch, p + (size_t)r * pitch, (size_t)w * bpp);
        } else {
            for (u32 r = 0; r < gh; r++) {
                const u8* srow = p + (size_t)(((u64)r * h + h / 2) / gh) * pitch;
                u8* drow = (u8*)out + (size_t)r * out_pitch;
                for (u32 c = 0; c < gw; c++)
                    memcpy(drow + (size_t)c * bpp, srow + (size_t)(((u64)c * w + w / 2) / gw) * bpp, bpp);
            }
        }
        D3D12_RANGE wr = {0, 0}; CALL(s_readback, Unmap, 0, &wr);
    }
}

/* Opt-in readback of the presented surface to a PPM, as the Metal backend:
 * RSX_REPLAY_DUMP_PATH (tools/rsx_replay names each file), PS3RECOMP_FRAME_GRAB
 * (<prefix>.req asks for one <prefix>.ppm; tools/autoplay.sh's HUD check) and
 * PS3RECOMP_FRAME_DUMP every PS3RECOMP_FRAME_EVERY presents. The METAL-named
 * variables are honoured too, so the Mac tools run unchanged. */
static const char* env2(const char* a, const char* b)
{
    const char* v = getenv(a);
    if (!v || !*v) v = getenv(b);
    return (v && *v) ? v : NULL;
}
static void eng_dump_frame(u32 surface)
{
    EngObj* o = eng_owner(surface);
    if (!o || o->fmt != DXGI_FORMAT_R8G8B8A8_UNORM) return;
    const char* path = getenv("RSX_REPLAY_DUMP_PATH");
    if (path && !*path) path = NULL;
    char seqpath[1024], grab_req[1024], grab_tmp[1024], grab_out[1024];
    int grabbing = 0;
    static const char* grab = (const char*)1;
    if (grab == (const char*)1) grab = env2("PS3RECOMP_FRAME_GRAB", "PS3RECOMP_METAL_FRAME_GRAB");
    if (!path && grab) {
        snprintf(grab_req, sizeof grab_req, "%s.req", grab);
        if (_access(grab_req, 0) == 0) {
            snprintf(grab_tmp, sizeof grab_tmp, "%s.ppm.tmp", grab);
            snprintf(grab_out, sizeof grab_out, "%s.ppm", grab);
            path = grab_tmp; grabbing = 1;
        }
    }
    if (!path) {
        static const char* dump = (const char*)1;
        if (dump == (const char*)1) dump = env2("PS3RECOMP_FRAME_DUMP", "PS3RECOMP_METAL_FRAME_DUMP");
        if (!dump) return;
        static unsigned frame, every; static int seq = -1;
        if (!every) { const char* e = env2("PS3RECOMP_FRAME_EVERY", "PS3RECOMP_METAL_FRAME_EVERY"); every = e ? (unsigned)atoi(e) : 120u; if (!every) every = 120u; }
        if (seq < 0) seq = env2("PS3RECOMP_FRAME_SEQ", "PS3RECOMP_METAL_FRAME_SEQ") ? 1 : 0;
        if ((++frame % every) != 0) return;
        path = dump;
        if (seq) { snprintf(seqpath, sizeof seqpath, "%s.%06u.ppm", dump, frame); path = seqpath; }
    }
    const u32 w = o->gw ? o->gw : o->w, h = o->gh ? o->gh : o->h;
    if (!w || !h || w > 16384 || h > 16384) return;
    u8* rgba = (u8*)malloc((size_t)w * h * 4);
    u8* rgb = (u8*)malloc((size_t)w * h * 3);
    if (!rgba || !rgb) { free(rgba); free(rgb); return; }
    eng_readback(NULL, surface, 0, 0, w, h, rgba, w * 4);
    for (size_t p = 0; p < (size_t)w * h; p++) { rgb[p*3] = rgba[p*4]; rgb[p*3+1] = rgba[p*4+1]; rgb[p*3+2] = rgba[p*4+2]; }
    FILE* f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%u %u\n255\n", w, h);
        fwrite(rgb, 1, (size_t)w * h * 3, f);
        fclose(f);
        if (grabbing) { remove(grab_out); rename(grab_tmp, grab_out); remove(grab_req); }
        else fprintf(stderr, "[rsx engine/d3d12] captured frame to %s\n", path);
    }
    free(rgba); free(rgb);
}

static void eng_present(void* user, u32 surface)
{
    (void)user;
    { static int on = -1; static unsigned n = 0;
      if (on < 0) on = getenv("RSX_OBJ_STATS") ? 1 : 0;
      if (on && (++n % 600) == 0) {
          fprintf(stderr, "[rsx engine/d3d12] objects: %u slots used, %u free (present %u); per 600 presents: %u textures, %u targets, %u snapshots (+%u reused, pool %u), %u vertex buffers created\n",
                  s_obj_count, s_obj_free_count, n, s_stat_tex, s_stat_rt, s_stat_snap, s_stat_snap_reused, s_snap_pool_n, s_stat_buf);
          s_stat_tex = s_stat_rt = s_stat_snap = s_stat_snap_reused = s_stat_buf = 0; } }
    if (!eng_owner(surface)) { eng_submit(0, 0); return; }
    eng_submit(surface, 0);
    eng_dump_frame(surface);
}

static const rsx_draw_backend s_engine_backend = {
    .user                 = NULL,
    .init                 = eng_init,
    .shutdown             = eng_shutdown,
    .submit_and_wait      = eng_submit_and_wait,
    .texture_create       = eng_texture_create,
    .texture_upload       = eng_texture_upload,
    .texture_release      = eng_obj_release,
    .color_target_create  = eng_color_target_create,
    .color_target_release = eng_obj_release,
    .surface_view         = eng_surface_view,
    .depth_target_create  = eng_depth_target_create,
    .depth_target_release = eng_obj_release,
    .depth_snapshot       = eng_depth_snapshot,
    .depth_snapshot_rgba8 = eng_depth_snapshot_rgba8,
    .color_snapshot       = eng_color_snapshot,
    .query_begin          = eng_query_begin,
    .query_set            = eng_query_set,
    .query_report         = eng_query_report,
    .pipeline_create      = eng_pipeline_create,
    .pipeline_release     = eng_pipeline_release,
    .bind_targets         = eng_bind_targets,
    .bind_pipeline        = eng_bind_pipeline,
    .bind_vs_constants    = eng_bind_vs_constants,
    .reuse_vs_constants   = eng_reuse_vs_constants,
    .bind_ps_constants    = eng_bind_ps_constants,
    .bind_textures        = eng_bind_textures,
    .bind_vertex_textures = eng_bind_vertex_textures,
    .set_viewport         = eng_set_viewport,
    .set_scissor          = eng_set_scissor,
    .set_stencil_ref      = eng_set_stencil_ref,
    .draw                 = eng_draw,
    .buffer_wrap          = eng_buffer_wrap,
    .buffer_release       = eng_buffer_release,
    .draw_buffer          = eng_draw_buffer,
    .clear_color          = eng_clear_color,
    .clear_depth_stencil  = eng_clear_depth_stencil,
    .clear_depth_stencil_rect = eng_clear_depth_stencil_rect,
    .present              = eng_present,
    .readback             = eng_readback,
};

/* ---- public API -------------------------------------------------------------- */

static void eng_release_device(void)
{
    if (s_swap && s_display == DISP_FULLSCREEN) CALL(s_swap, SetFullscreenState, FALSE, NULL);
    RELEASE(s_zero_cb); RELEASE(s_readback); s_readback_cap = 0; RELEASE(s_aux_alloc);
    RELEASE(s_null_tex); RELEASE(s_blit_pso); RELEASE(s_depth_pso); RELEASE(s_depth_pack_pso);
    RELEASE(s_rootsig); RELEASE(s_helper_rootsig);
    RELEASE(s_qheap); RELEASE(s_qread); RELEASE(s_tsheap); RELEASE(s_tsread); RELEASE(s_pheap); RELEASE(s_pread);
    for (u32 i = 0; i < ENG_FRAMES; i++) { RELEASE(s_smp_gpu[i]); RELEASE(s_alloc[i]); s_alloc_fence[i] = 0; }
    RELEASE(s_srv_gpu); RELEASE(s_srv_cpu); RELEASE(s_rtv_cpu); RELEASE(s_dsv_cpu); RELEASE(s_smp_cpu);
    RELEASE(s_list);
    if (s_fence_event) { CloseHandle(s_fence_event); s_fence_event = NULL; }
    RELEASE(s_fence); s_fence_value = 0;
    for (u32 i = 0; i < 3; i++) RELEASE(s_backbuf[i]);
    RELEASE(s_backbuf_rtv_heap); RELEASE(s_offscreen); RELEASE(s_swap);
    RELEASE(s_queue); RELEASE(s_dev);
    if (s_hwnd) { DestroyWindow(s_hwnd); s_hwnd = NULL; }
    s_ready = 0;
}

int rsx_d3d12_engine_init(u32 width, u32 height, const char* title)
{
    const char* hl = getenv("PS3RECOMP_D3D12_HEADLESS");
    s_headless = (hl && *hl && *hl != '0');
    s_width = width ? width : 1280; s_height = height ? height : 720;
    s_vsync = eng_read_vsync();
    eng_read_scale();
    { const char* e = getenv("RSX_BUF_POOL"); if (e && *e == '0') g_eng_buf_pool = 0; }
    /* Per-monitor DPI awareness, before any window exists: without it a
     * 4K display at 125% scaling reports 3072x1728, the borderless window
     * covers that and Windows stretches it, and a 1280x720 window is drawn
     * at 1600x900 through the desktop compositor's scaler. */
    { HMODULE u32 = GetModuleHandleA("user32.dll");
      typedef BOOL (WINAPI *SetCtxFn)(DPI_AWARENESS_CONTEXT);
      SetCtxFn set_ctx = u32 ? (SetCtxFn)(void*)GetProcAddress(u32, "SetProcessDpiAwarenessContext") : NULL;
      if (!set_ctx || !set_ctx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) SetProcessDPIAware(); }
    eng_read_display();

    /* The engine is this backend's default; PS3RECOMP_RSX_ENGINE=vtable is
     * the way back to the rsx_state path. The query needs a registered
     * backend to answer, so register first and undo on a no. */
    rsx_draw_engine_set_backend(&s_engine_backend);
    rsx_draw_engine_set_default(1);
    if (!rsx_draw_engine_enabled()) { rsx_draw_engine_set_backend(NULL); return -1; }

    if (!s_headless) {
        s_hwnd = eng_create_window(s_win_w, s_win_h, title);
        if (!s_hwnd) { fprintf(stderr, "[RSX d3d12] window creation failed\n"); rsx_draw_engine_set_backend(NULL); return -1; }
    }
    if (eng_init_device(s_win_w, s_win_h) != 0) {
        fprintf(stderr, "[RSX d3d12] device initialisation failed; falling back to the vtable path\n");
        eng_release_device();
        rsx_draw_engine_set_backend(NULL);
        return -1;
    }
    s_zero_cb = make_buffer(D3D12_HEAP_TYPE_UPLOAD, 16384, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (s_zero_cb) { void* m = NULL; D3D12_RANGE nr = {0, 0}; if (SUCCEEDED(CALL(s_zero_cb, Map, 0, &nr, &m))) { memset(m, 0, 16384); CALL(s_zero_cb, Unmap, 0, NULL); } }
    s_ready = 1;
    if (rsx_draw_engine_init(s_width, s_height) != 0) {
        fprintf(stderr, "[RSX d3d12] draw engine init failed\n");
        eng_release_device();
        rsx_draw_engine_set_backend(NULL);
        return -1;
    }
    s_active = 1;
    fprintf(stderr, "[RSX d3d12] %s (%ux%u), register-file draw engine, vsync %d, internal resolution x%.2f (%ux%u)\n",
            s_headless ? "headless" : s_display == DISP_FULLSCREEN ? "full screen" : s_display == DISP_BORDERLESS ? "borderless" : "windowed",
            s_win_w, s_win_h, s_vsync, s_scale, sc_dim(s_width), sc_dim(s_height));
    return 0;
}

void rsx_d3d12_engine_shutdown(void)
{
    if (!s_active) return;
    rsx_draw_engine_shutdown();       /* flushes through submit_and_wait */
    rsx_draw_engine_set_backend(NULL);
    eng_shutdown(NULL);
    eng_release_device();
    s_active = 0;
}

int rsx_d3d12_engine_active(void) { return s_active; }

int rsx_d3d12_engine_pump_messages(void)
{
    if (s_headless) return 0;
    if (g_rsx_display_reload && s_ready && s_swap) eng_display_reload();
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return -1;
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return s_window_closed ? -1 : 0;
}

u32 rsx_d3d12_engine_debug_color(void) { return s_clear_argb; }
u32 rsx_d3d12_engine_readback_center(void) { return rsx_draw_engine_readback_center(); }

#endif /* _WIN32 */
