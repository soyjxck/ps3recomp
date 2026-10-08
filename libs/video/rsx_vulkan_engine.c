/*
 * ps3recomp - the register-file draw engine's Vulkan backend (Windows for now)
 *
 * rsx_draw_engine.c decodes the NV4097 register file and drives a backend
 * through rsx_draw_backend. This is that interface over Vulkan 1.3, written
 * against rsx_d3d12_engine.c: the same record stream, object table, staging
 * arena, submit bookkeeping and per-target internal-resolution rule, so the
 * two read side by side. What differs is only how the records are encoded.
 *
 * Shape:
 *   - bind_* calls accumulate into s_pending; draw/clear/snapshot/upload
 *     calls append an EngRecord. Nothing touches a command buffer until a
 *     submit.
 *   - Vertices, indices, constants and texture rows go into one host-visible
 *     arena per submit (the "stage").
 *   - A submit records one command buffer on one of ENG_FRAMES slots. Draws
 *     into the same targets share a dynamic-rendering instance; anything that
 *     needs a layout change or a transfer ends it first. Barriers are
 *     conservative (all commands to all commands): correct first.
 *   - Completion is a timeline semaphore (the D3D12 fence's twin).
 *   - Shaders: the decompilers' HLSL through glslang (rsx_shader_spirv.cpp),
 *     with the fixed binding plan in rsx_shader_spirv.h; one descriptor set,
 *     its two constant buffers dynamic uniform buffers into the stage.
 *   - vulkan-1.dll is loaded at run time: a machine without a Vulkan driver
 *     still runs D3D12.
 *   - D3D clip space is kept by a negative viewport height (Vulkan 1.1).
 *
 * Switches: RSX_BACKEND=vulkan selects it; RSX_VK_VALIDATION=1 turns the
 * Khronos validation layer on; RSX_SCALE, RSX_SCALE_MIN, RSX_DISPLAY,
 * RSX_WINDOW and RSX_VSYNC as the D3D12 backend (exclusive full screen is
 * borderless here).
 */
#if defined(_WIN32) && defined(PS3RECOMP_HAVE_VULKAN)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>

#include "rsx_draw_engine.h"
#include "rsx_texture_layout.h"
#include "rsx_shader_spirv.h"
#include "rsx_vulkan_engine.h"
#include "rsx_present_passes.h"

#define ENG_MAX_OBJECTS   4096
#define ENG_MAX_PIPES     4096
#define ENG_MAX_RECORDS   8192
#define ENG_MAX_VIEWS     256
#define ENG_MAX_MODULES   4096
#define ENG_MAX_SAMPLERS  256
#define ENG_MAX_BUFS      16384
#define ENG_ALIGN         256u
#define ENG_STAGE_MAX     (384u << 20)
#define ENG_STAGE_START   (16u << 20)
#define ENG_UBO_RANGE     16384u      /* both constant blocks; the stage keeps this much slack */
#define ENG_FRAMES        4
#define ENG_INFLIGHT      2
#define ENG_SETS_PER_SLOT 16384u
#define ENG_QUERIES_PER_SLOT 16384u
#define ENG_VIS_SLOTS     16384u
#define ENG_MAX_REPORTS   4096
#define ENG_MAX_SUBMITS   16
#define ENG_STAGE_POOL    8
#define ENG_DEPTH_FMT     VK_FORMAT_D32_SFLOAT_S8_UINT

/* ---- the loader ------------------------------------------------------------ */

#define VK_GLOBAL_FNS(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceExtensionProperties) X(vkEnumerateInstanceLayerProperties) \
    X(vkEnumerateInstanceVersion)
#define VK_INSTANCE_FNS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceFormatProperties) X(vkCreateDevice) \
    X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateWin32SurfaceKHR) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR)
#define VK_DEVICE_FNS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkQueueSubmit) X(vkQueueWaitIdle) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) \
    X(vkQueuePresentKHR) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCreateSemaphore) \
    X(vkDestroySemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) X(vkAllocateMemory) \
    X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkCreateImage) X(vkDestroyImage) \
    X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateSampler) X(vkDestroySampler) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateQueryPool) X(vkDestroyQueryPool) \
    X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImage) X(vkCmdCopyImageToBuffer) X(vkCmdBlitImage) X(vkCmdClearColorImage) \
    X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments) X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdDraw) \
    X(vkCmdDrawIndexed) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetStencilReference) \
    X(vkCmdSetPrimitiveTopology) X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdResetQueryPool) \
    X(vkCmdCopyQueryPoolResults)

static HMODULE s_vk_dll;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#define VK_DECL(n) static PFN_##n n;
VK_GLOBAL_FNS(VK_DECL) VK_INSTANCE_FNS(VK_DECL) VK_DEVICE_FNS(VK_DECL)
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;

/* ---- internal resolution (as rsx_d3d12_engine.c) ---------------------------- */

static float s_scale = 1.0f;
extern volatile int g_rsx_display_reload;   /* rsx_draw_engine.c */
/* Colour and depth targets made without a seed, zeroed before anything else
 * in the next list (eng_encode_records). A D3D12 committed resource arrives
 * zeroed and the title relies on it: its 322x182 bloom targets have a
 * one-pixel border it never draws, which the blur passes then sample. Here
 * the memory is a recycled sub-allocation; its leftovers (NaNs, 59328.0)
 * spread through the blur into coloured blocks at the bottom-left of the
 * frame and a wrong top row -- the 'shadows' capture's 12 differing frames. */
static u32 s_clear_new[ENG_MAX_OBJECTS]; static u32 s_clear_new_n;
static void eng_clear_new(u32 handle) { if (handle && s_clear_new_n < ENG_MAX_OBJECTS) s_clear_new[s_clear_new_n++] = handle; }
static u32 s_scale_min = 64;
static u32 sc_dim(u32 v) { u32 r = (u32)((float)v * s_scale + 0.5f); return v && !r ? 1u : r; }
static u32 sc_pos(u32 v) { return (u32)((float)v * s_scale + 0.5f); }
static void sc_rect(u32* x, u32* y, u32* w, u32* h)
{
    const u32 x0 = sc_pos(*x), y0 = sc_pos(*y), x1 = sc_pos(*x + *w), y1 = sc_pos(*y + *h);
    *x = x0; *y = y0; *w = x1 - x0; *h = y1 - y0;
}
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
        else fprintf(stderr, "[RSX vulkan] RSX_SCALE=%s ignored (0.5 to 8)\n", e);
    }
}

/* ---- types ------------------------------------------------------------------ */

typedef enum { OBJ_NONE = 0, OBJ_TEXTURE, OBJ_COLOR, OBJ_DEPTH, OBJ_SNAPSHOT, OBJ_VIEW } EngObjKind;

/* A sub-allocation of a memory block. */
typedef struct { int block; VkDeviceSize off, size; } MemAlloc;

typedef struct {
    VkImage img;
    MemAlloc mem;
    VkImageView view;          /* sampled view: the crossbar, cube or 2D; depth aspect for a depth target */
    VkImageView att;           /* attachment view (identity; depth+stencil for a depth target) */
    EngObjKind kind;
    u32 alias;                 /* OBJ_VIEW: the surface it views */
    VkFormat fmt;
    u32 w, h, mips, faces;     /* host pixels */
    u32 gw, gh;                /* guest pixels */
    VkImageLayout layout;      /* the owner's */
    int retired;
    /* MSAA: the multisampled twin a 3D pass draws into (eng_ms_*), resolved
     * into this image at the end of every such pass. ms_stale: this image
     * was written another way, so the twin is refreshed before its next pass. */
    VkImage ms_img;
    MemAlloc ms_mem;
    VkImageView ms_att;
    VkImageLayout ms_layout;
    int ms_stale;
} EngObj;

typedef struct {
    VkShaderModule vs, fs, fs1;           /* fs1: WPOS undivided (see eng_pso_for) */
    char* ps_plain;
    VkPipeline pso[3], pso1[3];
    VkPipeline psoms[3]; int failedms[3];   /* the MSAA build (s_ms_n samples) */
    int failed[3], failed1[3];
    rsx_be_render_state rs;
    u32 nattr, stride, rt_count;
    VkFormat rt_fmt;
    u32 cube_mask;                         /* fragment units the program declares TextureCube */
    int no_depth;                          /* drawn without a depth attachment (the helpers) */
    int live;
} EngPipeline;

typedef struct { u64 hash; VkShaderModule mod; } EngModule;
typedef struct { u64 key; VkSampler smp; } EngSampler;
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
    u32   up_off, up_layer, up_mip, up_w, up_h, up_row_texels, up_rows_texels;
} EngRecord;

typedef struct {
    VkBuffer buf; MemAlloc mem; u8* mapped;
    u32 cap;
    u64 fence;
    int in_use;
} EngStage;

typedef struct { u32 slot, index; } EngVisReport;
typedef struct {
    u64 fence;
    int stage;
    u32 q_slot, q_count;
    u32* q_vis;
    EngVisReport* reports; u32 n_reports;
    int live;
} EngSubmit;

/* ---- state ------------------------------------------------------------------ */

typedef enum { DISP_WINDOWED = 0, DISP_BORDERLESS, DISP_FULLSCREEN } EngDisplayMode;
static EngDisplayMode s_display = DISP_WINDOWED;
static int s_active, s_headless, s_ready, s_vsync = 1;
static HWND s_hwnd; static int s_window_closed;
static u32 s_width, s_height, s_win_w, s_win_h;

static VkInstance s_inst;
static VkDebugUtilsMessengerEXT s_dbg;
static VkPhysicalDevice s_phys;
static VkPhysicalDeviceProperties s_props;
static VkPhysicalDeviceMemoryProperties s_memprops;
static VkDevice s_dev;
static VkQueue s_queue; static u32 s_qfam;
static VkSurfaceKHR s_surface;
static VkSwapchainKHR s_swap;
static VkImage s_swap_img[8]; static u32 s_swap_n;
static VkFormat s_swap_fmt; static VkExtent2D s_swap_ext;
static VkImageLayout s_swap_layout[8];
static VkSemaphore s_sem_acq[ENG_FRAMES], s_sem_done[8];
static VkSemaphore s_timeline; static u64 s_fence_value;
static VkCommandPool s_pool[ENG_FRAMES]; static VkCommandBuffer s_cmdbuf[ENG_FRAMES]; static u64 s_slot_fence[ENG_FRAMES];
static VkCommandPool s_aux_pool; static VkCommandBuffer s_aux_cmd;
static VkDescriptorPool s_dpool[ENG_FRAMES];
static VkQueryPool s_qpool[ENG_FRAMES]; static VkBuffer s_qbuf[ENG_FRAMES]; static MemAlloc s_qmem[ENG_FRAMES]; static u64* s_qmap[ENG_FRAMES];
static VkDescriptorSetLayout s_dsl; static VkPipelineLayout s_pl;
static VkBuffer s_zero_buf; static MemAlloc s_zero_mem;
static VkPipeline s_depth_pso, s_depth_pack_pso;
static VkShaderModule s_helper_vs, s_helper_ps_depth, s_helper_ps_pack;
static SRWLOCK s_pipe_lock = SRWLOCK_INIT;
static SRWLOCK s_mem_lock = SRWLOCK_INIT;
static int s_has_maint5, s_query_precise;

static VkCommandBuffer s_cmd; static u32 s_slot;
static EngObj s_obj[ENG_MAX_OBJECTS];
static u32 s_obj_count, s_obj_free[ENG_MAX_OBJECTS], s_obj_free_count;
static u32 s_null_tex, s_null_cube;
static EngPipeline s_pipe[ENG_MAX_PIPES]; static u32 s_pipe_count;
static EngModule s_mod[ENG_MAX_MODULES]; static u32 s_mod_count;
static EngSampler s_samp[ENG_MAX_SAMPLERS]; static u32 s_samp_count;
static VkSampler s_point_sampler, s_linear_sampler;
static u32 s_ms_n = 1;                      /* MSAA samples in use */
static VkShaderModule s_helper_ps_copy;     /* a target into its MSAA twin */
static VkPipeline s_fxaa_pso, s_area_pso;   /* rsx_present_passes.h */
static u32 s_aa_obj, s_area_obj;            /* their targets: frame-sized, window-sized */
int rsx_aa_mode(void);                      /* rsx_draw_engine.c */
static EngView s_view[ENG_MAX_VIEWS]; static u32 s_view_count;
static EngRecord s_rec[ENG_MAX_RECORDS]; static u32 s_rec_count, s_dropped;
static EngRecord s_pending;
static VkBuffer s_buf[ENG_MAX_BUFS]; static MemAlloc s_buf_mem[ENG_MAX_BUFS]; static u32 s_buf_bytes[ENG_MAX_BUFS], s_buf_cap[ENG_MAX_BUFS];
static u32 s_buf_count, s_buf_free[ENG_MAX_BUFS], s_buf_free_count; static u8 s_buf_retired[ENG_MAX_BUFS];
static u64 s_buf_retired_fence[ENG_MAX_BUFS];
static EngStage s_stage[ENG_STAGE_POOL]; static int s_stage_cur = -1;
static u32 s_stage_used, s_stage_want = ENG_STAGE_START;
static EngSubmit s_sub[ENG_MAX_SUBMITS];
static u32 s_submit_seq;
static u32 s_vs_cb_seq = ~0u, s_vs_cb_off, s_vs_cb_bytes;
static u64 s_vis_count[ENG_VIS_SLOTS]; static u32 s_vis_next, s_vis_cur;
static EngVisReport s_vis_pending[ENG_MAX_REPORTS]; static u32 s_vis_npending;
static u32 s_q_vis[ENG_QUERIES_PER_SLOT]; static u32 s_q_used;
#define ENG_Q_SCALED 0x80000000u
#define ENG_Q_MS_SHIFT 28          /* log2 of an MSAA pass's samples: its count is of samples */
#define ENG_Q_INDEX 0x0FFFFFFFu
static u32 s_fallback_depth[8]; static u32 s_fallback_n;
static u32 s_retired[ENG_MAX_OBJECTS]; static u32 s_retired_count; static u64 s_retired_fence[ENG_MAX_OBJECTS];

static u64 eng_q_count(u32 qv, u64 c)
{
    if ((qv & ENG_Q_SCALED) && s_scale != 1.0f) c = (u64)((double)c / ((double)s_scale * (double)s_scale) + 0.5);
    { const u32 ms = (qv >> ENG_Q_MS_SHIFT) & 3u; if (ms) c = (c + (1u << ms) / 2) >> ms; }
    return c;
}

static u64 fnv1a64(const void* data, size_t n, u64 h)
{
    const u8* p = (const u8*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

#define VKCHECK(call, what) do { VkResult _r = (call); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[rsx engine/vulkan] %s failed: %d\n", what, (int)_r); return -1; } } while (0)

/* ---- formats and state tables ------------------------------------------------ */

static VkFormat eng_vkfmt(rsx_be_format f)
{
    switch (f) {
    case RSX_BE_FMT_R8:              return VK_FORMAT_R8_UNORM;
    case RSX_BE_FMT_R8G8:            return VK_FORMAT_R8G8_UNORM;
    case RSX_BE_FMT_R8G8B8A8:        return VK_FORMAT_R8G8B8A8_UNORM;
    case RSX_BE_FMT_BC1:             return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case RSX_BE_FMT_BC2:             return VK_FORMAT_BC2_UNORM_BLOCK;
    case RSX_BE_FMT_BC3:             return VK_FORMAT_BC3_UNORM_BLOCK;
    case RSX_BE_FMT_R16:             return VK_FORMAT_R16_UNORM;
    case RSX_BE_FMT_R16G16:          return VK_FORMAT_R16G16_UNORM;
    case RSX_BE_FMT_R16G16F:         return VK_FORMAT_R16G16_SFLOAT;
    case RSX_BE_FMT_R16G16B16A16F:   return VK_FORMAT_R16G16B16A16_SFLOAT;
    case RSX_BE_FMT_R32F:            return VK_FORMAT_R32_SFLOAT;
    case RSX_BE_FMT_R32G32B32A32F:   return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:                         return VK_FORMAT_R8_UNORM;
    }
}
static int fmt_is_bc(VkFormat f) { return f == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || f == VK_FORMAT_BC2_UNORM_BLOCK || f == VK_FORMAT_BC3_UNORM_BLOCK; }
static u32 eng_bpp(VkFormat f)   /* bytes per texel, or per 4x4 block for BC */
{
    switch (f) {
    case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_UNORM: return 2;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return 8;
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC3_UNORM_BLOCK: return 16;
    default: return 4;
    }
}

static VkComponentSwizzle swz(u8 s)
{
    switch (s) {
    case 0: return VK_COMPONENT_SWIZZLE_R;
    case 1: return VK_COMPONENT_SWIZZLE_G;
    case 2: return VK_COMPONENT_SWIZZLE_B;
    case 3: return VK_COMPONENT_SWIZZLE_A;
    case 4: return VK_COMPONENT_SWIZZLE_ZERO;
    default: return VK_COMPONENT_SWIZZLE_ONE;
    }
}
/* The crossbar's selectors arrive in A,R,G,B order (as eng_mapping in the
 * D3D12 backend). */
static VkComponentMapping eng_mapping(u32 remap, u32 rsx_fmt)
{
    u8 sel[4];
    rsx_texture_component_remap(remap, rsx_fmt & 0x9Fu, sel);
    VkComponentMapping m = { swz(sel[1]), swz(sel[2]), swz(sel[3]), swz(sel[0]) };
    return m;
}
static const VkComponentMapping k_identity = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };

static VkCompareOp gcm_cmp(u32 f)
{
    switch (f) {
    case 0x0200: return VK_COMPARE_OP_NEVER;
    case 0x0201: return VK_COMPARE_OP_LESS;
    case 0x0202: return VK_COMPARE_OP_EQUAL;
    case 0x0203: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 0x0204: return VK_COMPARE_OP_GREATER;
    case 0x0205: return VK_COMPARE_OP_NOT_EQUAL;
    case 0x0206: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default:     return VK_COMPARE_OP_ALWAYS;
    }
}
static VkBlendFactor gcm_blend_factor(u32 f, int alpha)
{
    switch (f & 0xFFFFu) {
    case 0x0000: return VK_BLEND_FACTOR_ZERO;
    case 0x0001: return VK_BLEND_FACTOR_ONE;
    case 0x0300: return alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_SRC_COLOR;
    case 0x0301: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x0302: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x0303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x0304: return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x0305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x0306: return alpha ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_DST_COLOR;
    case 0x0307: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x0308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x8001: case 0x8003: return VK_BLEND_FACTOR_CONSTANT_COLOR;   /* as the D3D12 backend */
    case 0x8002: case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    default:     return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp gcm_blend_op(u32 e)
{
    switch (e & 0xFFFFu) {
    case 0x8007: return VK_BLEND_OP_MIN;
    case 0x8008: return VK_BLEND_OP_MAX;
    case 0x800A: return VK_BLEND_OP_SUBTRACT;
    case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
    default:     return VK_BLEND_OP_ADD;
    }
}
static VkStencilOp gcm_stencil_op(u32 op)
{
    switch (op) {
    case 0x0000: return VK_STENCIL_OP_ZERO;
    case 0x1E01: return VK_STENCIL_OP_REPLACE;
    case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 0x150A: return VK_STENCIL_OP_INVERT;
    case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default:     return VK_STENCIL_OP_KEEP;
    }
}
static VkSamplerAddressMode gcm_wrap(u32 w)
{
    switch (w & 0xFu) {
    case 1:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case 4:  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 6: case 7: case 8: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}
static VkPrimitiveTopology topo_vk(rsx_topology t)
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS:         return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case RSX_TOPOLOGY_LINES:          return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case RSX_TOPOLOGY_LINE_STRIP:     return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case RSX_TOPOLOGY_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    default:                          return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}
static int topo_class(rsx_topology t)
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS: return 0;
    case RSX_TOPOLOGY_LINES: case RSX_TOPOLOGY_LINE_STRIP: return 1;
    default: return 2;
    }
}

/* ---- memory ------------------------------------------------------------------ *
 * Blocks of device memory, sub-allocated first fit with a sorted free list.
 * Images and buffers never share a block (bufferImageGranularity); a request
 * over half a block gets one of its own. Host-visible blocks stay mapped. */

#define MEM_BLOCKS 512
typedef struct { VkDeviceSize off, size; } MemRange;
typedef struct {
    VkDeviceMemory mem; u32 type; int image; VkDeviceSize size; u8* mapped;
    MemRange* fr; u32 nfr, frcap;
    int dedicated;
} MemBlock;
static MemBlock s_blk[MEM_BLOCKS]; static u32 s_nblk;

static int mem_type(u32 bits, VkMemoryPropertyFlags want)
{
    for (u32 i = 0; i < s_memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (s_memprops.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}

static int blk_take(MemBlock* b, VkDeviceSize size, VkDeviceSize align, VkDeviceSize* out)
{
    for (u32 i = 0; i < b->nfr; i++) {
        const VkDeviceSize start = (b->fr[i].off + align - 1) & ~(align - 1);
        const VkDeviceSize end = b->fr[i].off + b->fr[i].size;
        if (start + size > end) continue;
        const VkDeviceSize lead = start - b->fr[i].off, tail = end - (start + size);
        if (lead && tail) {   /* split in two */
            if (b->nfr == b->frcap) {
                b->frcap = b->frcap ? b->frcap * 2 : 16;
                b->fr = (MemRange*)realloc(b->fr, b->frcap * sizeof(MemRange));
            }
            memmove(&b->fr[i + 2], &b->fr[i + 1], (b->nfr - i - 1) * sizeof(MemRange));
            b->fr[i].size = lead;
            b->fr[i + 1].off = start + size; b->fr[i + 1].size = tail;
            b->nfr++;
        } else if (lead) { b->fr[i].size = lead; }
        else if (tail) { b->fr[i].off = start + size; b->fr[i].size = tail; }
        else { memmove(&b->fr[i], &b->fr[i + 1], (b->nfr - i - 1) * sizeof(MemRange)); b->nfr--; }
        *out = start;
        return 1;
    }
    return 0;
}

static void blk_give(MemBlock* b, VkDeviceSize off, VkDeviceSize size)
{
    u32 i = 0;
    while (i < b->nfr && b->fr[i].off < off) i++;
    if (b->nfr == b->frcap) {
        b->frcap = b->frcap ? b->frcap * 2 : 16;
        b->fr = (MemRange*)realloc(b->fr, b->frcap * sizeof(MemRange));
    }
    memmove(&b->fr[i + 1], &b->fr[i], (b->nfr - i) * sizeof(MemRange));
    b->fr[i].off = off; b->fr[i].size = size; b->nfr++;
    if (i + 1 < b->nfr && b->fr[i].off + b->fr[i].size == b->fr[i + 1].off) {
        b->fr[i].size += b->fr[i + 1].size;
        memmove(&b->fr[i + 1], &b->fr[i + 2], (b->nfr - i - 2) * sizeof(MemRange)); b->nfr--;
    }
    if (i > 0 && b->fr[i - 1].off + b->fr[i - 1].size == b->fr[i].off) {
        b->fr[i - 1].size += b->fr[i].size;
        memmove(&b->fr[i], &b->fr[i + 1], (b->nfr - i - 1) * sizeof(MemRange)); b->nfr--;
    }
}

static int mem_alloc(const VkMemoryRequirements* req, VkMemoryPropertyFlags want, int image, MemAlloc* out)
{
    const int type = mem_type(req->memoryTypeBits, want);
    if (type < 0) return 0;
    const int host = (want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const VkDeviceSize bsize = host ? ((VkDeviceSize)64 << 20) : ((VkDeviceSize)256 << 20);
    AcquireSRWLockExclusive(&s_mem_lock);
    int ok = 0;
    if (req->size <= bsize / 2) {
        for (u32 b = 0; b < s_nblk && !ok; b++) {
            MemBlock* m = &s_blk[b];
            if (!m->mem || m->dedicated || m->type != (u32)type || m->image != image) continue;
            VkDeviceSize off;
            if (blk_take(m, req->size, req->alignment ? req->alignment : 1, &off)) {
                out->block = (int)b; out->off = off; out->size = req->size; ok = 1;
            }
        }
    }
    if (!ok) {
        u32 b;
        for (b = 0; b < s_nblk; b++) if (!s_blk[b].mem) break;
        if (b == s_nblk) { if (s_nblk >= MEM_BLOCKS) { ReleaseSRWLockExclusive(&s_mem_lock); return 0; } s_nblk++; }
        MemBlock* m = &s_blk[b];
        memset(m, 0, sizeof *m);
        const int dedicated = req->size > bsize / 2;
        VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = dedicated ? req->size : bsize;
        ai.memoryTypeIndex = (u32)type;
        if (vkAllocateMemory(s_dev, &ai, NULL, &m->mem) != VK_SUCCESS) {
            m->mem = VK_NULL_HANDLE;
            ReleaseSRWLockExclusive(&s_mem_lock);
            static int n = 0;
            if (n++ < 8) fprintf(stderr, "[rsx engine/vulkan] out of device memory (%llu bytes)\n", (unsigned long long)ai.allocationSize);
            return 0;
        }
        m->type = (u32)type; m->image = image; m->size = ai.allocationSize; m->dedicated = dedicated;
        if (host) vkMapMemory(s_dev, m->mem, 0, VK_WHOLE_SIZE, 0, (void**)&m->mapped);
        if (dedicated) { out->block = (int)b; out->off = 0; out->size = req->size; ok = 1; }
        else {
            blk_give(m, 0, bsize);
            VkDeviceSize off;
            ok = blk_take(m, req->size, req->alignment ? req->alignment : 1, &off);
            out->block = (int)b; out->off = off; out->size = req->size;
        }
    }
    ReleaseSRWLockExclusive(&s_mem_lock);
    return ok;
}

static void mem_free(MemAlloc* a)
{
    if (!a->size) return;
    AcquireSRWLockExclusive(&s_mem_lock);
    MemBlock* m = &s_blk[a->block];
    if (m->dedicated) {
        if (m->mapped) vkUnmapMemory(s_dev, m->mem);
        vkFreeMemory(s_dev, m->mem, NULL);
        free(m->fr);
        memset(m, 0, sizeof *m);
    } else {
        blk_give(m, a->off, a->size);
    }
    ReleaseSRWLockExclusive(&s_mem_lock);
    memset(a, 0, sizeof *a);
}

static u8* mem_ptr(const MemAlloc* a) { return s_blk[a->block].mapped ? s_blk[a->block].mapped + a->off : NULL; }

static VkBuffer make_buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, MemAlloc* mem)
{
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = bytes; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer b = VK_NULL_HANDLE;
    if (vkCreateBuffer(s_dev, &bi, NULL, &b) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(s_dev, b, &req);
    if (!mem_alloc(&req, want, 0, mem)) { vkDestroyBuffer(s_dev, b, NULL); return VK_NULL_HANDLE; }
    vkBindBufferMemory(s_dev, b, s_blk[mem->block].mem, mem->off);
    return b;
}
#define HOST_MEM (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

/* ---- timeline ---------------------------------------------------------------- */

extern double g_rsx_frame_gpu_wait_ms;   /* rsx_draw_engine.c */
static int fence_done(u64 v)
{
    if (!v) return 1;
    u64 cur = 0;
    vkGetSemaphoreCounterValue(s_dev, s_timeline, &cur);
    return cur >= v;
}
static void fence_wait(u64 v)
{
    if (fence_done(v)) return;
    LARGE_INTEGER qf, q0, q1; QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&q0);
    VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    wi.semaphoreCount = 1; wi.pSemaphores = &s_timeline; wi.pValues = &v;
    for (int tries = 0; tries < 5; tries++) {
        const VkResult r = vkWaitSemaphores(s_dev, &wi, 2000000000ull);
        if (r == VK_SUCCESS) {
            QueryPerformanceCounter(&q1);
            g_rsx_frame_gpu_wait_ms += (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)qf.QuadPart;
            return;
        }
        fprintf(stderr, "[rsx engine/vulkan] submit %llu stuck %ds (result %d)\n", (unsigned long long)v, 2 * (tries + 1), (int)r);
        if (r == VK_ERROR_DEVICE_LOST) { s_ready = 0; return; }
    }
}

/* ---- object table ------------------------------------------------------------ */

static EngObj* eng_obj(u32 handle)
{
    if (!handle || handle > s_obj_count) return NULL;
    EngObj* o = &s_obj[handle - 1];
    return o->kind ? o : NULL;
}
static EngObj* eng_owner(u32 handle)
{
    EngObj* o = eng_obj(handle);
    if (o && o->kind == OBJ_VIEW) o = eng_obj(o->alias);
    return o && o->img ? o : NULL;
}
static int eng_obj_scaled(u32 handle)
{
    const EngObj* o = eng_owner(handle);
    return o && o->gw && o->gh && (o->w != o->gw || o->h != o->gh);
}
static VkImageAspectFlags obj_aspect(const EngObj* o)
{
    return o->kind == OBJ_DEPTH ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
}

static u32 eng_obj_add(EngObjKind kind)
{
    u32 slot;
    if (s_obj_free_count) slot = s_obj_free[--s_obj_free_count];
    else {
        if (s_obj_count >= ENG_MAX_OBJECTS) {
            static unsigned long n = 0;
            if (n++ % 10000 == 0) fprintf(stderr, "[rsx engine/vulkan] object table full (%u)\n", s_obj_count);
            return 0;
        }
        slot = s_obj_count++;
    }
    memset(&s_obj[slot], 0, sizeof s_obj[slot]);
    s_obj[slot].kind = kind;
    return slot + 1;
}

static VkImageView make_view(VkImage img, VkFormat fmt, VkImageViewType type, VkImageAspectFlags aspect,
                             u32 mips, u32 layers, VkComponentMapping map)
{
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = img; vi.viewType = type; vi.format = fmt; vi.components = map;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = mips; vi.subresourceRange.layerCount = layers;
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(s_dev, &vi, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    return v;
}

/* An image of the given use, bound to device memory, in its object. */
static u32 eng_image(EngObjKind kind, VkFormat fmt, u32 w, u32 h, u32 mips, u32 faces, VkImageUsageFlags usage)
{
    if (!s_dev || !w || !h) return 0;
    if (!mips) mips = 1;
    if (!faces) faces = 1;
    VkImageCreateInfo ii = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt;
    ii.extent.width = w; ii.extent.height = h; ii.extent.depth = 1;
    ii.mipLevels = mips; ii.arrayLayers = faces; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (faces == 6) ii.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    VkImage img = VK_NULL_HANDLE;
    if (vkCreateImage(s_dev, &ii, NULL, &img) != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 16) fprintf(stderr, "[rsx engine/vulkan] image %ux%u fmt %d failed\n", w, h, (int)fmt);
        return 0;
    }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(s_dev, img, &req);
    MemAlloc mem;
    if (!mem_alloc(&req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1, &mem)) { vkDestroyImage(s_dev, img, NULL); return 0; }
    vkBindImageMemory(s_dev, img, s_blk[mem.block].mem, mem.off);
    const u32 handle = eng_obj_add(kind);
    if (!handle) { vkDestroyImage(s_dev, img, NULL); mem_free(&mem); return 0; }
    EngObj* o = &s_obj[handle - 1];
    o->img = img; o->mem = mem; o->fmt = fmt; o->w = w; o->h = h; o->gw = w; o->gh = h;
    o->mips = mips; o->faces = faces; o->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return handle;
}

static void eng_obj_destroy(EngObj* o)
{
    if (o->ms_att) vkDestroyImageView(s_dev, o->ms_att, NULL);
    if (o->ms_img) { vkDestroyImage(s_dev, o->ms_img, NULL); mem_free(&o->ms_mem); }
    if (o->view) vkDestroyImageView(s_dev, o->view, NULL);
    if (o->att) vkDestroyImageView(s_dev, o->att, NULL);
    if (o->img) vkDestroyImage(s_dev, o->img, NULL);
    mem_free(&o->mem);
    memset(o, 0, sizeof *o);
}

static void eng_collect_retired(void)
{
    u32 k = 0;
    for (u32 i = 0; i < s_retired_count; i++) {
        const u32 h = s_retired[i];
        if (s_rec_count || !fence_done(s_retired_fence[i])) { s_retired[k] = h; s_retired_fence[k] = s_retired_fence[i]; k++; continue; }
        eng_obj_destroy(&s_obj[h - 1]);
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
    for (u32 i = 0; i < s_view_count; i++)
        if (s_view[i].surface == handle) {
            eng_obj_release(NULL, s_view[i].view);
            s_view[i] = s_view[--s_view_count];
            i--;
        }
    if (s_retired_count < ENG_MAX_OBJECTS) {
        s_retired[s_retired_count] = handle;
        s_retired_fence[s_retired_count] = s_fence_value;
        s_retired_count++;
    }
}

/* ---- staging ------------------------------------------------------------------- */

static void eng_submit(u32 present_surface, int wait);

static int eng_stage_make(EngStage* s, u32 cap)
{
    s->buf = make_buffer((VkDeviceSize)cap + ENG_UBO_RANGE,
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         HOST_MEM, &s->mem);
    if (!s->buf) return 0;
    s->mapped = mem_ptr(&s->mem);
    s->cap = cap;
    return s->mapped != NULL;
}
static void eng_stage_free(EngStage* s)
{
    if (s->buf) vkDestroyBuffer(s_dev, s->buf, NULL);
    mem_free(&s->mem);
    s->buf = VK_NULL_HANDLE; s->mapped = NULL; s->cap = 0;
}

static int eng_stage_pick(void)
{
    int best = -1;
    for (int i = 0; i < ENG_STAGE_POOL; i++) {
        EngStage* s = &s_stage[i];
        if (s->in_use && fence_done(s->fence)) s->in_use = 0;
        if (s->in_use) continue;
        if (s->buf && s->cap >= s_stage_want && (best < 0 || s->cap < s_stage[best].cap)) best = i;
    }
    if (best >= 0) return best;
    for (int i = 0; i < ENG_STAGE_POOL; i++) if (!s_stage[i].in_use && !s_stage[i].buf) { best = i; break; }
    if (best < 0) {
        for (int i = 0; i < ENG_STAGE_POOL; i++)
            if (!s_stage[i].in_use && (best < 0 || s_stage[i].cap < s_stage[best].cap)) best = i;
        if (best < 0) return -1;
        eng_stage_free(&s_stage[best]);
    }
    return eng_stage_make(&s_stage[best], s_stage_want) ? best : -1;
}

static int eng_stage_reserve(u32 bytes, u32* out_off)
{
    const u32 start = (s_stage_used + ENG_ALIGN - 1u) & ~(ENG_ALIGN - 1u);
    if ((u64)start + bytes > ENG_STAGE_MAX) { eng_submit(0, 1); return eng_stage_reserve(bytes, out_off); }
    if (s_stage_cur < 0) {
        s_stage_cur = eng_stage_pick();
        if (s_stage_cur < 0) return 0;
        s_stage_used = 0;
    }
    EngStage* s = &s_stage[s_stage_cur];
    if (start + bytes > s->cap) {
        u32 cap = s->cap ? s->cap : ENG_STAGE_START;
        while (start + bytes > cap) cap *= 2u;
        EngStage n; memset(&n, 0, sizeof n);
        if (!eng_stage_make(&n, cap)) return 0;
        if (s_stage_used) memcpy(n.mapped, s->mapped, s_stage_used);
        eng_stage_free(s);
        s->buf = n.buf; s->mem = n.mem; s->mapped = n.mapped; s->cap = n.cap;
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

/* ---- buffers the engine keeps (vertex cache) -------------------------------------- */

static u32 buf_class(u32 bytes) { u32 c = 65536u; while (c < bytes) c <<= 1; return c; }
#define ENG_BUF_POOL 256
static struct { VkBuffer buf; MemAlloc mem; u32 cap; } s_buf_pool[ENG_BUF_POOL];
static u32 s_buf_pool_n;

static u32 eng_buffer_wrap(void* user, void* data, u32 bytes)
{
    (void)user;
    if (!s_dev || !data || !bytes) return 0;
    u32 slot;
    if (s_buf_free_count) slot = s_buf_free[s_buf_free_count - 1];
    else if (s_buf_count < ENG_MAX_BUFS) slot = s_buf_count;
    else return 0;
    const u32 cap = buf_class(bytes);
    VkBuffer b = VK_NULL_HANDLE; MemAlloc mem; memset(&mem, 0, sizeof mem);
    for (u32 i = 0; i < s_buf_pool_n; i++) {
        if (s_buf_pool[i].cap != cap) continue;
        b = s_buf_pool[i].buf; mem = s_buf_pool[i].mem;
        s_buf_pool[i] = s_buf_pool[--s_buf_pool_n];
        break;
    }
    if (!b) b = make_buffer(cap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, HOST_MEM, &mem);
    if (!b) return 0;
    memcpy(mem_ptr(&mem), data, bytes);
    _aligned_free(data);
    if (s_buf_free_count) s_buf_free_count--; else s_buf_count++;
    s_buf[slot] = b; s_buf_mem[slot] = mem; s_buf_bytes[slot] = bytes; s_buf_cap[slot] = cap; s_buf_retired[slot] = 0;
    return slot + 1;
}
static void eng_buffer_release(void* user, u32 h)
{
    (void)user;
    if (!h || h > s_buf_count || !s_buf[h - 1] || s_buf_retired[h - 1]) return;
    s_buf_retired[h - 1] = 1;
    s_buf_retired_fence[h - 1] = s_fence_value;
}
static void eng_collect_retired_buffers(void)
{
    if (s_rec_count) return;
    for (u32 i = 0; i < s_buf_count; i++) {
        if (!s_buf_retired[i] || !fence_done(s_buf_retired_fence[i])) continue;
        if (s_buf_pool_n < ENG_BUF_POOL) {
            s_buf_pool[s_buf_pool_n].buf = s_buf[i]; s_buf_pool[s_buf_pool_n].mem = s_buf_mem[i];
            s_buf_pool[s_buf_pool_n].cap = s_buf_cap[i]; s_buf_pool_n++;
        } else {
            vkDestroyBuffer(s_dev, s_buf[i], NULL); mem_free(&s_buf_mem[i]);
        }
        s_buf[i] = VK_NULL_HANDLE; s_buf_retired[i] = 0;
        memset(&s_buf_mem[i], 0, sizeof s_buf_mem[i]);
        s_buf_free[s_buf_free_count++] = i;
    }
}

/* ---- window ---------------------------------------------------------------------- */

static LRESULT CALLBACK eng_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_CLOSE:   s_window_closed = 1; DestroyWindow(hwnd); return 0;
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
    wc.hIcon = LoadIconA(GetModuleHandle(NULL), MAKEINTRESOURCEA(101));
    wc.hIconSm = wc.hIcon;
    wc.lpszClassName = "ps3recomp_vulkan_engine";
    RegisterClassExA(&wc);
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    if (s_display != DISP_WINDOWED)
        return CreateWindowExA(0, wc.lpszClassName, title ? title : "ps3recomp (Vulkan)", WS_POPUP | WS_VISIBLE,
                               0, 0, (int)width, (int)height, NULL, NULL, GetModuleHandle(NULL), NULL);
    RECT wr = {0, 0, (LONG)width, (LONG)height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExA(0, wc.lpszClassName, title ? title : "ps3recomp (Vulkan)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                           CW_USEDEFAULT, CW_USEDEFAULT, wr.right - wr.left, wr.bottom - wr.top,
                           NULL, NULL, GetModuleHandle(NULL), NULL);
}

/* ---- shaders and pipelines ---------------------------------------------------------- */

/* A shader module for the HLSL text, translated once whatever references it. */
static VkShaderModule eng_module(const char* hlsl, int stage, const char* what)
{
    const u64 hash = fnv1a64(hlsl, strlen(hlsl), 1469598103934665603ull ^ (u64)stage);
    for (u32 i = 0; i < s_mod_count; i++) if (s_mod[i].hash == hash) return s_mod[i].mod;
    u32* words = NULL; char log[2048];
    const u32 n = rsx_hlsl_to_spirv(hlsl, stage, &words, log, sizeof log);
    if (!n) {
        static int k = 0;
        if (k++ < 32) fprintf(stderr, "[rsx engine/vulkan] %s: HLSL -> SPIR-V failed: %.900s\n", what, log);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo mi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    mi.codeSize = (size_t)n * 4u; mi.pCode = words;
    VkShaderModule m = VK_NULL_HANDLE;
    const VkResult r = vkCreateShaderModule(s_dev, &mi, NULL, &m);
    free(words);
    if (r != VK_SUCCESS) return VK_NULL_HANDLE;
    if (s_mod_count < ENG_MAX_MODULES) { s_mod[s_mod_count].hash = hash; s_mod[s_mod_count].mod = m; s_mod_count++; }
    return m;
}

/* Which fragment units a program declares as TextureCube ("TextureCube
 * rsx_tex<u>"), so an empty unit gets a cube-shaped null. */
static u32 eng_cube_mask(const char* ps)
{
    u32 m = 0;
    for (const char* q = ps; (q = strstr(q, "TextureCube")) != NULL; q += 11) {
        const char* t = strstr(q, "rsx_tex");
        if (!t || t - q > 24) continue;
        const int u = atoi(t + 7);
        if (u >= 0 && u < 16) m |= 1u << u;
    }
    return m;
}

/* The sample count eng_build_pso builds with: 1, or s_ms_n for the MSAA
 * builds -- set under s_pipe_lock, which every build holds. */
static VkSampleCountFlagBits s_build_samples = VK_SAMPLE_COUNT_1_BIT;
static VkPipeline eng_build_pso(EngPipeline* p, int cls, VkShaderModule fs)
{
    VkPipelineShaderStageCreateInfo st[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                              { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = p->vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";

    VkVertexInputBindingDescription vb = { 0, p->stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[RSX_DSP_NUM_VERTEX_ATTR];
    for (u32 s = 0; s < p->nattr; s++) {   /* location = layout slot (the Metal path's rule) */
        va[s].location = s; va[s].binding = 0; va[s].format = VK_FORMAT_R32G32B32A32_SFLOAT; va[s].offset = s * 16u;
    }
    VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = p->nattr ? 1 : 0; vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = p->nattr; vi.pVertexAttributeDescriptions = va;

    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = cls == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : cls == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                                                   : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vps.viewportCount = 1; vps.scissorCount = 1;

    const rsx_be_render_state* rs = &p->rs;
    VkPipelineRasterizationStateCreateInfo rz = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rz.polygonMode = VK_POLYGON_MODE_FILL; rz.lineWidth = 1.0f;
    rz.cullMode = VK_CULL_MODE_NONE;
    if (rs->cull_enable && rs->cull_face)
        rz.cullMode = rs->cull_face == 0x0404u ? VK_CULL_MODE_FRONT_BIT
                    : rs->cull_face == 0x0405u ? VK_CULL_MODE_BACK_BIT
                    : rs->cull_face == 0x0408u ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE;
    /* D3D's winding carries over unchanged under the negative viewport. */
    rz.frontFace = rs->front_face == 0x0901u ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = s_build_samples;

    VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = rs->depth_test ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = (rs->depth_test && rs->depth_write) ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = rs->depth_test ? gcm_cmp(rs->depth_func) : VK_COMPARE_OP_ALWAYS;
    if (rs->stencil_enable) {
        ds.stencilTestEnable = VK_TRUE;
        ds.front.compareOp = gcm_cmp(rs->s_func);
        ds.front.failOp = gcm_stencil_op(rs->s_fail);
        ds.front.depthFailOp = gcm_stencil_op(rs->s_zfail);
        ds.front.passOp = gcm_stencil_op(rs->s_zpass);
        ds.front.compareMask = rs->s_func_mask & 0xFFu;
        ds.front.writeMask = rs->s_write_mask & 0xFFu;
        ds.back = ds.front;
        if (rs->stencil_two_sided) {
            ds.back.compareOp = gcm_cmp(rs->bs_func);
            ds.back.failOp = gcm_stencil_op(rs->bs_fail);
            ds.back.depthFailOp = gcm_stencil_op(rs->bs_zfail);
            ds.back.passOp = gcm_stencil_op(rs->bs_zpass);
        }
    }

    VkColorComponentFlags wm = 0;
    if ((rs->color_mask >>  0) & 0xFF) wm |= VK_COLOR_COMPONENT_B_BIT;
    if ((rs->color_mask >>  8) & 0xFF) wm |= VK_COLOR_COMPONENT_G_BIT;
    if ((rs->color_mask >> 16) & 0xFF) wm |= VK_COLOR_COMPONENT_R_BIT;
    if ((rs->color_mask >> 24) & 0xFF) wm |= VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendAttachmentState ba = {0};
    ba.colorWriteMask = wm;
    if (rs->blend_enable) {
        ba.blendEnable = VK_TRUE;
        ba.srcColorBlendFactor = gcm_blend_factor(rs->sf_rgb, 0);
        ba.dstColorBlendFactor = gcm_blend_factor(rs->df_rgb, 0);
        ba.colorBlendOp = gcm_blend_op(rs->eq_rgb);
        ba.srcAlphaBlendFactor = gcm_blend_factor(rs->sf_a, 1);
        ba.dstAlphaBlendFactor = gcm_blend_factor(rs->df_a, 1);
        ba.alphaBlendOp = gcm_blend_op(rs->eq_a);
    }
    VkPipelineColorBlendAttachmentState bas[RSX_BE_MAX_COLOR_TARGETS];
    VkFormat fmts[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 r = 0; r < p->rt_count; r++) { bas[r] = ba; fmts[r] = p->rt_fmt; }
    VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = p->rt_count; cb.pAttachments = bas;

    VkDynamicState dyn[4] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                              VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 4; dy.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo ri = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    ri.colorAttachmentCount = p->rt_count; ri.pColorAttachmentFormats = fmts;
    ri.depthAttachmentFormat = p->no_depth ? VK_FORMAT_UNDEFINED : ENG_DEPTH_FMT;
    ri.stencilAttachmentFormat = ri.depthAttachmentFormat;

    VkGraphicsPipelineCreateInfo gi = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gi.pNext = &ri;
    gi.stageCount = 2; gi.pStages = st;
    gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia; gi.pViewportState = &vps;
    gi.pRasterizationState = &rz; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cb; gi.pDynamicState = &dy; gi.layout = s_pl;
    VkPipeline pso = VK_NULL_HANDLE;
    const VkResult r = vkCreateGraphicsPipelines(s_dev, VK_NULL_HANDLE, 1, &gi, NULL, &pso);
    if (r != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 16) fprintf(stderr, "[rsx engine/vulkan] pipeline failed: %d\n", (int)r);
        return VK_NULL_HANDLE;
    }
    return pso;
}

/* A fragment program at the current internal resolution. */
static VkShaderModule eng_fp_module(const char* ps_hlsl)
{
    /* WPOS (input.position) arrives in host pixels at a raised internal
     * resolution: divide it back, as the D3D12 backend does, and keep the
     * plain text for passes into a target kept at the guest size. */
    VkShaderModule fs = VK_NULL_HANDLE;
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
            fs = eng_module(text, RSX_SPV_STAGE_FRAGMENT, "fp");
            free(text);
        }
    }
    if (!fs) fs = eng_module(ps_hlsl, RSX_SPV_STAGE_FRAGMENT, "fp");
    return fs;
}

static u32 eng_pipeline_create_locked(const char* vs_hlsl, const char* ps_hlsl, const rsx_be_render_state* rs,
                                      const rsx_vertex_layout_plan* layout, u32 vertex_stride,
                                      rsx_be_format rt_fmt, u32 rt_count)
{
    if (!s_dev || !vertex_stride || s_pipe_count >= ENG_MAX_PIPES) return 0;
    if (!rt_count) rt_count = 1;
    if (rt_count > RSX_BE_MAX_COLOR_TARGETS) rt_count = RSX_BE_MAX_COLOR_TARGETS;
    VkShaderModule vs = eng_module(vs_hlsl, RSX_SPV_STAGE_VERTEX, "vp");
    if (!vs) return 0;
    const int wp = strstr(ps_hlsl, "input.position") != NULL;
    VkShaderModule fs = eng_fp_module(ps_hlsl);
    if (!fs) return 0;
    EngPipeline* p = &s_pipe[s_pipe_count];
    memset(p, 0, sizeof *p);
    p->vs = vs; p->fs = fs;
    if (wp) p->ps_plain = _strdup(ps_hlsl);   /* at any scale: a rescale rebuilds from it */
    p->rs = *rs;
    p->nattr = layout->count < RSX_DSP_NUM_VERTEX_ATTR ? layout->count : RSX_DSP_NUM_VERTEX_ATTR;
    p->stride = vertex_stride;
    p->rt_fmt = eng_vkfmt(rt_fmt);
    p->rt_count = rt_count;
    p->cube_mask = eng_cube_mask(ps_hlsl);
    p->live = 1;
    return ++s_pipe_count;
}

static VkPipeline eng_pso_for(u32 pipeline, int cls, int unscaled);
static u32 eng_pipeline_create(void* user, const char* vs_hlsl, const char* ps_hlsl,
                               const rsx_be_render_state* rs, const rsx_vertex_layout_plan* layout,
                               u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    (void)user;
    AcquireSRWLockExclusive(&s_pipe_lock);
    const u32 h = eng_pipeline_create_locked(vs_hlsl, ps_hlsl, rs, layout, vertex_stride, rt_fmt, rt_count);
    ReleaseSRWLockExclusive(&s_pipe_lock);
    /* The triangle pipeline here, on the engine's build thread, not at the
     * first draw on the walker (RSX_PSO_PREWARM=0: on first use). */
    { static int warm = -1;
      if (warm < 0) { const char* e = getenv("RSX_PSO_PREWARM"); warm = !(e && e[0] == '0'); }
      if (h && warm) eng_pso_for(h, 2, 0); }
    return h;
}
static void eng_pipeline_release(void* user, u32 pipeline)
{
    (void)user;
    AcquireSRWLockExclusive(&s_pipe_lock);
    if (pipeline && pipeline <= s_pipe_count) s_pipe[pipeline - 1].live = 0;   /* kept: lists may name it */
    ReleaseSRWLockExclusive(&s_pipe_lock);
}

extern double g_rsx_frame_pso_ms;   /* rsx_draw_engine.c */
static DWORD s_walker_tid;
static VkPipeline eng_pso_for(u32 pipeline, int cls, int unscaled)
{
    if (!pipeline || pipeline > s_pipe_count) return VK_NULL_HANDLE;
    EngPipeline* p = &s_pipe[pipeline - 1];
    const int alt = unscaled && p->ps_plain && s_scale != 1.0f;
    VkPipeline* slot = alt ? &p->pso1[cls] : &p->pso[cls];
    int* failed = alt ? &p->failed1[cls] : &p->failed[cls];
    if (*slot) return *slot;
    if (*failed || !p->vs) return VK_NULL_HANDLE;
    LARGE_INTEGER q0, q1, qf; QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&q0);
    AcquireSRWLockExclusive(&s_pipe_lock);
    if (!*slot && !*failed) {
        VkShaderModule fs = p->fs;
        if (alt) {
            if (!p->fs1) p->fs1 = eng_module(p->ps_plain, RSX_SPV_STAGE_FRAGMENT, "fp (WPOS undivided)");
            fs = p->fs1;
        }
        *slot = fs ? eng_build_pso(p, cls, fs) : VK_NULL_HANDLE;
        if (!*slot) *failed = 1;
    }
    ReleaseSRWLockExclusive(&s_pipe_lock);
    QueryPerformanceCounter(&q1);
    if (GetCurrentThreadId() == s_walker_tid)
        g_rsx_frame_pso_ms += (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)qf.QuadPart;
    return *slot;
}

/* ---- samplers ---------------------------------------------------------------------- */

/* RSX_ANISO=<1..16> (default 16): anisotropic filtering for the textures the
 * title filters linearly and mipmaps -- the world's ground and walls, which
 * the RSX's own trilinear blurs at a glancing angle. Only for a guest
 * texture with a real mip chain bound to a linear, mipmapped sampler: render
 * targets, their views and snapshots (shadow maps, packed depth, post-process
 * inputs) carry data that spreading samples corrupts -- 16x over them put red
 * and cyan speckle on the ground and the cloth. 1 is off. */
static int s_has_aniso;
extern int g_rsx_aniso;   /* rsx_d3d12_engine.c */
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

/* A guest texture with a mip chain (not a target, view or snapshot). */
static int eng_aniso_tex(u32 handle)
{
    const EngObj* o = eng_obj(handle);
    return o && o->kind == OBJ_TEXTURE && o->mips > 1;
}
static int eng_sampler_slot(const rsx_be_sampler_desc* d, int aniso_ok)
{
    aniso_ok = aniso_ok && eng_aniso() > 1 && d->min_linear && d->mag_linear && d->mip_present;
    const u64 key = ((u64)(aniso_ok ? eng_aniso() : 0) << 56) | (u64)d->min_linear | ((u64)d->mag_linear << 1)
                  | ((u64)d->mip_linear << 2) | ((u64)d->mip_present << 3)
                  | ((u64)d->wrap_s << 4) | ((u64)d->wrap_t << 8) | ((u64)d->wrap_r << 12)
                  | ((u64)(u32)(d->min_lod * 256.0f) << 16)
                  | ((u64)(u32)(d->max_lod * 256.0f) << 32);
    for (u32 i = 0; i < s_samp_count; i++) if (s_samp[i].key == key) return (int)i;
    if (s_samp_count >= ENG_MAX_SAMPLERS) return -1;
    VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.minFilter = d->min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.magFilter = d->mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = d->mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = gcm_wrap(d->wrap_s); si.addressModeV = gcm_wrap(d->wrap_t); si.addressModeW = gcm_wrap(d->wrap_r);
    si.minLod = d->min_lod;
    si.maxLod = d->mip_present ? d->max_lod : 0.0f;
    if (si.maxLod < si.minLod) si.maxLod = si.minLod;
    si.maxAnisotropy = 1.0f;
    if (s_has_aniso && aniso_ok) {
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = (float)eng_aniso();
        if (si.maxAnisotropy > s_props.limits.maxSamplerAnisotropy) si.maxAnisotropy = s_props.limits.maxSamplerAnisotropy;
    }
    si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkSampler s = VK_NULL_HANDLE;
    if (vkCreateSampler(s_dev, &si, NULL, &s) != VK_SUCCESS) return -1;
    s_samp[s_samp_count].key = key; s_samp[s_samp_count].smp = s;
    return (int)s_samp_count++;
}

/* ---- resources ------------------------------------------------------------------------ */

static void eng_upload_rows(u32 handle, u32 layer, u32 mip, u32 w, u32 h, const void* src, u32 row_bytes, u32 rows)
{
    EngObj* o = eng_obj(handle);
    if (!o || s_rec_count >= ENG_MAX_RECORDS) { s_dropped++; return; }
    const u32 bpp = eng_bpp(o->fmt);
    const int bc = fmt_is_bc(o->fmt);
    /* A row of texels (or of 4x4 blocks) the buffer can describe in texels. */
    u32 row_units = (row_bytes + bpp - 1u) / bpp;
    const u32 pitch = row_units * bpp;
    u32 off;
    if (!eng_stage_reserve(pitch * rows, &off)) return;
    u8* dst = s_stage[s_stage_cur].mapped + off;
    if (pitch == row_bytes) memcpy(dst, src, (size_t)row_bytes * rows);
    else for (u32 y = 0; y < rows; y++) memcpy(dst + (size_t)y * pitch, (const u8*)src + (size_t)y * row_bytes, row_bytes);
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_UPLOAD;
    r->depth = handle;
    r->up_off = off; r->up_layer = layer; r->up_mip = mip; r->up_w = w; r->up_h = h;
    r->up_row_texels = bc ? row_units * 4u : row_units;
    r->up_rows_texels = bc ? rows * 4u : rows;
}

static u32 eng_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h, u32 mips, u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)user;
    const VkFormat vf = eng_vkfmt(fmt);
    const u32 handle = eng_image(OBJ_TEXTURE, vf, w, h, mips, faces,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!handle) return 0;
    EngObj* o = &s_obj[handle - 1];
    o->view = make_view(o->img, vf, o->faces == 6 ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D,
                        VK_IMAGE_ASPECT_COLOR_BIT, o->mips, o->faces, eng_mapping(remap, rsx_fmt));
    return handle;
}
static void eng_texture_upload(void* user, u32 handle, u32 face, u32 mip, u32 w, u32 h, const void* src, u32 row_bytes, u32 rows)
{
    (void)user;
    EngObj* o = eng_obj(handle);
    if (!o || !o->img || !src || !row_bytes || !rows) return;
    eng_upload_rows(handle, face, mip, w, h, src, row_bytes, rows);
}

static u32 eng_color_target_create(void* user, rsx_be_format fmt, u32 w, u32 h, const void* seed, u32 seed_row_bytes)
{
    (void)user;
    if (!w || !h) return 0;
    const VkFormat vf = eng_vkfmt(fmt);
    const int scaled = eng_scales(w, h);
    const u32 sw = scaled ? sc_dim(w) : w, sh = scaled ? sc_dim(h) : h;
    const u32 handle = eng_image(OBJ_COLOR, vf, sw, sh, 1, 1,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!handle) return 0;
    EngObj* o = &s_obj[handle - 1];
    o->gw = w; o->gh = h;
    o->view = make_view(o->img, vf, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, k_identity);
    o->att = make_view(o->img, vf, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, k_identity);
    if (!seed || !seed_row_bytes) eng_clear_new(handle);
    if (seed && seed_row_bytes) {
        if (sw == w && sh == h) eng_upload_rows(handle, 0, 0, w, h, seed, seed_row_bytes, h);
        else {
            const u32 bpp = eng_bpp(vf);
            u8* big = (u8*)malloc((size_t)sw * bpp * sh);
            if (big) {
                for (u32 y = 0; y < sh; y++) {
                    const u8* srow = (const u8*)seed + (size_t)((u64)y * h / sh) * seed_row_bytes;
                    u8* drow = big + (size_t)y * sw * bpp;
                    for (u32 x = 0; x < sw; x++)
                        memcpy(drow + (size_t)x * bpp, srow + (size_t)((u64)x * w / sw) * bpp, bpp);
                }
                eng_upload_rows(handle, 0, 0, sw, sh, big, sw * bpp, sh);
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
    if (!o || !o->img) return 0;
    for (u32 i = 0; i < s_view_count; i++)
        if (s_view[i].surface == surface && s_view[i].remap == remap && s_view[i].format == rsx_format)
            return s_view[i].view;
    if (s_view_count >= ENG_MAX_VIEWS) return 0;
    const u32 handle = eng_obj_add(OBJ_VIEW);
    if (!handle) return 0;
    EngObj* v = &s_obj[handle - 1];
    o = eng_obj(surface);
    v->alias = surface; v->fmt = o->fmt; v->w = o->w; v->h = o->h; v->gw = o->gw; v->gh = o->gh; v->mips = 1; v->faces = 1;
    v->view = make_view(o->img, o->fmt, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, eng_mapping(remap, rsx_format));
    s_view[s_view_count].surface = surface; s_view[s_view_count].remap = remap;
    s_view[s_view_count].format = rsx_format; s_view[s_view_count].view = handle;
    s_view_count++;
    return handle;
}

static u32 eng_depth_target_create_host(u32 w, u32 h, u32 gw, u32 gh)
{
    const u32 handle = eng_image(OBJ_DEPTH, ENG_DEPTH_FMT, w, h, 1, 1,
                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!handle) return 0;
    EngObj* o = &s_obj[handle - 1];
    o->gw = gw; o->gh = gh;
    o->view = make_view(o->img, ENG_DEPTH_FMT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1, k_identity);
    o->att = make_view(o->img, ENG_DEPTH_FMT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 1, 1, k_identity);
    eng_clear_new(handle);
    return handle;
}
static u32 eng_depth_target_create(void* user, u32 w, u32 h)
{
    (void)user;
    if (!eng_scales(w, h)) return eng_depth_target_create_host(w, h, w, h);
    return eng_depth_target_create_host(sc_dim(w), sc_dim(h), w, h);
}
static u32 eng_fallback_depth(u32 w, u32 h)
{
    for (u32 i = 0; i < s_fallback_n; i++) {
        EngObj* o = eng_obj(s_fallback_depth[i]);
        if (o && o->w == w && o->h == h) return s_fallback_depth[i];
    }
    const u32 d = eng_depth_target_create_host(w, h, w, h);
    if (d && s_fallback_n < 8) s_fallback_depth[s_fallback_n++] = d;
    return d;
}

static u32 eng_color_snapshot(void* user, u32 surface)
{
    (void)user;
    EngObj* t = eng_obj(surface);
    if (!t || !t->img || s_rec_count >= ENG_MAX_RECORDS) return 0;
    static struct { u32 surface, snap; } pool[32]; static u32 npool;
    u32 dst = 0, slot = npool;
    for (u32 i = 0; i < npool; i++) if (pool[i].surface == surface) { slot = i; break; }
    if (slot < npool) {
        EngObj* d = eng_obj(pool[slot].snap);
        if (d && d->img && d->w == t->w && d->h == t->h && d->fmt == t->fmt) dst = pool[slot].snap;
    }
    if (!dst) {
        dst = eng_image(OBJ_SNAPSHOT, t->fmt, t->w, t->h, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        if (!dst) return 0;
        t = eng_obj(surface);
        EngObj* d = &s_obj[dst - 1];
        d->gw = t->gw; d->gh = t->gh;
        d->view = make_view(d->img, d->fmt, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, k_identity);
        if (slot == npool) { if (npool >= 32) return 0; npool++; }
        pool[slot].surface = surface; pool[slot].snap = dst;
    }
    EngRecord* r = &s_rec[s_rec_count++];
    memset(r, 0, sizeof *r);
    r->kind = ENG_REC_COLOR_COPY; r->depth = surface; r->resolve_dst = dst;
    return dst;
}

static u32 eng_depth_snapshot_common(u32 depth, u32 w, u32 h, int packed)
{
    EngObj* z = eng_obj(depth);
    if (!z || !z->img || !(packed ? s_depth_pack_pso : s_depth_pso) || s_rec_count >= ENG_MAX_RECORDS) return 0;
    const u32 gw = w, gh = h;
    if (eng_obj_scaled(depth)) { w = sc_dim(w); h = sc_dim(h); }
    const VkFormat f = packed ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R32_SFLOAT;
    const u32 dst = eng_image(OBJ_SNAPSHOT, f, w, h, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!dst) return 0;
    EngObj* d = &s_obj[dst - 1];
    d->gw = gw; d->gh = gh;
    d->view = make_view(d->img, f, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, k_identity);
    d->att = make_view(d->img, f, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, k_identity);
    EngRecord* rec = &s_rec[s_rec_count++];
    memset(rec, 0, sizeof *rec);
    rec->kind = ENG_REC_DEPTH_RESOLVE; rec->depth = depth; rec->resolve_dst = dst; rec->resolve_packed = packed;
    return dst;
}
static u32 eng_depth_snapshot(void* user, u32 depth, u32 w, u32 h) { (void)user; return eng_depth_snapshot_common(depth, w, h, 0); }
static u32 eng_depth_snapshot_rgba8(void* user, u32 depth, u32 w, u32 h) { (void)user; return eng_depth_snapshot_common(depth, w, h, 1); }

/* ---- occlusion queries -------------------------------------------------------------- */

static u32 eng_query_begin(void* user)
{
    (void)user;
    const u32 slot = s_vis_next++ % ENG_VIS_SLOTS;
    s_vis_count[slot] = 0;
    return slot + 1;
}
static void eng_query_set(void* user, u32 query) { (void)user; s_vis_cur = query; }
static void eng_query_report(void* user, u32 query, u32 report_index)
{
    (void)user;
    if (!query || s_vis_npending >= ENG_MAX_REPORTS) return;
    s_vis_pending[s_vis_npending].slot = query - 1;
    s_vis_pending[s_vis_npending].index = report_index;
    s_vis_npending++;
}

/* ---- recording ---------------------------------------------------------------------- */

static void eng_bind_targets(void* user, const u32* surfaces, u32 count, u32 depth)
{
    (void)user;
    if (count > RSX_BE_MAX_COLOR_TARGETS) count = RSX_BE_MAX_COLOR_TARGETS;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++) s_pending.rt[i] = (i < count) ? surfaces[i] : 0;
    s_pending.nrt = count;
    s_pending.depth = depth;
}
static void eng_bind_pipeline(void* user, u32 pipeline) { (void)user; s_pending.pipeline = pipeline; }
static void eng_bind_vs_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (bytes > ENG_UBO_RANGE) bytes = ENG_UBO_RANGE;
    if (!eng_stage_copy(data, bytes, &s_pending.vs_cb_off)) bytes = 0;
    s_pending.vs_cb_bytes = bytes;
    s_vs_cb_seq = s_submit_seq;
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
    if (bytes > ENG_UBO_RANGE) bytes = ENG_UBO_RANGE;
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

#include "rsx_vulkan_encode.inc"

#elif defined(_WIN32)   /* built without the Vulkan SDK: the entry points say so */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rsx_vulkan_engine.h"

int rsx_vulkan_engine_wanted(void)
{
    const char* b = getenv("RSX_BACKEND");
    return b && !_stricmp(b, "vulkan");
}
int rsx_vulkan_engine_init(u32 width, u32 height, const char* title)
{
    (void)width; (void)height; (void)title;
    fprintf(stderr, "[RSX vulkan] this build has no Vulkan backend (configured without the Vulkan SDK)\n");
    return -1;
}
void rsx_vulkan_engine_shutdown(void) {}
int rsx_vulkan_engine_active(void) { return 0; }
int rsx_vulkan_engine_pump_messages(void) { return 0; }

#endif /* _WIN32 */
