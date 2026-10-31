/*
 * ps3recomp - the platform-neutral halves of the RSX draw engine
 *
 * Everything here runs with no GPU and no host graphics API: the engine is
 * driven through rsx_draw_engine_method(), exactly as the FIFO walker drives
 * it, against a stub backend that records what it was asked for. That is the
 * only way to check the four pieces whose bugs are silent -- a surface that is
 * reallocated when it should not be, a texture unit that samples guest memory
 * instead of the render target at the same address, a texture that never
 * updates because its content hash is only taken once, and a pipeline that is
 * recompiled every time a constant changes.
 *
 * Builds and runs on Linux as well as macOS; nothing in it is Apple-specific.
 */
#include "rsx_draw_engine.h"
#include "rsx_capture.h"
#include "ps3emu/env_cache.h"
#include "ps3emu/vm_watch.h"
#include "rsx_primitives.h"   /* RSX_PRIMITIVE_* */
#include "rsx_test_programs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            printf("[PASS] " __VA_ARGS__); printf("\n");                      \
        } else {                                                              \
            printf("[FAIL] " __VA_ARGS__);                                    \
            printf("   (%s:%d)\n", __FILE__, __LINE__);                       \
            g_failures++;                                                     \
        }                                                                     \
    } while (0)

/* ---- the guest memory the engine reads ---------------------------------- */
/*
 * Two number spaces, as on the hardware: an RSX offset means one thing in the
 * IO-mapped main-memory context and another in local video memory. The test's
 * map puts main at the bottom of the buffer and local above it, which is what
 * lets the surface-keying check bind two different objects at one offset.
 */
#define GUEST_BYTES     (16u << 20)
#define GUEST_LOCAL_EA  (8u << 20)

u8* vm_base;
u32 ppu_vm_size;

u32 cellGcmResolveLocated(int local, u32 offset)
{
    return local ? GUEST_LOCAL_EA + offset : offset;
}

u32 cellGcmResolveIO(u32 offset)
{
    return offset < GUEST_LOCAL_EA ? offset : 0;
}

/* The engine's hooks into the rest of the runtime, none of which is here: the
 * GCM report and trace hook, the frame capture (off), the store watch (not
 * available: the engine hashes), the env cache (served straight from getenv)
 * and the job-chain walk timers. */
void cellGcm_set_report_value(u32 index, u32 value) { (void)index; (void)value; }
void (*g_gcm_trace_hook)(u32 type, u32 a, u32 b);
double   g_jc_walk_ms;
unsigned g_jc_walks;
ps3_env_slot g_ps3_env_cache[PS3_ENV_CACHE_SLOTS];
const char* ps3_env_fill(const char* name, unsigned first_slot) { (void)first_slot; return (getenv)(name); }   /* the real one: these files alias getenv to ps3_env */
int g_rsx_capture_on;
void rsx_capture_read(u32 location, u32 offset, u32 len) { (void)location; (void)offset; (void)len; }
void rsx_capture_method(u32 method, u32 arg) { (void)method; (void)arg; }
void rsx_capture_flip(u32 buffer_id) { (void)buffer_id; }
void rsx_capture_display_buffer(u32 id, u32 location, u32 offset, u32 pitch, u32 width, u32 height)
{ (void)id; (void)location; (void)offset; (void)pitch; (void)width; (void)height; }
int  rsx_capture_wants_mean(u32 frame) { (void)frame; return 0; }
void rsx_capture_present(u32 frame, double mean, const u32* regs, u32 nregs, const u32* vp, u32 nvp,
                         const u32* constants, u32 nconst_words, const u32* dispbuf, u32 ndispbuf)
{ (void)frame; (void)mean; (void)regs; (void)nregs; (void)vp; (void)nvp; (void)constants; (void)nconst_words; (void)dispbuf; (void)ndispbuf; }
int      vm_watch_available(void) { return 0; }
uint64_t vm_watch_arm(uint32_t ea, uint32_t len, uint32_t* unwatched) { (void)ea; if (unwatched) *unwatched = len; return 0; }
void     vm_watch_stats(uint64_t* faults, uint64_t* pages_protected) { if (faults) *faults = 0; if (pages_protected) *pages_protected = 0; }

/* ---- the stub backend ---------------------------------------------------- */

#define STUB_MAX_OBJECTS 8192
/* Colour clears kept by surface, so a test can say which targets were hit. */
#define STUB_MAX_CLEARED 8

typedef struct {
    u32 next_handle;
    u8  alive[STUB_MAX_OBJECTS];

    int n_texture_create, n_texture_release, n_texture_upload;
    int n_color_create, n_color_release, n_color_seeded;
    int n_depth_create, n_depth_snapshot;
    u32 snapshot_w, snapshot_h;        /* the last depth snapshot request */
    int n_pipeline_create;
    rsx_be_render_state pipeline_rs;   /* the last pipeline's render state */
    int n_draw, n_clear_color, n_clear_ds, n_present;
    u32 cleared[STUB_MAX_CLEARED];

    /* the last draw's bindings */
    u32 bound_tex[RSX_BE_MAX_TEXTURES];
    u32 bound_mask;
    u32 bound_rt[RSX_BE_MAX_COLOR_TARGETS];
    u32 bound_rt_count;
    u32 pipeline_rt_count;
    u32 bound_surface, bound_depth, bound_pipeline;
    u32 draw_vertices, draw_indices, draw_stride;
    float draw_first_vertex[4];   /* ATTR0 of the first vertex handed over */
    float captured_vertices[16][64];
    u32 captured_indices[24];
    rsx_topology draw_topology;
    u32 scissor[4];
    u32 stencil_ref;
    u32 last_created_texture, last_created_color;
    u32 last_released;
    float last_clear_depth;
} StubState;

static StubState stub;

static u32 stub_handle(void)
{
    if (stub.next_handle + 1 >= STUB_MAX_OBJECTS) return 0;
    const u32 h = ++stub.next_handle;
    stub.alive[h] = 1;
    return h;
}

static int stub_init(void* u, u32 w, u32 h) { (void)u; (void)w; (void)h; return 0; }
static void stub_shutdown(void* u) { (void)u; }
static void stub_submit(void* u, u32 reason) { (void)u; (void)reason; }

static u32 stub_texture_create(void* u, rsx_be_format f, u32 w, u32 h, u32 mips,
                               u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)u; (void)f; (void)w; (void)h; (void)mips; (void)faces;
    (void)remap; (void)rsx_fmt;
    stub.n_texture_create++;
    stub.last_created_texture = stub_handle();
    return stub.last_created_texture;
}

static void stub_texture_upload(void* u, u32 t, u32 face, u32 mip, u32 w, u32 h,
                                const void* src, u32 row, u32 rows)
{
    (void)u; (void)t; (void)face; (void)mip; (void)w; (void)h;
    (void)src; (void)row; (void)rows;
    stub.n_texture_upload++;
}

static void stub_release(void* u, u32 handle)
{
    (void)u;
    if (handle && handle < STUB_MAX_OBJECTS) stub.alive[handle] = 0;
    stub.last_released = handle;
}

static void stub_texture_release(void* u, u32 handle)
{ stub.n_texture_release++; stub_release(u, handle); }

static u32 stub_color_create(void* u, rsx_be_format f, u32 w, u32 h,
                             const void* seed, u32 seed_row)
{
    (void)u; (void)f; (void)w; (void)h; (void)seed_row;
    stub.n_color_create++;
    if (seed) stub.n_color_seeded++;
    stub.last_created_color = stub_handle();
    return stub.last_created_color;
}

static void stub_color_release(void* u, u32 handle)
{ stub.n_color_release++; stub_release(u, handle); }

static u32 stub_surface_view(void* u, u32 surface, u32 remap, u32 fmt)
{ (void)u; (void)remap; (void)fmt; return surface; }

static u32 stub_depth_create(void* u, u32 w, u32 h)
{ (void)u; (void)w; (void)h; stub.n_depth_create++; return stub_handle(); }

static u32 stub_depth_snapshot(void* u, u32 d, u32 w, u32 h)
{ (void)u; (void)d; stub.snapshot_w = w; stub.snapshot_h = h; stub.n_depth_snapshot++; return stub_handle(); }

static u32 stub_pipeline_create(void* u, const char* vs, const char* ps,
                                const rsx_be_render_state* rs,
                                const rsx_vertex_layout_plan* layout,
                                u32 stride, rsx_be_format rt, u32 rt_count)
{
    (void)u; (void)rs; (void)layout; (void)stride; (void)rt;
    if (!vs || !ps || !*vs || !*ps) return 0;
    stub.n_pipeline_create++;
    stub.pipeline_rt_count = rt_count;
    stub.pipeline_rs = *rs;
    return stub_handle();
}

static void stub_pipeline_release(void* u, u32 p) { (void)u; (void)p; }

static void stub_bind_targets(void* u, const u32* s, u32 count, u32 d)
{
    (void)u;
    stub.bound_surface = count ? s[0] : 0;
    stub.bound_rt_count = count;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++)
        stub.bound_rt[i] = (i < count) ? s[i] : 0;
    stub.bound_depth = d;
}
static void stub_bind_pipeline(void* u, u32 p) { (void)u; stub.bound_pipeline = p; }
static void stub_bind_vs(void* u, const void* d, u32 n) { (void)u; (void)d; (void)n; }
static void stub_bind_ps(void* u, const void* d, u32 n) { (void)u; (void)d; (void)n; }

static void stub_bind_textures(void* u, const u32* tex,
                               const rsx_be_sampler_desc* smp, u32 mask)
{
    (void)u; (void)smp;
    memcpy(stub.bound_tex, tex, sizeof stub.bound_tex);
    stub.bound_mask = mask;
}

static void stub_bind_vtextures(void* u, const u32* tex,
                                const rsx_be_sampler_desc* smp, u32 mask)
{ (void)u; (void)tex; (void)smp; (void)mask; }

static void stub_viewport(void* u, float x, float y, float w, float h)
{ (void)u; (void)x; (void)y; (void)w; (void)h; }

static void stub_scissor(void* u, u32 x, u32 y, u32 w, u32 h)
{
    (void)u;
    stub.scissor[0] = x; stub.scissor[1] = y;
    stub.scissor[2] = w; stub.scissor[3] = h;
}

static void stub_stencil_ref(void* u, u32 r) { (void)u; stub.stencil_ref = r; }

static void stub_draw(void* u, rsx_topology topology, const void* verts,
                      u32 count, u32 stride, const u32* indices,
                      u32 index_count)
{
    (void)u; (void)indices;
    stub.n_draw++;
    memset(stub.draw_first_vertex, 0, sizeof stub.draw_first_vertex);
    if (verts && stride >= sizeof stub.draw_first_vertex)
        memcpy(stub.draw_first_vertex, verts, sizeof stub.draw_first_vertex);
    memset(stub.captured_vertices, 0, sizeof stub.captured_vertices);
    memset(stub.captured_indices, 0, sizeof stub.captured_indices);
    if (verts && stride <= sizeof stub.captured_vertices[0])
        for (u32 i = 0; i < count && i < 16; i++)
            memcpy(stub.captured_vertices[i], (const u8*)verts + i * stride, stride);
    if (indices)
        memcpy(stub.captured_indices, indices,
               (index_count < 24 ? index_count : 24) * sizeof(u32));
    stub.draw_topology = topology;
    stub.draw_vertices = count;
    stub.draw_stride = stride;
    stub.draw_indices = index_count;
}

static void stub_clear_color(void* u, u32 s, const float rgba[4])
{
    (void)u; (void)rgba;
    if (stub.n_clear_color < (int)STUB_MAX_CLEARED)
        stub.cleared[stub.n_clear_color] = s;
    stub.n_clear_color++;
}

static void stub_clear_ds(void* u, u32 d, u32 f, float z, u8 s)
{ (void)u; (void)d; (void)f; (void)s; stub.n_clear_ds++; stub.last_clear_depth = z; }

static u32 presented_surface;
static void stub_present(void* u, u32 s) { (void)u; presented_surface = s; stub.n_present++; }

/* One pixel of a surface, in the R,G,B,A order rsx_texture_decode produces:
 * the readback hook has to put those back into the guest's A8R8G8B8. */
static u8 g_stub_pixel[4] = { 0x11, 0x22, 0x33, 0x44 };

static void stub_readback(void* u, u32 s, u32 x, u32 y, u32 w, u32 h,
                          void* out, u32 pitch)
{
    (void)u; (void)s; (void)x; (void)y; (void)pitch;
    if (w == 1 && h == 1) memcpy(out, g_stub_pixel, 4);
}

static const rsx_draw_backend g_stub_backend = {
    .user = NULL,
    .init = stub_init,
    .shutdown = stub_shutdown,
    .submit_and_wait = stub_submit,
    .texture_create = stub_texture_create,
    .texture_upload = stub_texture_upload,
    .texture_release = stub_texture_release,
    .color_target_create = stub_color_create,
    .color_target_release = stub_color_release,
    .surface_view = stub_surface_view,
    .depth_target_create = stub_depth_create,
    .depth_target_release = stub_release,
    .depth_snapshot = stub_depth_snapshot,
    .pipeline_create = stub_pipeline_create,
    .pipeline_release = stub_pipeline_release,
    .bind_targets = stub_bind_targets,
    .bind_pipeline = stub_bind_pipeline,
    .bind_vs_constants = stub_bind_vs,
    .bind_ps_constants = stub_bind_ps,
    .bind_textures = stub_bind_textures,
    .bind_vertex_textures = stub_bind_vtextures,
    .set_viewport = stub_viewport,
    .set_scissor = stub_scissor,
    .set_stencil_ref = stub_stencil_ref,
    .draw = stub_draw,
    .clear_color = stub_clear_color,
    .clear_depth_stencil = stub_clear_ds,
    .present = stub_present,
    .readback = stub_readback,
};

/* ---- driving the engine -------------------------------------------------- */

#define M_CONTEXT_DMA_COLOR_B 0x018C
#define M_CONTEXT_DMA_COLOR_A 0x0194
#define M_CONTEXT_DMA_COLOR_C 0x01B4
#define M_CONTEXT_DMA_COLOR_D 0x01B8
#define M_SURFACE_CLIP_H      0x0200
#define M_SURFACE_CLIP_V      0x0204
#define M_SURFACE_FORMAT      0x0208
#define M_COLOR_A_OFFSET      0x0210
#define M_ZETA_OFFSET         0x0214
#define M_COLOR_B_OFFSET      0x0218
#define M_COLOR_TARGET        0x0220
#define M_COLOR_C_OFFSET      0x0288
#define M_COLOR_D_OFFSET      0x028C
#define M_SCISSOR_H           0x08C0
#define M_SCISSOR_V           0x08C4
#define M_VTXBUF_OFFSET       0x1680
#define M_VTXFMT              0x1740
#define M_BEGIN_END           0x1808
#define M_DRAW_ARRAYS         0x1814
#define M_INLINE_ARRAY        0x1818
#define M_TEX_UNIT            0x1A00   /* + unit * 0x20 */
#define M_STENCIL_FUNC_REF    0x0334
#define M_CONTEXT_DMA_ZETA    0x0198
#define M_ZSTENCIL_CLEAR      0x1D8C
#define M_ZETA_PITCH          0x022C
#define M_COLOR_MASK          0x0324
#define M_DEPTH_WRITE         0x0A70
#define M_DEPTH_TEST          0x0A74
#define M_CLEAR_BUFFERS       0x1D94
#define M_VP_UPLOAD_CONST_ID  0x1EFC
#define M_VP_UPLOAD_CONST     0x1F00

#define DMA_LOCAL  0xFEED0000u
#define DMA_MAIN   0xFEED0001u

#define VTX_OFFSET  0x00010000u        /* main memory, well clear of ea 0 */
#define TEX_OFFSET  0x00020000u

static void m(u32 method, u32 arg) { rsx_draw_engine_method(method, arg); }

static void engine_up_with(const rsx_draw_backend* be)
{
    memset(&stub, 0, sizeof stub);
    rsx_draw_engine_set_backend(be);
    if (rsx_draw_engine_init(1280, 720) != 0) {
        printf("[FAIL] engine init\n");
        exit(1);
    }
    /* One 256x256 surface in local memory, and a vertex array in main. */
    m(M_CONTEXT_DMA_COLOR_A, DMA_LOCAL);
    m(M_SURFACE_CLIP_H, 256u << 16);
    m(M_SURFACE_CLIP_V, 256u << 16);
    m(M_COLOR_A_OFFSET, 0x40000u);
    /* type 2 (float32), size 4, stride 16; bit 31 of the offset = main. */
    m(M_VTXFMT, 2u | (4u << 4) | (16u << 8));
    m(M_VTXBUF_OFFSET, 0x80000000u | VTX_OFFSET);
}

static void engine_up(void) { engine_up_with(&g_stub_backend); }

static void engine_down(void)
{
    rsx_draw_engine_shutdown();
    rsx_draw_engine_set_backend(NULL);
}

static void draw_triangle(void)
{
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    m(M_DRAW_ARRAYS, 0u | ((3u - 1u) << 24));
    m(M_BEGIN_END, 0u);
}

/* Bind unit 0 at `offset` in `location` (0 = local, 1 = main), 4x4 A8R8G8B8
 * linear with one level. */
static void bind_texture_unit(u32 location, u32 offset)
{
    const u32 dma = (location == RSX_LOCATION_MAIN) ? 2u : 1u;
    m(M_TEX_UNIT + 0x00, offset);
    m(M_TEX_UNIT + 0x04, dma | (2u << 4) | ((0x85u | 0x20u) << 8) | (1u << 16));
    m(M_TEX_UNIT + 0x0C, 0x80000000u);        /* control0: unit enable */
    m(M_TEX_UNIT + 0x10, 0xAAE4u);            /* identity crossbar */
    m(M_TEX_UNIT + 0x18, (4u << 16) | 4u);    /* image rect 4x4 */
}

/* ---- 1. the surface set -------------------------------------------------- */

static void test_surface_set(void)
{
    printf("-- surface set\n");
    engine_up();

    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 1 && stub.n_clear_color == 1,
          "a clear registers the bound colour surface (%d create, %d clear)",
          stub.n_color_create, stub.n_clear_color);
    const u32 first = stub.last_created_color;

    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 1,
          "the same (context, offset, size, format) is the same surface");

    /* A size change reallocates and retires the old target. */
    m(M_SURFACE_CLIP_H, 512u << 16);
    m(M_SURFACE_CLIP_V, 512u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 2 && stub.n_color_release == 1 &&
          stub.alive[first] == 0,
          "a size change reallocates and releases the old target");

    /* An offset change is a different surface, and does not disturb the
     * first. */
    m(M_COLOR_A_OFFSET, 0x80000u);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 3 && stub.n_color_release == 1,
          "a different offset is a different surface");

    /* The same offset in the OTHER context DMA is also a different surface:
     * RSX offsets are two overlapping number spaces. */
    m(M_CONTEXT_DMA_COLOR_A, DMA_MAIN);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 4 && stub.n_color_release == 1,
          "the same offset in another context DMA is another surface");

    /* Main memory that the IO table maps is seeded from the guest's own
     * bytes; local memory is not, because nothing guarantees it is backed. */
    CHECK(stub.n_color_seeded == 1,
          "only the main-memory surface was seeded from guest bytes (%d)",
          stub.n_color_seeded);

    /* An implausible clip does not destroy a live target. */
    const int creates = stub.n_color_create;
    m(M_SURFACE_CLIP_H, 60000u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == creates,
          "an implausible surface declaration keeps the existing target");

    engine_down();
}

/* ---- 2. surface aliasing on texture bind --------------------------------- */

static void test_surface_aliasing(void)
{
    printf("-- surface aliasing\n");
    engine_up();

    /* Render into a surface in MAIN memory, then sample a texture unit bound
     * at the same offset in the same context: the unit must read the live
     * target, not the guest bytes behind it. That is what every reflection,
     * shadow composite and post-process pass depends on. */
    m(M_CONTEXT_DMA_COLOR_A, DMA_MAIN);
    m(M_COLOR_A_OFFSET, TEX_OFFSET);
    m(M_CLEAR_BUFFERS, 0xF0u);
    const u32 surface = stub.last_created_color;

    /* Now draw into a DIFFERENT surface while sampling that one. */
    m(M_COLOR_A_OFFSET, 0x90000u);
    bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET);
    draw_triangle();
    CHECK(stub.n_draw == 1, "the draw executed");
    CHECK(stub.bound_mask == 1u, "unit 0 is the only bound unit (0x%X)",
          stub.bound_mask);
    CHECK(stub.bound_tex[0] == surface,
          "the unit samples the registered surface (%u, want %u)",
          stub.bound_tex[0], surface);
    CHECK(stub.n_texture_create == 0,
          "and nothing was uploaded from guest memory (%d uploads)",
          stub.n_texture_create);

    /* A unit naming the surface the pass is drawing INTO must not alias it:
     * reading a target a pass is writing is undefined on every API. */
    m(M_COLOR_A_OFFSET, TEX_OFFSET);
    draw_triangle();
    CHECK(stub.bound_tex[0] != surface,
          "a unit naming its own render target does not alias it");

    /* The same offset in the other context DMA is a different object. */
    m(M_COLOR_A_OFFSET, 0x90000u);
    bind_texture_unit(RSX_LOCATION_LOCAL, TEX_OFFSET);
    const int before = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.n_texture_create == before + 1 && stub.bound_tex[0] != surface,
          "a unit in the other context uploads instead of aliasing");

    engine_down();
}

/* ---- 2b. the MRT set ----------------------------------------------------- */

static void test_mrt_set(void)
{
    printf("-- MRT set\n");
    engine_up();

    /* Targets B, C and D alongside the A engine_up declared, then MRT3 --
     * the way a deferred pass declares its G-buffer. */
    m(M_CONTEXT_DMA_COLOR_B, DMA_LOCAL);
    m(M_CONTEXT_DMA_COLOR_C, DMA_LOCAL);
    m(M_CONTEXT_DMA_COLOR_D, DMA_LOCAL);
    m(M_COLOR_B_OFFSET, 0x50000u);
    m(M_COLOR_C_OFFSET, 0x60000u);
    m(M_COLOR_D_OFFSET, 0x70000u);
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_MRT3);

    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_color_create == 4,
          "an MRT set registers all four targets (%d)", stub.n_color_create);
    CHECK(stub.n_clear_color == 4,
          "and one CLEAR_SURFACE clears every one of them (%d)",
          stub.n_clear_color);
    CHECK(stub.cleared[0] != stub.cleared[1] &&
          stub.cleared[1] != stub.cleared[2] &&
          stub.cleared[2] != stub.cleared[3] &&
          stub.cleared[0] != stub.cleared[3],
          "four distinct surfaces, not target A four times (%u,%u,%u,%u)",
          stub.cleared[0], stub.cleared[1], stub.cleared[2], stub.cleared[3]);

    draw_triangle();
    CHECK(stub.bound_rt_count == 4 && stub.bound_rt[0] == stub.cleared[0],
          "the draw binds all four, A first (%u bound)", stub.bound_rt_count);
    CHECK(stub.pipeline_rt_count == 4,
          "and its pipeline is built for four attachments (%u)",
          stub.pipeline_rt_count);

    /* A texture unit naming target C must not alias it: reading a target the
     * pass writes is undefined, and an MRT set writes more than one. */
    bind_texture_unit(RSX_LOCATION_LOCAL, 0x60000u);
    const int before = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.n_texture_create == before + 1 &&
          stub.bound_tex[0] != stub.bound_rt[2],
          "a unit naming target C uploads instead of aliasing it");

    /* Back to target A alone: one attachment, and a pipeline of its own,
     * because the attachment count is part of the pipeline key. */
    const int pipelines = stub.n_pipeline_create;
    m(M_COLOR_TARGET, CELL_GCM_SURFACE_TARGET_0);
    draw_triangle();
    CHECK(stub.bound_rt_count == 1 && stub.pipeline_rt_count == 1 &&
          stub.n_pipeline_create == pipelines + 1,
          "target A alone binds one and builds its own pipeline (%u bound, %d)",
          stub.bound_rt_count, stub.n_pipeline_create);

    engine_down();
}

/* ---- 3. the texture cache ------------------------------------------------ */

static void test_texture_cache(void)
{
    printf("-- texture cache\n");
    engine_up();
    bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET);
    memset(vm_base + TEX_OFFSET, 0x11, 4 * 4 * 4);

    draw_triangle();
    CHECK(stub.n_texture_create == 1 && stub.n_texture_upload == 1,
          "the first bind uploads the texture");
    const u32 first = stub.last_created_texture;

    draw_triangle();
    draw_triangle();
    CHECK(stub.n_texture_create == 1,
          "a repeat bind in the same frame reuses it (%d uploads)",
          stub.n_texture_create);

    /* The guest rewrites the image in place -- an animated UI, a video frame,
     * a surface the 2D engine wrote. The cache must notice, and it costs one
     * content hash per entry per frame, not one per draw. */
    rsx_draw_engine_present();
    memset(vm_base + TEX_OFFSET, 0x22, 4 * 4 * 4);
    draw_triangle();
    CHECK(stub.n_texture_create == 2 && stub.alive[first] == 0,
          "changed guest bytes are re-decoded and the old resource retired");

    rsx_draw_engine_present();
    draw_triangle();
    CHECK(stub.n_texture_create == 2,
          "unchanged bytes in a later frame do not re-decode");

    /* Fill the cache past its capacity with distinct descriptors, then come
     * back to the first: it must have been evicted, not still resident, and
     * the most recent one must still be there. Returning nothing instead of
     * evicting is what made a recovered scene render as flat geometry. */
    for (u32 i = 0; i < RSX_DRAW_ENGINE_TEXTURE_CACHE; i++) {
        rsx_draw_engine_present();
        bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET + 0x1000u + i * 0x100u);
        draw_triangle();
    }
    CHECK(stub.n_texture_release >= 1,
          "the full cache evicts rather than giving up (%d evictions)",
          stub.n_texture_release);
    const u32 newest = stub.last_created_texture;
    const int creates = stub.n_texture_create;
    draw_triangle();
    CHECK(stub.n_texture_create == creates && stub.bound_tex[0] == newest,
          "the most recently used entry survived");

    rsx_draw_engine_present();
    bind_texture_unit(RSX_LOCATION_MAIN, TEX_OFFSET);
    draw_triangle();
    CHECK(stub.n_texture_create == creates + 1,
          "the least recently used entry was the one evicted");

    engine_down();
}

/* ---- 4. the pipeline key ------------------------------------------------- */

static void test_pipeline_key(void)
{
    printf("-- pipeline key\n");

    /* The key over the render state directly: what must and must not change
     * it. The alpha REFERENCE lives in the fragment constant block, so a
     * title fading something out must not compile a pipeline per frame. */
    rsx_be_render_state a;
    memset(&a, 0, sizeof a);
    a.alpha_test_enable = 1;
    a.alpha_func = 0x0204;
    a.alpha_ref_raw = 0x40;
    a.depth_func = 0x0201;
    const u64 base = rsx_draw_engine_hash_render_state(&a, 1469598103934665603ull);

    rsx_be_render_state b = a;
    b.alpha_ref_raw = 0xC0;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) == base,
          "the alpha reference is not part of the key");

    b = a; b.alpha_func = 0x0202;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "the alpha compare mode is");

    b = a; b.depth_func = 0x0203;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is the depth compare mode");

    b = a; b.color_mask = 0x01010101u;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is the colour mask");

    b = a; b.rt_fp16 = 1;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is an FP16 render target");

    /* MRT targets 1..3 carry their own colour mask (COLOR_MASK_MRT, 0x0370)
     * and blend enable (BLEND_ENABLE_MRT, 0x036C). inFAMOUS 2's light volumes
     * add into target 0 with targets 1 and 2 masked off; giving them target
     * 0's state added every light into the G-buffer. */
    b = a; b.color_mask_mrt = 0x0DF0u;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is the MRT colour mask");

    b = a; b.blend_mrt = 0x2u;
    CHECK(rsx_draw_engine_hash_render_state(&b, 1469598103934665603ull) != base,
          "so is the MRT blend enable");

    /* And through the engine: transform constants change every draw of every
     * frame, and must not multiply pipelines. */
    engine_up();
    draw_triangle();
    CHECK(stub.n_pipeline_create == 1, "the first draw builds a pipeline");

    for (u32 i = 0; i < 8; i++) {
        m(M_VP_UPLOAD_CONST_ID, i);
        for (u32 w = 0; w < 4; w++) m(M_VP_UPLOAD_CONST + w * 4, 0x3F800000u + i);
        draw_triangle();
    }
    CHECK(stub.n_pipeline_create == 1,
          "changing transform constants does not (%d pipelines)",
          stub.n_pipeline_create);

    /* The stencil reference is encoder state, not pipeline state. */
    m(M_STENCIL_FUNC_REF, 0x7Fu);
    draw_triangle();
    CHECK(stub.n_pipeline_create == 1 && stub.stencil_ref == 0x7Fu,
          "nor does the stencil reference, which reaches the encoder (%u)",
          stub.stencil_ref);

    /* The decode carries the two MRT registers through to the backend. */
    m(0x0370, 0x0DF0);
    m(0x036C, 0x2);
    draw_triangle();
    CHECK(stub.pipeline_rs.color_mask_mrt == 0x0DF0u && stub.pipeline_rs.blend_mrt == 0x2u,
          "the pipeline is built with COLOR_MASK_MRT and BLEND_ENABLE_MRT");

    engine_down();
}

/* ---- 5. restart cuts and topology expansion ------------------------------ */

static void test_restart_expansion(void)
{
    printf("-- restart cuts\n");

    /* Eight strip vertices with the guest's restart sentinel after the
     * fourth. Without the cut the expansion joins the two runs with two
     * triangles that were never in the mesh -- upstream's "exploded spiky
     * mesh that occluded the scene". */
    const u32 cuts[1] = { 4 };
    u32 idx[64];

    const u32 whole = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_TRIANGLE_STRIP, 8, NULL, 0);
    const u32 cut = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_TRIANGLE_STRIP, 8, cuts, 1);
    CHECK(whole == (8 - 2) * 3, "an uncut strip of 8 is 6 triangles (%u)", whole);
    CHECK(cut == (4 - 2) * 3 * 2,
          "a strip cut in half is two runs of 2 triangles (%u)", cut);

    memset(idx, 0xFF, sizeof idx);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_TRIANGLE_STRIP, 8,
                                           cuts, 1, NULL, idx);
    int spans = 0;
    for (u32 t = 0; t < cut / 3; t++) {
        int below = 0, above = 0;
        for (u32 k = 0; k < 3; k++)
            if (idx[t * 3 + k] < 4) below++; else above++;
        if (below && above) spans = 1;
    }
    CHECK(!spans, "no triangle spans the cut");

    /* Winding alternates per triangle within a segment, and each segment
     * starts over: a strip's second triangle swaps its first two vertices. */
    CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2,
          "strip triangle 0 is 0,1,2 (%u,%u,%u)", idx[0], idx[1], idx[2]);
    CHECK(idx[3] == 2 && idx[4] == 1 && idx[5] == 3,
          "strip triangle 1 swaps to 2,1,3 (%u,%u,%u)", idx[3], idx[4], idx[5]);
    CHECK(idx[6] == 4 && idx[7] == 5 && idx[8] == 6,
          "the segment after the cut starts its winding again (%u,%u,%u)",
          idx[6], idx[7], idx[8]);

    /* Fans and quads are cut the same way. */
    const u32 fan = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_TRIANGLE_FAN, 8, cuts, 1);
    CHECK(fan == (4 - 2) * 3 * 2, "a cut fan is two runs of 2 triangles (%u)", fan);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_TRIANGLE_FAN, 8,
                                           cuts, 1, NULL, idx);
    CHECK(idx[0] == 0 && idx[3] == 0 && idx[6] == 4 && idx[9] == 4,
          "each fan segment fans around its own first vertex");

    const u32 quads = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_QUADS, 8, NULL, 0);
    CHECK(quads == 12, "two quads are four triangles (%u)", quads);

    /* A quad strip advances two vertices per quad, so eight vertices are
     * three quads whole and one quad per half once the strip is cut. */
    const u32 qstrip = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_QUAD_STRIP, 8, NULL, 0);
    const u32 qstrip_cut = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_QUAD_STRIP, 8, cuts, 1);
    CHECK(qstrip == 3 * 6, "an uncut quad strip of 8 is three quads (%u)", qstrip);
    CHECK(qstrip_cut == 2 * 6,
          "a quad strip cut in half is one quad per half (%u)", qstrip_cut);

    memset(idx, 0xFF, sizeof idx);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_QUAD_STRIP, 8,
                                           cuts, 1, NULL, idx);
    CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2 &&
          idx[3] == 1 && idx[4] == 3 && idx[5] == 2,
          "a quad is split along the diagonal between its pairs "
          "(%u,%u,%u %u,%u,%u)", idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
    CHECK(idx[6] == 4 && idx[7] == 5 && idx[8] == 6,
          "the segment after the cut pairs from its own first vertex "
          "(%u,%u,%u)", idx[6], idx[7], idx[8]);
    spans = 0;
    for (u32 t = 0; t < qstrip_cut / 3; t++) {
        int below = 0, above = 0;
        for (u32 k = 0; k < 3; k++)
            if (idx[t * 3 + k] < 4) below++; else above++;
        if (below && above) spans = 1;
    }
    CHECK(!spans, "and no quad spans the cut");

    /* A polygon is a fan around its own first vertex, and a cut starts a
     * second polygon rather than joining the two. */
    const u32 poly = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_POLYGON, 5, NULL, 0);
    const u32 poly_cut = rsx_draw_engine_topology_index_count(
        RSX_PRIMITIVE_POLYGON, 8, cuts, 1);
    CHECK(poly == (5 - 2) * 3, "a pentagon is three triangles (%u)", poly);
    CHECK(poly_cut == (4 - 2) * 3 * 2,
          "a cut polygon is two runs of 2 triangles (%u)", poly_cut);

    memset(idx, 0xFF, sizeof idx);
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_POLYGON, 8,
                                           cuts, 1, NULL, idx);
    CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2 &&
          idx[3] == 0 && idx[4] == 2 && idx[5] == 3,
          "the first polygon fans around vertex 0 (%u,%u,%u %u,%u,%u)",
          idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
    CHECK(idx[6] == 4 && idx[9] == 4,
          "the one after the cut fans around vertex 4 (%u, %u)", idx[6], idx[9]);

    /* An occurrence-to-unique map is applied to every emitted index, which is
     * what lets a remapped draw share one uploaded vertex per reference. */
    u32 remap[8];
    for (u32 i = 0; i < 8; i++) remap[i] = 7u - i;
    rsx_draw_engine_write_topology_indices(RSX_PRIMITIVE_TRIANGLES, 6,
                                           NULL, 0, remap, idx);
    CHECK(idx[0] == 7 && idx[5] == 2, "the remap is applied to every index");
}

/* ---- 6. what reaches the backend as an indexed draw ---------------------- */

static void test_indexed_draws(void)
{
    printf("-- indexed draws\n");
    engine_up();

    /* A triangle list of distinct vertices has nothing to share, so it is
     * drawn straight. */
    draw_triangle();
    CHECK(stub.draw_indices == 0 && stub.draw_vertices == 3,
          "a triangle list of distinct vertices is not indexed (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    /* A strip is expanded into a triangle list through an INDEX buffer: six
     * vertices become twelve indices over six uploaded vertices, rather than
     * twelve duplicated ones. */
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLE_STRIP);
    m(M_DRAW_ARRAYS, 0u | ((6u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_indices == (6 - 2) * 3 && stub.draw_vertices == 6,
          "a strip is indexed, not duplicated (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    /* Only the attributes the program reads are fetched, so the stride is the
     * layout's rather than all sixteen registers. The built-in program reads
     * every register, so this is the full one; a guest program that reads two
     * would halve it. */
    CHECK(stub.draw_stride == 16u * 16u,
          "the built-in program's layout is all sixteen attributes (%u bytes)",
          stub.draw_stride);

    /* Quads have no host equivalent and become an indexed triangle list. */
    m(M_BEGIN_END, RSX_PRIMITIVE_QUADS);
    m(M_DRAW_ARRAYS, 0u | ((8u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES &&
          stub.draw_indices == 12,
          "two quads become twelve triangle indices (%u)", stub.draw_indices);

    /* So do quad strips and polygons, the two the engine used to drop. Six
     * quad-strip vertices are two quads, and a five-vertex polygon is a fan
     * of three triangles. */
    m(M_BEGIN_END, RSX_PRIMITIVE_QUAD_STRIP);
    m(M_DRAW_ARRAYS, 0u | ((6u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES &&
          stub.draw_indices == 12 && stub.draw_vertices == 6,
          "a quad strip of six becomes twelve indices over six vertices "
          "(%u verts, %u indices)", stub.draw_vertices, stub.draw_indices);

    m(M_BEGIN_END, RSX_PRIMITIVE_POLYGON);
    m(M_DRAW_ARRAYS, 0u | ((5u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES &&
          stub.draw_indices == 9 && stub.draw_vertices == 5,
          "a five-vertex polygon becomes nine indices (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    /* Points and lines are drawn as they are, in the order they arrived: an
     * index buffer would be pointless and collapsing repeats would reorder
     * them. */
    m(M_BEGIN_END, RSX_PRIMITIVE_LINES);
    m(M_DRAW_ARRAYS, 0u | ((4u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_LINES && stub.draw_indices == 0 &&
          stub.draw_vertices == 4,
          "lines pass through unindexed (%u verts, %u indices)",
          stub.draw_vertices, stub.draw_indices);

    m(M_BEGIN_END, RSX_PRIMITIVE_POINTS);
    m(M_DRAW_ARRAYS, 0u | ((2u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_POINTS && stub.draw_vertices == 2,
          "so do points (%u verts)", stub.draw_vertices);

    engine_down();
}

/* ---- 7. the guest scissor ------------------------------------------------ */

static void test_scissor(void)
{
    printf("-- scissor\n");
    engine_up();

    draw_triangle();
    CHECK(stub.scissor[2] == 256 && stub.scissor[3] == 256,
          "an unwritten scissor register is the whole surface (%ux%u)",
          stub.scissor[2], stub.scissor[3]);

    m(M_SCISSOR_H, (64u << 16) | 16u);
    m(M_SCISSOR_V, (32u << 16) | 8u);
    draw_triangle();
    CHECK(stub.scissor[0] == 16 && stub.scissor[1] == 8 &&
          stub.scissor[2] == 64 && stub.scissor[3] == 32,
          "a guest scissor narrows it (%u,%u %ux%u)",
          stub.scissor[0], stub.scissor[1], stub.scissor[2], stub.scissor[3]);

    m(M_SCISSOR_H, (4096u << 16) | 0u);
    m(M_SCISSOR_V, (4096u << 16) | 0u);
    draw_triangle();
    CHECK(stub.scissor[2] == 256 && stub.scissor[3] == 256,
          "a scissor larger than the surface is clipped to it (%ux%u)",
          stub.scissor[2], stub.scissor[3]);

    engine_down();
}

/* ---- 8. the presented pixel ---------------------------------------------- */

static void test_readback(void)
{
    printf("-- readback\n");
    engine_up();
    m(M_CLEAR_BUFFERS, 0xF0u);
    rsx_draw_engine_present();
    CHECK(stub.n_present == 1, "the flip presented a surface");
    CHECK(rsx_draw_engine_readback_center() == 0x44112233u,
          "the presented pixel comes back as A8R8G8B8 (0x%08X)",
          rsx_draw_engine_readback_center());
    engine_down();
}

static void test_queued_flip_buffer(void)
{
    engine_up();
    draw_triangle();
    u32 first = stub.bound_rt[0];
    rsx_draw_engine_set_display_buffer(0, RSX_LOCATION_LOCAL, 0x40000, 1024, 256, 256);
    m(M_COLOR_A_OFFSET, 0x60000);
    draw_triangle();
    u32 second = stub.bound_rt[0];
    rsx_draw_engine_set_display_buffer(1, RSX_LOCATION_LOCAL, 0x60000, 1024, 256, 256);
    rsx_draw_engine_present_buffer(0);
    CHECK(presented_surface == first, "queued flip selects first buffer over current target");
    rsx_draw_engine_present_buffer(1);
    CHECK(presented_surface == second && first != second, "queued flip selects second buffer");
    rsx_draw_engine_present();
    CHECK(presented_surface == second, "host present retains last queued buffer");
    engine_down();
}

/* ---- indexed draws: big-endian index runs, restart sentinels ------------ */

#define IDX_OFFSET 0x00030000u
static void put_be(u8* p, u32 v, u32 bytes)
{
    for (u32 i = 0; i < bytes; i++) p[i] = (u8)(v >> (8u * (bytes - 1u - i)));
}
/* The x of the draw's k-th vertex, through its index list when it has one
 * (a list of distinct vertices is drawn without). */
static float drawn_x(u32 k)
{
    float x;
    memcpy(&x, stub.captured_vertices[stub.draw_indices ? stub.captured_indices[k] : k], 4);
    return x;
}
static int triangles_are(const u32* want, u32 n)
{
    if ((stub.draw_indices ? stub.draw_indices : stub.draw_vertices) != n) return 0;
    for (u32 k = 0; k < n; k++) if (drawn_x(k) != (float)want[k]) return 0;
    return 1;
}
static void indexed_case(const u32* idx, u32 n, int u32_indices, int restart, u32 sentinel,
                         u32 prim, const u32* want, u32 nwant, const char* what)
{
    engine_up();
    u8 saved[8 * 16]; memcpy(saved, vm_base + VTX_OFFSET, sizeof saved);
    for (u32 i = 0; i < 8; i++) {
        const float f[4] = { (float)i, 0.0f, 0.0f, 1.0f };
        for (u32 c = 0; c < 4; c++) { u32 w; memcpy(&w, &f[c], 4); put_be(vm_base + VTX_OFFSET + i * 16 + c * 4, w, 4); }
    }
    const u32 esz = u32_indices ? 4u : 2u;
    for (u32 i = 0; i < n; i++) put_be(vm_base + IDX_OFFSET + 8 + i * esz, idx[i], esz);
    m(0x181C, IDX_OFFSET);                                 /* index array offset */
    m(0x1820, 1u | ((u32_indices ? 0u : 1u) << 4));       /* main memory, type */
    m(0x1DAC, restart ? 1u : 0u);
    m(0x1DB0, sentinel);
    m(M_BEGIN_END, prim);
    m(0x1824, (8u / esz) | ((n - 1u) << 24));             /* first index past 8 bytes */
    m(M_BEGIN_END, 0u);
    CHECK(stub.n_draw == 1 && triangles_are(want, nwant), "%s (%u indices, %u vertices drawn)", what, stub.draw_indices, stub.draw_vertices);
    memcpy(vm_base + VTX_OFFSET, saved, sizeof saved);
    memset(vm_base + IDX_OFFSET, 0, 64);
    m(0x1DAC, 0u);
    engine_down();
}
static void test_indexed_runs(void)
{
    printf("-- indexed runs\n");
    static const u32 strip[] = { 0, 1, 2, 3, 0xFFFF, 4, 5, 6, 7 };
    static const u32 strip_tris[] = { 0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7 };
    indexed_case(strip, 9, 0, 1, 0xFFFF, RSX_PRIMITIVE_TRIANGLE_STRIP, strip_tris, 12,
                 "16-bit strip cut by its restart sentinel");
    static const u32 strip32[] = { 0, 1, 2, 3, 0xFFFFFFFFu, 4, 5, 6, 7 };
    indexed_case(strip32, 9, 1, 1, 0xFFFFFFFFu, RSX_PRIMITIVE_TRIANGLE_STRIP, strip_tris, 12,
                 "32-bit strip cut by its restart sentinel");
    static const u32 tris[] = { 3, 1, 2, 7, 0, 5 };
    indexed_case(tris, 6, 0, 0, 0xFFFF, RSX_PRIMITIVE_TRIANGLES, tris, 6,
                 "16-bit triangle list in index order");
    static const u32 tris32[] = { 0x0003, 0x0001, 0x0002, 0x0007, 0x0000, 0x0005 };
    indexed_case(tris32, 6, 1, 0, 0xFFFFFFFFu, RSX_PRIMITIVE_TRIANGLES, tris, 6,
                 "32-bit triangle list in index order");
    /* A 16-bit buffer ignores a sentinel that does not fit in 16 bits. */
    static const u32 big_sentinel_tris[] = { 3, 1, 2 };
    indexed_case(big_sentinel_tris, 3, 0, 1, 0x10002u, RSX_PRIMITIVE_TRIANGLES, big_sentinel_tris, 3,
                 "16-bit restart off when the sentinel is wider than 16 bits");
}

static unsigned custom_reads;
static const u8* custom_guest_read(void* user, u32 location, u32 offset, u32 bytes)
{
    CHECK(user == &custom_reads, "custom memory resolver receives its user data");
    ++custom_reads;
    u32 ea = location == 0 ? GUEST_LOCAL_EA + offset : offset;
    if ((u64)ea + bytes > GUEST_BYTES) return NULL;
    return vm_base + ea;
}

static void test_custom_guest_mapping(void)
{
    engine_up();
    rsx_draw_engine_set_guest_memory(custom_guest_read, &custom_reads);
    custom_reads = 0;
    draw_triangle();
    CHECK(custom_reads > 0, "draw resolves vertices through runner's memory map");
    unsigned calls = custom_reads;
    rsx_draw_engine_set_guest_memory(NULL, NULL);
    draw_triangle();
    CHECK(custom_reads == calls, "clearing resolver restores default mapping");
    engine_down();
}

/* ---- polygon offset ------------------------------------------------------ */

#define M_POLY_OFFSET_POINT  0x0A60
#define M_POLY_OFFSET_LINE   0x0A64
#define M_POLY_OFFSET_FILL   0x0A68
#define M_POLY_OFFSET_FACTOR 0x0A78
#define M_POLY_OFFSET_BIAS   0x0A7C

static u32 po_bits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

/* What the backend was told before the last draw. */
static float po_slope = -99.0f, po_units = -99.0f;
static u32 po_depth_bits, po_calls;
static void stub_set_depth_bias(void* u, float slope, float units, u32 bits)
{ (void)u; po_slope = slope; po_units = units; po_depth_bits = bits; po_calls++; }

/* SET_POLY_OFFSET_FILL_ENABLE with SCALE_FACTOR and BIAS reaches the backend
 * as set_depth_bias(slope, units, depth bits) before each draw, Z16 or Z24
 * by the zeta format, and 0/0 while the offset is off. A title draws its
 * shadow casters, decals and projected shadows with an offset so they do
 * not fight the depth they sit on; the engine used to drop it. */
static void test_polygon_offset(void)
{
    printf("-- polygon offset\n");
    engine_up();
    static rsx_draw_backend be;
    be = g_stub_backend;
    be.set_depth_bias = stub_set_depth_bias;
    rsx_draw_engine_set_backend(&be);
    po_calls = 0;

    draw_triangle();
    CHECK(stub.n_draw == 1, "the draw executed");
    CHECK(po_calls == 1 && po_slope == 0.0f && po_units == 0.0f,
          "no offset: the backend is told 0/0 (%u calls, slope %g units %g)",
          po_calls, po_slope, po_units);

    m(M_POLY_OFFSET_FACTOR, po_bits(-1.0f));
    m(M_POLY_OFFSET_BIAS, po_bits(-256.0f));
    draw_triangle();
    CHECK(po_slope == 0.0f && po_units == 0.0f,
          "factor and bias alone change nothing while FILL is off");

    m(M_POLY_OFFSET_FILL, 1u);
    draw_triangle();
    CHECK(po_slope == -1.0f && po_units == -256.0f && po_depth_bits == 24u,
          "FILL on: -1/-256 against a Z24S8 zeta (slope %g units %g bits %u)",
          po_slope, po_units, po_depth_bits);

    m(M_SURFACE_FORMAT, 8u | (1u << 5));          /* A8R8G8B8, Z16 */
    m(M_POLY_OFFSET_FACTOR, po_bits(0.25f));
    m(M_POLY_OFFSET_BIAS, po_bits(200.0f));
    draw_triangle();
    CHECK(po_slope == 0.25f && po_units == 200.0f && po_depth_bits == 16u,
          "a Z16 zeta says 16 bits (slope %g units %g bits %u)",
          po_slope, po_units, po_depth_bits);
    m(M_SURFACE_FORMAT, 8u | (2u << 5));          /* back to Z24S8 */

    /* The enable for what the primitive rasterises as: LINE for lines. */
    m(M_POLY_OFFSET_FILL, 0u);
    m(M_POLY_OFFSET_LINE, 1u);
    draw_triangle();
    CHECK(po_slope == 0.0f && po_units == 0.0f,
          "LINE alone does not offset a triangle (slope %g units %g)", po_slope, po_units);
    m(M_BEGIN_END, RSX_PRIMITIVE_LINES);
    m(M_DRAW_ARRAYS, 0u | ((2u - 1u) << 24));
    m(M_BEGIN_END, 0u);
    CHECK(po_slope == 0.25f && po_units == 200.0f,
          "LINE offsets a line list (slope %g units %g)", po_slope, po_units);
    m(M_POLY_OFFSET_LINE, 0u);

    /* A NaN factor is no offset, not a NaN bias. */
    m(M_POLY_OFFSET_FILL, 1u);
    m(M_POLY_OFFSET_FACTOR, 0x7FC00000u);
    draw_triangle();
    CHECK(po_slope == 0.0f && po_units == 200.0f, "a NaN factor reads 0 (slope %g)", po_slope);

    m(M_POLY_OFFSET_FILL, 0u);
    draw_triangle();
    CHECK(po_slope == 0.0f && po_units == 0.0f, "FILL off again: 0/0");
    engine_down();
}

/* NV4097_INLINE_ARRAY: the vertices arrive IN the pushbuffer.
 *
 * Nothing consumed method 0x1818 at all, so a title that builds geometry this
 * way (Guitar Hero III draws its whole 2D layer as inline QUADS) issued draws
 * that reached the backend with no vertices and were counted as empty groups.
 * The regression this guards is silent by construction -- the draw is simply
 * absent -- so the check is both that a draw happens AND that its vertices
 * came from the stream rather than from the still-bound vertex array. */
static u32 f32_bits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

static void test_inline_array(void)
{
    engine_up();
    /* ATTR0 alone: float4, stride 16. The array at VTX_OFFSET holds 1.0f in
     * every component (see main), so reading the array instead of the stream
     * is a distinguishable wrong answer. */
    m(M_VTXFMT, 2u | (4u << 4) | (16u << 8));

    const int before = stub.n_draw;
    m(M_BEGIN_END, RSX_PRIMITIVE_TRIANGLES);
    for (u32 v = 0; v < 3; v++) {
        m(M_INLINE_ARRAY, f32_bits(10.0f + (float)v));  /* x: 10, 11, 12 */
        m(M_INLINE_ARRAY, f32_bits(20.0f));
        m(M_INLINE_ARRAY, f32_bits(30.0f));
        m(M_INLINE_ARRAY, f32_bits(1.0f));
    }
    m(M_BEGIN_END, 0u);

    CHECK(stub.n_draw == before + 1, "inline array issues a draw");
    CHECK(stub.draw_vertices == 3, "inline array draws 3 vertices (got %u)",
          stub.draw_vertices);
    CHECK(stub.draw_topology == RSX_TOPOLOGY_TRIANGLES,
          "inline array keeps the primitive type");
    CHECK(stub.draw_first_vertex[0] == 10.0f &&
          stub.draw_first_vertex[1] == 20.0f &&
          stub.draw_first_vertex[2] == 30.0f,
          "inline vertices come from the stream, not the vertex array "
          "(got %g,%g,%g)", stub.draw_first_vertex[0],
          stub.draw_first_vertex[1], stub.draw_first_vertex[2]);
    engine_down();
}

/* ---- the depth clear value ---------------------------------------------- */

#define ZETA_OFFSET  0x00100000u        /* local memory */

/* ZSTENCIL_CLEAR_VALUE in the zeta's own layout, and 0 as the near plane. */
static void test_depth_clear_value(void)
{
    printf("-- depth clear value\n");
    engine_up();
    m(M_CONTEXT_DMA_ZETA, DMA_LOCAL);
    m(M_ZETA_OFFSET, ZETA_OFFSET);

    /* Z16 (depth field 1): the clear value is a bare depth16, and 0xFFFF its
     * far plane. Read as Z24S8 it would clear to 255/16777215. */
    m(M_SURFACE_FORMAT, 8u | (1u << 5));
    m(M_ZSTENCIL_CLEAR, 0xFFFFu);
    m(M_CLEAR_BUFFERS, 0x01u);
    CHECK(stub.n_clear_ds == 1 && stub.last_clear_depth == 1.0f,
          "a Z16 zeta cleared with 0xFFFF clears to 1.0 (%g)", stub.last_clear_depth);

    /* Z24S8 (depth field 2): depth24 << 8 | stencil8. */
    m(M_SURFACE_FORMAT, 8u | (2u << 5));
    m(M_ZSTENCIL_CLEAR, 0xFFFFFF00u);
    m(M_CLEAR_BUFFERS, 0x01u);
    CHECK(stub.last_clear_depth == 1.0f,
          "a Z24S8 zeta cleared with 0xFFFFFF00 clears to 1.0 (%g)", stub.last_clear_depth);

    /* 0 is the near plane. It used to read as 1.0, standing in for the
     * reset a flip does (the FIFO walker's job now). */
    m(M_ZSTENCIL_CLEAR, 0u);
    m(M_CLEAR_BUFFERS, 0x01u);
    CHECK(stub.last_clear_depth == 0.0f,
          "a zeta cleared with 0 clears to 0.0 (%g)", stub.last_clear_depth);
    engine_down();
}

/* ---- full surface and zeta tables ----------------------------------------- */

/* A depth-writing draw through a 256x256 Z24S8 zeta at `offset`; returns the
 * depth target it was given, or 0. */
static u32 depth_draw_at(u32 offset)
{
    m(M_CONTEXT_DMA_ZETA, DMA_LOCAL);
    m(M_SURFACE_FORMAT, 8u | (2u << 5));
    m(M_ZETA_OFFSET, offset);
    m(M_ZETA_PITCH, 256u * 4u);
    m(M_COLOR_MASK, 0u);
    m(M_DEPTH_TEST, 1u);
    m(M_DEPTH_WRITE, 1u);
    m(M_TEX_UNIT + 0x0C, 0u);
    stub.bound_depth = 0;
    draw_triangle();
    return stub.bound_depth;
}

/* The zeta and surface tables used to keep every target forever: once 64
 * addresses had been rendered to, a new one got no slot, and a level loaded
 * after enough cutscenes rendered its shadow maps nowhere. A full table now
 * gives up the entry used longest ago, but never one used this frame or the
 * last, and a texture read counts as a use. */
static void test_full_tables_evict_the_oldest(void)
{
    printf("-- full surface and zeta tables evict the least recently used\n");
    engine_up();
#define ZETA_AT(k) (0x00100000u + (u32)(k) * 0x1000u)
    u32 h[64];
    for (u32 k = 0; k < 64; k++) h[k] = depth_draw_at(ZETA_AT(k));
    CHECK(h[0] && h[63], "64 zetas got depth targets");
    rsx_draw_engine_present();
    rsx_draw_engine_present();

    /* Zeta 1 is read as a texture this frame, through zeta 63. */
    m(M_COLOR_MASK, 0x01010101u);
    m(M_DEPTH_TEST, 0u);
    m(M_DEPTH_WRITE, 0u);
    bind_texture_unit(RSX_LOCATION_LOCAL, ZETA_AT(1));
    m(0x1840 /* control3: pitch */, 256u * 4u);
    draw_triangle();

    int creates = stub.n_depth_create;
    const u32 n64 = depth_draw_at(ZETA_AT(64));
    CHECK(n64 && stub.n_depth_create == creates + 1 && !stub.alive[h[0]],
          "a 65th zeta takes the oldest one's slot (%d creates, zeta 0 %s)",
          stub.n_depth_create - creates, stub.alive[h[0]] ? "alive" : "released");
    const u32 n65 = depth_draw_at(ZETA_AT(65));
    CHECK(n65 && stub.alive[h[1]] && !stub.alive[h[2]] && stub.alive[h[63]],
          "the next skips the zeta read this frame and the one bound (1 %d, 2 %d, 63 %d)",
          stub.alive[h[1]], stub.alive[h[2]], stub.alive[h[63]]);

    /* Zetas 3..62 are the only old ones left; after them nothing is evicted. */
    u32 made = 0;
    for (u32 k = 66; k < 66 + 61; k++) {
        creates = stub.n_depth_create;
        depth_draw_at(ZETA_AT(k));
        if (stub.n_depth_create == creates + 1) made++;
    }
    CHECK(made == 60 && stub.alive[h[1]] && stub.alive[h[63]] && stub.alive[n64],
          "60 more evict zetas 3..62, then a full table of this frame's zetas refuses (%u)", made);
#undef ZETA_AT

    /* Colour surfaces: the same rule, and the presented one stays. */
    const u32 base_color = stub.bound_rt[0];
    m(M_DEPTH_TEST, 0u);
    m(M_DEPTH_WRITE, 0u);
    m(M_COLOR_MASK, 0x01010101u);
    m(M_CONTEXT_DMA_ZETA, 0u);
    u32 c[64];
    for (u32 k = 0; k < 63; k++) {
        m(M_COLOR_A_OFFSET, 0x00200000u + k * 0x10000u);
        draw_triangle();
        c[k] = stub.bound_rt[0];
    }
    rsx_draw_engine_present();
    rsx_draw_engine_present();
    creates = stub.n_color_create;
    m(M_COLOR_A_OFFSET, 0x00200000u + 63u * 0x10000u);
    draw_triangle();
    CHECK(stub.n_color_create == creates + 1 && !stub.alive[base_color] &&
          stub.alive[c[62]] && stub.alive[c[0]],
          "a 65th surface takes the oldest one's slot, not the presented one's (%d creates)",
          stub.n_color_create - creates);
    engine_down();
}

/* ---- clears limited to the scissor ---------------------------------------- */

/* What the rectangle hooks were asked for. */
static struct {
    int n_color, n_depth;
    u32 color_surface, depth;
    u32 color_rect[4], depth_rect[4];
    u32 depth_flags;
} g_rect;

static void stub_clear_color_rect(void* u, u32 s, const float rgba[4],
                                  u32 x, u32 y, u32 w, u32 h)
{
    (void)u; (void)rgba;
    g_rect.n_color++;
    g_rect.color_surface = s;
    g_rect.color_rect[0] = x; g_rect.color_rect[1] = y;
    g_rect.color_rect[2] = w; g_rect.color_rect[3] = h;
}

static void stub_clear_ds_rect(void* u, u32 d, u32 f, float z, u8 st,
                               u32 x, u32 y, u32 w, u32 h)
{
    (void)u; (void)z; (void)st;
    g_rect.n_depth++;
    g_rect.depth = d;
    g_rect.depth_flags = f;
    g_rect.depth_rect[0] = x; g_rect.depth_rect[1] = y;
    g_rect.depth_rect[2] = w; g_rect.depth_rect[3] = h;
}

#define RECT_IS(r, x, y, w, h) ((r)[0] == (x) && (r)[1] == (y) && (r)[2] == (w) && (r)[3] == (h))

/* CLEAR_SURFACE writes only what the scissor lets through, when the backend
 * can clear part of a target: a split-screen title clears each view under a
 * scissor on its half of one target. */
static void test_clear_rect(void)
{
    printf("-- clears limited to the scissor\n");
    static rsx_draw_backend be;
    be = g_stub_backend;
    be.clear_color_rect = stub_clear_color_rect;
    be.clear_depth_stencil_rect = stub_clear_ds_rect;

    /* The scissor on the right half of a 256x256 surface: a split-screen
     * view's clear leaves the other view alone. */
    engine_up_with(&be);
    memset(&g_rect, 0, sizeof g_rect);
    m(M_SCISSOR_H, (128u << 16) | 128u);
    m(M_SCISSOR_V, 256u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(g_rect.n_color == 1 && stub.n_clear_color == 0 &&
          RECT_IS(g_rect.color_rect, 128u, 0u, 128u, 256u) &&
          g_rect.color_surface == stub.last_created_color,
          "a colour clear under a right-half scissor clears 128,0 128x256 only"
          " (%d rect, %d whole; %u,%u %ux%u)", g_rect.n_color, stub.n_clear_color,
          g_rect.color_rect[0], g_rect.color_rect[1], g_rect.color_rect[2],
          g_rect.color_rect[3]);

    /* A scissor covering the surface, or none, is a whole clear the backend
     * can fold into the next pass. */
    m(M_SCISSOR_H, 4096u << 16);
    m(M_SCISSOR_V, 4096u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(g_rect.n_color == 1 && stub.n_clear_color == 1,
          "a scissor covering the surface clears it whole (%d rect, %d whole)",
          g_rect.n_color, stub.n_clear_color);

    /* A scissor entirely outside the surface clears nothing. */
    m(M_SCISSOR_H, (64u << 16) | 300u);
    m(M_SCISSOR_V, 64u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(g_rect.n_color == 1 && stub.n_clear_color == 1,
          "a scissor outside the surface clears nothing (%d rect, %d whole)",
          g_rect.n_color, stub.n_clear_color);

    /* Depth under the same right-half scissor: that part of the zeta, and a
     * zeta nothing initialised yet is cleared whole first, as its first bind
     * would have. */
    m(M_CONTEXT_DMA_ZETA, DMA_LOCAL);
    m(M_ZETA_OFFSET, ZETA_OFFSET);
    m(M_SCISSOR_H, (128u << 16) | 128u);
    m(M_SCISSOR_V, 256u << 16);
    const int ds_before = stub.n_clear_ds;
    m(M_CLEAR_BUFFERS, 0x03u);
    CHECK(g_rect.n_depth == 1 && stub.n_clear_ds == ds_before + 1 &&
          RECT_IS(g_rect.depth_rect, 128u, 0u, 128u, 256u),
          "a depth clear under a right-half scissor clears 128,0 128x256 of a fresh zeta"
          " after one whole clear (%d rect, %d whole; %u,%u %ux%u)", g_rect.n_depth,
          stub.n_clear_ds - ds_before, g_rect.depth_rect[0], g_rect.depth_rect[1],
          g_rect.depth_rect[2], g_rect.depth_rect[3]);
    m(M_CLEAR_BUFFERS, 0x03u);
    CHECK(g_rect.n_depth == 2 && stub.n_clear_ds == ds_before + 1,
          "the next one is the rectangle alone (%d rect, %d whole)",
          g_rect.n_depth, stub.n_clear_ds - ds_before);
    engine_down();

    /* A backend without the rectangle hooks keeps the whole-target clear. */
    engine_up();
    m(M_SCISSOR_H, (128u << 16) | 128u);
    m(M_SCISSOR_V, 256u << 16);
    m(M_CLEAR_BUFFERS, 0xF0u);
    CHECK(stub.n_clear_color == 1,
          "without clear_color_rect a scissored clear falls back to a whole one (%d)",
          stub.n_clear_color);
    engine_down();
}

int main(void)
{
    vm_base = (u8*)calloc(1, GUEST_BYTES);
    if (!vm_base) { printf("[FAIL] guest memory\n"); return 1; }
    ppu_vm_size = GUEST_BYTES;
    /* Three vertices of float4 position, so the fetch has something real to
     * read: the engine treats an unreadable ATTR0 as a failed group. */
    for (u32 i = 0; i < 3 * 4; i++) {
        u8* p = vm_base + VTX_OFFSET + i * 4;
        p[0] = 0x3F; p[1] = 0x80; p[2] = 0; p[3] = 0;    /* 1.0f big-endian */
    }

    test_surface_set();
    test_surface_aliasing();
    test_mrt_set();
    test_texture_cache();
    test_pipeline_key();
    test_restart_expansion();
    test_indexed_runs();
    test_indexed_draws();
    test_scissor();
    test_readback();
    test_custom_guest_mapping();
    test_polygon_offset();
    test_queued_flip_buffer();
    test_inline_array();
    test_depth_clear_value();
    test_full_tables_evict_the_oldest();
    test_clear_rect();

    free(vm_base);
    printf(g_failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n",
           g_failures);
    return g_failures ? 1 : 0;
}
