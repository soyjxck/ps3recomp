/*
 * ps3recomp - platform-neutral RSX draw engine over the register file
 *
 * The vtable backends in rsx_commands.c are driven by rsx_state, a curated
 * dirty-flag struct that models no primitive restart, no vertex textures and
 * no two-sided stencil, and whose setters fire only at SET_BEGIN_END. The
 * register-file dispatcher in rsx_dispatch.c has all of it already, in
 * portable C, behind the rsx_dispatch_sink interface. This engine implements
 * that sink and drives a backend through the record-oriented interface below,
 * so a title's frame reaches the host API the way the hardware issued it.
 *
 * The engine logic is carried across from libs/video/rsx_live_draw.c, the
 * NV4097 -> D3D12 engine a title has actually shipped on. That file is
 * vendored and is NOT touched: what moves here is the orchestration, and the
 * comments name the reference line numbers so the two can be compared. The
 * mathematics is already shared and already portable -- rsx_vertex_compact.c,
 * rsx_restart_cuts.h, rsx_texture_layout.c, the two decompilers.
 *
 * The split between engine and backend is semantic, at the granularity the
 * engine thinks in, not a 1:1 wrapper over one host API's calls: resources by
 * what they are for, pipelines from the two HLSL strings plus render state
 * plus the vertex layout, and per-draw binding of what a draw names. Resource
 * state tracking, descriptor lifetime and command encoding belong entirely to
 * the backend, which is why nothing here mentions a barrier.
 *
 * Handles are small integers owned by the backend; 0 always means "none".
 */
#ifndef PS3RECOMP_RSX_DRAW_ENGINE_H
#define PS3RECOMP_RSX_DRAW_ENGINE_H

#include "rsx_dispatch.h"
#include "rsx_primitives.h"
#include "rsx_texture_layout.h"
#include "rsx_vertex_compact.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- what the engine hands a backend ------------------------------------ */

/* Surface and texture formats the engine asks for. The texture side is
 * rsx_texfmt's classification, which rsx_texture_decode already produces host
 * rows for, so a backend maps one enum rather than two. */
typedef enum {
    RSX_BE_FMT_NONE = 0,
    RSX_BE_FMT_R8,
    RSX_BE_FMT_R8G8,
    RSX_BE_FMT_R8G8B8A8,      /* decoded colour: always R,G,B,A byte order */
    RSX_BE_FMT_BC1,
    RSX_BE_FMT_BC2,
    RSX_BE_FMT_BC3,
    RSX_BE_FMT_R16,
    RSX_BE_FMT_R16G16,
    RSX_BE_FMT_R16G16F,
    RSX_BE_FMT_R16G16B16A16F, /* SET_SURFACE_FORMAT 0xB, the HDR target    */
    RSX_BE_FMT_R32F,
    RSX_BE_FMT_R32G32B32A32F
} rsx_be_format;

/* One texture unit's sampler, decoded out of SET_TEXTURE_FILTER,
 * SET_TEXTURE_ADDRESS and SET_TEXTURE_CONTROL0 the way rsx_live_draw.c's
 * decode_sampler (1464-1487) reads them. The wrap fields stay as the guest's
 * own 1..8 codes because that is what both backends already translate. */
typedef struct rsx_be_sampler_desc {
    u8  min_linear, mag_linear, mip_linear, mip_present;
    u8  wrap_s, wrap_t, wrap_r;
    /* What the BORDER wrap returns outside 0..1, reduced to the colours
     * every backend can express: SET_TEXTURE_BORDER_COLOR read the way the
     * unit's format reads it (a depth format takes its depth from the low 24
     * bits). 0 transparent black (also the old behaviour), 1 opaque black,
     * 2 opaque white, 3 any other colour, which the backend clamps to the
     * edge texel instead -- nearer than black for the colours seen (a video
     * title's chroma planes border at 0x80, neutral). */
    u8  border;
    float min_lod, max_lod;
} rsx_be_sampler_desc;

/* The NV4097 render state a pipeline is built from: rsx_live_draw.c's
 * render_state_t (1312-1327) with the D3D12 enums left undecoded, since the
 * guest values are the portable form and each backend has its own table.
 * The stencil REFERENCE is deliberately absent -- it is set on the encoder,
 * not baked into the pipeline, and so is not part of the pipeline key. */
typedef struct rsx_be_render_state {
    u32 alpha_test_enable, alpha_func, alpha_ref_raw, alpha_ref_format;
    u32 blend_enable, sf_rgb, df_rgb, sf_a, df_a, eq_rgb, eq_a;
    u32 depth_test, depth_write, depth_func;
    u32 cull_enable, cull_face, front_face;
    u32 color_mask;
    /* Targets 1..3 of an MRT set: COLOR_MASK_MRT (0x0370), one nibble per
     * target at [4i..4i+3] = A,R,G,B, and BLEND_ENABLE_MRT (0x036C), bit i.
     * Target 0 keeps color_mask/blend_enable. Both reset to 0 (RPCS3). */
    u32 color_mask_mrt, blend_mrt;
    u32 rt_fp16;                    /* the target is an FP16 HDR surface */
    u32 stencil_enable, stencil_two_sided;
    u32 s_func, s_func_mask, s_write_mask, s_fail, s_zfail, s_zpass;
    u32 bs_func, bs_fail, bs_zfail, bs_zpass;
} rsx_be_render_state;

/* submit_and_wait reasons, for a backend that wants to log or count them. */
#define RSX_BE_FLUSH_VERTEX_RING     0
#define RSX_BE_FLUSH_GUEST_REFERENCE 1
#define RSX_BE_FLUSH_SHUTDOWN        2
#define RSX_BE_FLUSH_QUERY_FENCE     3   /* a fence follows occlusion queries */

/* clear_depth_stencil flags */
#define RSX_BE_CLEAR_DEPTH   0x1u
#define RSX_BE_CLEAR_STENCIL 0x2u

#define RSX_BE_MAX_TEXTURES        RSX_DSP_NUM_TEXTURES
#define RSX_BE_MAX_VERTEX_TEXTURES RSX_DSP_NUM_VERTEX_TEXTURES
/* Colour targets a draw can name: A alone, or an MRT set of A, B, C and D. */
#define RSX_BE_MAX_COLOR_TARGETS   4

/* How many distinct guest textures the engine keeps before it evicts the
 * least recently used one. Public so the eviction test can be exact about
 * where the boundary is rather than assuming a number. */
#define RSX_DRAW_ENGINE_TEXTURE_CACHE 1024

/* ---- the backend interface ---------------------------------------------- */

typedef struct rsx_draw_backend {
    void* user;

    /* Lifecycle and submission. submit_and_wait is fully synchronous: the
     * engine recycles its staging the moment it returns, which is the same
     * contract as rsx_live_draw.c's ld_flush (1659-1736). */
    int  (*init)(void* user, u32 width, u32 height);
    void (*shutdown)(void* user);
    void (*submit_and_wait)(void* user, u32 reason);

    /* Textures. `remap` is the raw TEXTURE_CONTROL1 crossbar and `rsx_fmt`
     * the SET_TEXTURE_FORMAT byte it is decoded against, because the crossbar
     * is a property of the view rather than of the pixels. Levels arrive one
     * upload each, already decoded to host rows by rsx_texture_decode. */
    u32  (*texture_create)(void* user, rsx_be_format fmt, u32 w, u32 h,
                           u32 mips, u32 faces, u32 remap, u32 rsx_fmt);
    void (*texture_upload)(void* user, u32 texture, u32 face, u32 mip,
                           u32 w, u32 h, const void* src, u32 row_bytes,
                           u32 rows);
    void (*texture_release)(void* user, u32 texture);

    /* Colour targets. `seed` is the guest's own bytes for the surface,
     * decoded to R,G,B,A rows, or NULL when they could not be resolved: a
     * title CPU-initialises its render targets and then samples them before
     * anything draws into them, which is why the D3D12 backend's off_rt_get
     * seeds a new target the same way. */
    u32  (*color_target_create)(void* user, rsx_be_format fmt, u32 w, u32 h,
                                const void* seed, u32 seed_row_bytes);
    void (*color_target_release)(void* user, u32 surface);
    /* A texture handle for a colour target a texture unit names, wearing that
     * unit's TEXTURE_CONTROL1 crossbar. An uploaded texture bakes the crossbar
     * in when it is decoded, so only a surface bind needs this; 0 means the
     * backend has no view and the engine binds the target itself. */
    u32  (*surface_view)(void* user, u32 surface, u32 remap, u32 rsx_format);

    /* One depth target per guest zeta address. */
    u32  (*depth_target_create)(void* user, u32 w, u32 h);
    void (*depth_target_release)(void* user, u32 depth);
    /* Resolve a depth target into a sampleable texture and return it; 0 when
     * the backend cannot. The engine only asks after a depth-writing draw. */
    u32  (*depth_snapshot)(void* user, u32 depth, u32 w, u32 h);
    /* The depth target packed into RGBA8 as its D24S8 bytes read through an
     * A8R8G8B8 texture (R = depth[15:8], G = depth[7:0], B = stencil,
     * A = depth[23:16]); 0 when unsupported. */
    u32  (*depth_snapshot_rgba8)(void* user, u32 depth, u32 w, u32 h);
    /* A copy of a colour target as it stands at this point of the stream,
     * for a draw that samples the target it writes (a title's in-place
     * post-process, legal on the RSX and undefined on every host API); 0
     * when the backend cannot. The engine binds the copy, with the unit's
     * crossbar, in place of the live target. */
    u32  (*color_snapshot)(void* user, u32 surface);
    /* Occlusion queries (NV4097 ZPASS reports). query_begin returns a fresh
     * zeroed sample counter (0 when the backend has none); query_set routes
     * the samples that pass the depth/stencil tests in the following draws
     * into it (0 stops counting); query_report asks for its total to be
     * delivered, once the GPU has run those draws, through
     * rsx_draw_engine_query_result(report_index, count). */
    u32  (*query_begin)(void* user);
    void (*query_set)(void* user, u32 query);
    void (*query_report)(void* user, u32 query, u32 report_index);

    /* Pipelines, from the decompilers' HLSL plus the state that selects a
     * variant. This is the one call that hides D3DCompile against glslang
     * plus spirv-cross plus -newLibraryWithSource:, and the input layout with
     * it. 0 means the pair could not be built; the engine caches that.
     *
     * `rt_count` is how many colour attachments the pass will have, and every
     * one of them takes target A's blend and colour mask: a zero-initialised
     * secondary attachment writes nothing at all, which would leave a
     * deferred pass with only its first G-buffer plane. It is part of the
     * engine's pipeline key, because a host API matches a pipeline's
     * attachments against the pass it is encoded in. */
    u32  (*pipeline_create)(void* user, const char* vs_hlsl, const char* ps_hlsl,
                            const rsx_be_render_state* rs,
                            const rsx_vertex_layout_plan* layout,
                            u32 vertex_stride, rsx_be_format rt_fmt,
                            u32 rt_count);
    void (*pipeline_release)(void* user, u32 pipeline);

    /* Per-draw binding. The backend owns every ring the D3D12 engine used to
     * own itself, so the engine hands over CPU-side arrays and never mentions
     * a descriptor. */
    /* The colour targets this draw writes, and the depth attachment under
     * them. `count` is 1 for an ordinary draw and up to
     * RSX_BE_MAX_COLOR_TARGETS for an MRT set, whose members are all target
     * A's size and format -- the rule both render paths bind by, and the one
     * a host API enforces on a pass's attachments. */
    void (*bind_targets)(void* user, const u32* surfaces, u32 count, u32 depth);
    void (*bind_pipeline)(void* user, u32 pipeline);
    void (*bind_vs_constants)(void* user, const void* data, u32 bytes);
    /* Optional: bind the block the last bind_vs_constants staged, unchanged.
     * Returns 0 when that block is gone (a submit retired it) -- the caller
     * then binds the data again. */
    int  (*reuse_vs_constants)(void* user);
    void (*bind_ps_constants)(void* user, const void* data, u32 bytes);
    void (*bind_textures)(void* user, const u32* textures,
                          const rsx_be_sampler_desc* samplers, u32 mask);
    void (*bind_vertex_textures)(void* user, const u32* textures,
                                 const rsx_be_sampler_desc* samplers, u32 mask);
    void (*set_viewport)(void* user, float x, float y, float w, float h);
    void (*set_scissor)(void* user, u32 x, u32 y, u32 w, u32 h);
    void (*set_stencil_ref)(void* user, u32 ref);
    /* Optional. The NV4097 polygon offset for the next draw, called before
     * every draw like the stencil reference (SET_POLY_OFFSET_{POINT,LINE,
     * FILL}_ENABLE 0x0A60/0x0A64/0x0A68, SCALE_FACTOR 0x0A78, BIAS 0x0A7C).
     * glPolygonOffset's terms: `slope` multiplies the primitive's maximum
     * depth slope and `units` the smallest resolvable difference of a depth
     * buffer of `depth_bits` bits (16 for Z16, 24 for Z24S8), so a backend
     * whose depth format is another converts the units. Both are 0 when the
     * offset is off. Left NULL, every draw is drawn without it. */
    void (*set_depth_bias)(void* user, float slope, float units, u32 depth_bits);
    /* One draw. `indices` is NULL for a non-indexed one. Everything the
     * hardware has and the host does not -- quads, quad strips, fans,
     * polygons -- has already been expanded into a triangle list by the time
     * it gets here, so the topology is only ever one a host API has. */
    void (*draw)(void* user, rsx_topology topology, const void* vertices,
                 u32 vertex_count, u32 stride, const u32* indices,
                 u32 index_count);
    /* Optional: vertex and index data the engine keeps across frames.
     * buffer_wrap takes ownership of `data` -- page-aligned, `bytes` a whole
     * number of pages -- and frees it once the buffer has been released and
     * the GPU is done with it; on failure (0) the caller still owns it.
     * draw_buffer is draw() with the vertices, and the indices when
     * index_count is nonzero, read from such a buffer at byte offsets. */
    u32  (*buffer_wrap)(void* user, void* data, u32 bytes);
    void (*buffer_release)(void* user, u32 buffer);
    void (*draw_buffer)(void* user, rsx_topology topology, u32 buffer,
                        u32 vb_off, u32 vertex_count, u32 stride,
                        u32 ib_off, u32 index_count);

    /* Clears, which are ordered against draws rather than folded into a pass:
     * a title draws the world, clears depth, and draws the weapon over it. */
    void (*clear_color)(void* user, u32 surface, const float rgba[4]);
    void (*clear_depth_stencil)(void* user, u32 depth, u32 flags,
                                float depth_value, u8 stencil);
    /* The same, limited to a rectangle in target pixels (top-left origin),
     * w and h never 0: the RSX's CLEAR_SURFACE honours the scissor. A title
     * that packs several shadow maps into one depth target clears each
     * region on its own; a split-screen title renders the left view, then
     * clears and renders the right one in the same target with the scissor
     * on that half. Each optional, and asked only for a clear the scissor
     * leaves part of the target out of; without one the engine clears that
     * kind of target whole. */
    void (*clear_depth_stencil_rect)(void* user, u32 depth, u32 flags,
                                     float depth_value, u8 stencil,
                                     u32 x, u32 y, u32 w, u32 h);
    void (*clear_color_rect)(void* user, u32 surface, const float rgba[4],
                             u32 x, u32 y, u32 w, u32 h);

    /* Present the named surface, and read a rectangle of one back. The rows
     * come back in the format the target was created with, so R,G,B,A for an
     * ordinary colour surface. */
    void (*present)(void* user, u32 surface);
    void (*readback)(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h,
                     void* out, u32 out_pitch);

    /* Optional. 3D textures (SET_TEXTURE_FORMAT dimension 3, CONTROL3 depth):
     * a volume of `depth` slices per level, each slice one upload of decoded
     * rows. A backend that leaves these NULL gets the first slice as a 2D
     * texture through texture_create/texture_upload, as before. */
    u32  (*texture_create_3d)(void* user, rsx_be_format fmt, u32 w, u32 h,
                              u32 depth, u32 mips, u32 remap, u32 rsx_fmt);
    void (*texture_upload_3d)(void* user, u32 texture, u32 z, u32 mip,
                              u32 w, u32 h, const void* src, u32 row_bytes);
    /* Nonzero when texture_upload is ordered after every draw already handed
     * to the backend, so rewriting a texture those draws sample is safe. The
     * engine then refreshes a changed 2D texture in place instead of creating
     * a new one and releasing the old (an allocation, a free and a wait for
     * the GPU each time). Zero keeps the create-and-swap. */
    int texture_upload_ordered;
    /* Optional. A depth target's top-left w x h read as a texture of base
     * format `rsx_fmt` reads the bytes the zeta holds: a fresh texture (a
     * colour target for surface_view) whose texel is what the zeta's bytes,
     * stored in guest memory in its own layout, decode to.
     *  - A8R8G8B8 (0x85) of a Z24S8 zeta: R8G8B8A8, R = depth bits 15..8,
     *    G = bits 7..0, B = 0 (the stencil byte), A = bits 23..16, with
     *    round(depth * 0xFFFFFF) taken exactly;
     *  - DEPTH16 (0x92) of a Z16 zeta: R16 unorm of round(depth * 0xFFFF).
     * Any other format: 0. Made in command order, like
     * depth_snapshot, so no wait, and from the depth the draws so far left,
     * which may be the target the current pass still attaches; the engine
     * owns the handle. 0 when the backend cannot. This engine does not ask
     * for it: a D24S8 zeta read as A8R8G8B8 is depth_snapshot_rgba8's copy,
     * and other formats read guest memory. Kept for the backends that have
     * it. */
    u32  (*depth_as_color)(void* user, u32 depth, u32 w, u32 h, u32 rsx_fmt);
    /* Nonzero when the backend's HDR plane (a title's own composite that
     * names rsx_hdr_plane, at its second target) is the packed one: R32G32
     * uint, x the HDR colour as RGB9E5, y the 8-bit colour that went to the
     * frame (bit 24: written). The engine puts "#define RSX_HDR_PLANE_PACKED 1"
     * above such a shader's text; zero leaves the RGBA16F plane (rgb what
     * went to the frame, a how many times brighter it was). */
    int hdr_plane_packed;
} rsx_draw_backend;

/* ---- the engine --------------------------------------------------------- */

/* Register the backend the engine drives. Passing NULL unregisters it. */
void rsx_draw_engine_set_backend(const rsx_draw_backend* backend);

/* Is the engine the path this run takes? True when PS3RECOMP_RSX_ENGINE names
 * "dispatch" (or the backend asked for it by default) and a backend has been
 * registered. Read once and cached, so it is cheap in the FIFO walker. */
int  rsx_draw_engine_enabled(void);

/* Called by a backend that wants the engine to be the default path with no
 * environment set. PS3RECOMP_RSX_ENGINE=vtable still turns it off. */
void rsx_draw_engine_set_default(int on);

/* A runner with its own RSX IO map supplies its address resolver before init.
 * NULL restores the toolkit's cellGcm mapping. */
/* A host that serves guest memory through the reader can stand in for the
 * page write-watch too (the replay does, from its page records): the stamp
 * for [offset, offset+len) at that location changes whenever any byte of it
 * has been written since, and *unwatched is the number of pages it cannot
 * vouch for (a caller with any hashes as before). */
typedef u64 (*rsx_guest_watch_fn)(void* user, u32 location, u32 offset, u32 len, u32* unwatched);
void rsx_draw_engine_set_guest_watch(rsx_guest_watch_fn fn);
void rsx_draw_engine_set_guest_memory(rsx_vertex_guest_ptr_fn reader, void* user);

/* Counts the uploads that are a decoded video's planes (8-bit, at least 256
 * wide): it moves while a movie plays. */
extern volatile unsigned g_rsx_video_plane_uploads;

/* Where the presented frame goes in the window. The title's frame is 16:9; a
 * host that has the title box its view to another ratio says so, for the
 * frames that have it (rsx_present_set_content_aspect; 0, or 16:9: the whole
 * frame, as while a movie plays over it), and that view is shown without the
 * bars. A host also names the shape of the screen
 * the player has (rsx_present_set_screen_aspect; 0: the window's own): the
 * frame is shown in a rectangle of that shape in the middle of the window,
 * the view filling it, anything 16:9 -- the whole frame -- as big as it
 * takes in its middle. So a 16:9 window set to 21:9 shows what a 21:9 one
 * would, with bars above and below. Whatever the shapes, what is shown keeps
 * its proportions, with black bars where they differ (a frame used to be
 * stretched over the window). A frame with a movie's picture in it (a quad
 * sampling its 8-bit planes) is shown whole. RSX_PRESENT_LAYOUT_LOG=1 logs
 * each change of what is shown. Any thread. */
void rsx_present_set_content_aspect(float aspect);
void rsx_present_set_screen_aspect(float aspect);

/* Draws a host marks to be moved on the target, with nothing else their
 * programs compute changed: a draw whose transform constants hold the bit
 * pattern mark (in any word; 0 turns it off) has its window coordinates
 * scaled by k and moved by (x, y), in the target's pixels. For a canvas whose
 * shaders read their own screen position (a material masking itself at a
 * place on the screen): the host draws it where it was made to be, and the
 * draw lands where the host wants it, the shaders none the wiser. Until the
 * next flip, while the word holding the mark is not written again. Any
 * thread. */
void rsx_canvas_remap_set(u32 mark, float k, float x, float y);

/* The window's width over its height, as the last present found it; 0 before
 * the first. Any thread. */
float rsx_present_window_aspect(void);

/* A host overlay over each frame presented to the window: the port's own UI
 * (a trophy pop-up, a menu) drawn by the port with the backend's own objects,
 * after the frame is in the window and before it is presented. The callback
 * runs on the thread that presents, records into the frame's command list
 * with the window's target bound, and may change any of the list's state (it
 * is the frame's last work; the backend sets its own again). A headless
 * present has none. RSX_OVERLAY=0 leaves it out. Any thread sets it; NULL
 * removes it. Backends: Direct3D 12 (more to come). */
enum { RSX_OVERLAY_D3D12 = 1, RSX_OVERLAY_VULKAN = 2, RSX_OVERLAY_METAL = 3 };
typedef struct rsx_overlay_frame {
    int  api;                /* RSX_OVERLAY_* */
    u32  width, height;      /* the target: the whole window, in pixels */
    u32  frames_in_flight;   /* frames the GPU may still be reading */
    void* device;            /* D3D12: ID3D12Device* */
    void* queue;             /* D3D12: ID3D12CommandQueue* */
    void* list;              /* D3D12: ID3D12GraphicsCommandList*, recording */
    u32  format;             /* D3D12: the target's DXGI_FORMAT */
    u64  target;             /* D3D12: the target's RTV (D3D12_CPU_DESCRIPTOR_HANDLE.ptr) */
} rsx_overlay_frame;
typedef void (*rsx_overlay_fn)(const rsx_overlay_frame* f, void* user);
void rsx_overlay_set(rsx_overlay_fn fn, void* user);
/* The overlay, or NULL (also when RSX_OVERLAY=0). For the backends. */
rsx_overlay_fn rsx_overlay_get(void** user);

/* For a backend's present of a fw x fh frame into a ww x wh window: vp, the
 * rectangle (x, y, w, h, window pixels) the whole frame is drawn to -- taller
 * than the window when the bars are cut away -- and rect, the part of the
 * window it is shown in; the rest of the window is black. Returns 1 when that
 * part is not the whole window (the backend clears the window first). The
 * viewport is not the window's whenever the bars are cut away, whatever it
 * returns. */
int rsx_present_layout(u32 fw, u32 fh, u32 ww, u32 wh, float vp[4], int rect[4]);

/* The draws of the last frame presented. */
extern volatile unsigned g_rsx_frame_draws;

/* A port's stand-in for one of the title's textures. The engine asks at every
 * upload of a 2D texture, with the guest's description of it and its bytes
 * (the whole mip chain, `span` bytes); a nonzero return hands back another
 * texture in a guest format, which is decoded and uploaded in its place. Its
 * size may differ where the title samples by normalised coordinates: a
 * larger stand-in is then only sharper. The bytes must stay valid while the
 * texture may be asked about again, and one answer is told from another by
 * its `bytes` pointer. Called on the FIFO walker's thread. */
typedef struct rsx_texture_standin {
    const u8* bytes;
    u32 fmt;               /* CELL_GCM_TEXTURE_* with its layout flags */
    u32 w, h, levels, pitch;
} rsx_texture_standin;
typedef int (*rsx_texture_standin_fn)(void* user, u32 fmt, u32 w, u32 h, u32 levels,
                                      const u8* bytes, u32 span,
                                      rsx_texture_standin* out);
void rsx_draw_engine_set_texture_standin(rsx_texture_standin_fn fn, void* user);
/* The port's answers changed (another set of button glyphs, say): every
 * cached texture is asked again at its next use. Any thread. */
void rsx_draw_engine_texture_standins_changed(void);

/* A backend's occlusion-query result: the samples that passed for the query
 * that GET_REPORT named report_index. Forwarded to the GCM report table. */
void rsx_draw_engine_query_result(u32 report_index, u64 count);
/* Seed the dispatcher from captured state (tools/rsx_replay). Word counts. */
void rsx_draw_engine_seed_state(const u32* regs, u32 nregs, const u32* vp, u32 nvp,
                                const u32* constants, u32 nconst_words);

int  rsx_draw_engine_init(u32 width, u32 height);
void rsx_draw_engine_shutdown(void);

/* One FIFO method, with its subchannel bits still on: the engine masks them
 * out itself, as rsx_live_draw_method (6902-6976) does, because a title whose
 * SPU-built command lists bind NV4097 to another subchannel would otherwise
 * write its state into the wrong register bank. */
void rsx_draw_engine_method(u32 method, u32 arg);
/* The FIFO walker's bulk path: an incrementing run of `count` words from
 * `method` (subchannel bits allowed), big-endian at `be`, wholly inside the
 * transform constant (0x1F00) or program (0x0B80) upload window --
 * rsx_draw_engine_upload_window says whether it is. The same as calling
 * rsx_draw_engine_method on each word. */
int  rsx_draw_engine_upload_window(u32 method, u32 count);
void rsx_draw_engine_upload_run(u32 method, u32 count, const u8* be);

/* cellGcmSetDisplayBuffer, so the flip can resolve a buffer id to a surface
 * the engine has actually rendered rather than to whatever is bound. */
void rsx_draw_engine_set_display_buffer(u32 buffer_id, u32 location, u32 offset,
                                        u32 pitch, u32 width, u32 height);

/* The guest's SET_REFERENCE sync point: submit and wait, so a title spinning
 * on its fence sees the GPU catch up. */
void rsx_draw_engine_flush(void);
/* Finish the GPU work behind any occlusion reports handed to the backend and
 * write their counts into the report area now (no-op when none are out). */
void rsx_draw_engine_sync_queries(void);

/* Present whatever the frame has recorded, for a host that drives the flip
 * itself rather than through the FIFO's 0xE944. */
void rsx_draw_engine_present(void);
/* A runner retiring a queued flip names the buffer explicitly. */
void rsx_draw_engine_present_buffer(u32 buffer_id);
/* A flip the GCM drain decoded from the FIFO (a 0xFEADxxxx word): presented
 * here, in order with the draws, exactly as a 0xE944 flip is. Returns 1 when
 * the engine presented; from then on the host clock's presents are ignored. */
int rsx_draw_engine_fifo_flip(u32 buffer_id);

/* --- test hooks ---------------------------------------------------------- */

/* Draws that ran the guest's OWN programs in the last presented frame. A draw
 * through the built-in pair is not one of them. */
u32  rsx_draw_engine_guest_draws(void);

/* The centre pixel of the surface the last flip presented, as A8R8G8B8 --
 * the convention the RSX clear colour arrives in, so a host can compare the
 * two. Zero when nothing has been presented or the surface is not an
 * 8-bit-per-channel one. */
u32  rsx_draw_engine_readback_center(void);

/* ---- pieces the engine exposes for its own tests ------------------------ */
/*
 * These are the neutral parts the unit tests in libs/video/tests exercise
 * without a GPU: the surface set's keying, the texture cache's revalidation
 * and eviction, the pipeline key's stability, and restart-cut expansion.
 */

/* Decode the NV4097 render state out of the register file. */
void rsx_draw_engine_decode_render_state(const rsx_dispatch* rsx,
                                         rsx_be_render_state* out);

/* Fold the structural render state into a pipeline key. The alpha REFERENCE
 * and everything else that lives in a constant buffer are deliberately left
 * out, so a title changing constants per draw does not multiply pipelines --
 * rsx_live_draw.c's ld_hash_structural_render_state (3148-3187). */
u64  rsx_draw_engine_hash_render_state(const rsx_be_render_state* rs, u64 hash);

/* How many indices the topology expansion produces for `source_refs`
 * references cut at `cuts`, and the indices themselves. `indices` must have
 * room for the count. Triangle strips alternate winding per segment; a quad
 * strip pairs its vertices from each segment's own start, and a polygon is a
 * fan around each segment's own first vertex. */
u32  rsx_draw_engine_topology_index_count(u32 primitive, u32 source_refs,
                                          const u32* cuts, u32 cut_count);
void rsx_draw_engine_write_topology_indices(u32 primitive, u32 source_refs,
                                            const u32* cuts, u32 cut_count,
                                            const u32* occurrence_to_unique,
                                            u32* indices);

#ifdef __cplusplus
}
#endif
#endif /* PS3RECOMP_RSX_DRAW_ENGINE_H */
