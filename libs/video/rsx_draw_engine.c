/*
 * ps3recomp - platform-neutral RSX draw engine (see rsx_draw_engine.h)
 *
 * The orchestration carried across from libs/video/rsx_live_draw.c, the
 * NV4097 -> D3D12 engine a title has shipped on. Line numbers in the comments
 * below are that file's, so the two can be read side by side; the file itself
 * is vendored and stays exactly as it is.
 *
 * What is deliberately left out, because it is not renderer behaviour a title
 * needs: the shader disk cache, movie mode and its compositor, the a010
 * probe, and every YZ_PERF_PROFILE block.
 */
#include "rsx_draw_engine.h"
#include "rsx_capture.h"

#include "rsx_fp_decompiler.h"
#include "rsx_primitives.h"
#include "rsx_restart_cuts.h"
#include "rsx_vertex_formats.h"
#include <math.h>
#include <time.h>
#include "rsx_vp_decompiler.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string.h>

/* Environment switches are read where they are used, some of them per call:
 * answer from the pointer-keyed cache (ps3emu/env_cache.h). */
#include "ps3emu/vm_watch.h"
#include "ps3emu/env_cache.h"
#define getenv(name) ps3_env(name)

/* Guest memory and the RSX offset resolvers, declared the way every backend
 * that reads guest data declares them (rsx_vertex_fetch.h). */
extern u8* vm_base;
u32 cellGcmResolveLocated(int local, u32 offset);
/* Non-zero only when the offset's page is in the IO table, so a caller can
 * tell IO-mapped main memory from "assume VRAM". */
u32 cellGcmResolveIO(u32 offset);
/* The guest VM's OOB guard: the size a host backed, or 0 for "the whole 32-bit
 * space is backed and no check is needed" (runtime/memory/vm.h). */
extern u32 ppu_vm_size;

/* ------------------------------------------------------------------------- */

#define ENG_MAX_SURFACES   64
#define ENG_MAX_ZDEPTHS    64
#define ENG_MAX_TEXTURES   RSX_DRAW_ENGINE_TEXTURE_CACHE
#define ENG_MAX_VTEXTURES  64
#define ENG_MAX_PIPELINES  8192
#define ENG_MAX_BATCHES    256
#define ENG_VERT_STRIDE    (RSX_DSP_NUM_VERTEX_ATTR * 16u)
#define ENG_INVALID        0xFFFFFFFFu
/* Guest textures larger than this are a misdecoded register, not an image;
 * rsx_live_draw.c rejects the same 4096 bound (2033). */
#define ENG_MAX_TEX_DIM    4096u
/* A surface declaration beyond this is a guest pointer read as clip
 * dimensions. rsx_live_draw.c's surface-guard (2625) preserves the existing
 * target rather than destroying a live one. */
#define ENG_MAX_SURFACE_DIM 8192u

#define ENG_VP_CB_BYTES    ((RSX_DSP_NUM_CONSTANTS + 2) * 16u)
#define ENG_FP_MAX_BYTES   0x10000u

#define M_ALPHA_TEST_ENABLE   0x0304
#define M_ALPHA_FUNC          0x0308
#define M_ALPHA_REF           0x030C
#define M_BLEND_ENABLE        0x0310
#define M_BLEND_SFACTOR       0x0314
#define M_BLEND_DFACTOR       0x0318
#define M_BLEND_EQUATION      0x0320
#define M_COLOR_MASK          0x0324
#define M_STENCIL_TEST_ENABLE 0x0328
#define M_STENCIL_MASK        0x032C
#define M_STENCIL_FUNC        0x0330
#define M_STENCIL_FUNC_REF    0x0334
#define M_STENCIL_FUNC_MASK   0x0338
#define M_STENCIL_OP_FAIL     0x033C
#define M_STENCIL_OP_ZFAIL    0x0340
#define M_STENCIL_OP_ZPASS    0x0344
#define M_TWO_SIDED_STENCIL   0x0348
#define M_BACK_STENCIL_FUNC   0x0350
#define M_BACK_STENCIL_OP_FAIL  0x035C
#define M_BACK_STENCIL_OP_ZFAIL 0x0360
#define M_BACK_STENCIL_OP_ZPASS 0x0364
#define M_SCISSOR_HORIZONTAL  0x08C0
#define M_SCISSOR_VERTICAL    0x08C4
#define M_DEPTH_FUNC          0x0A6C
#define M_DEPTH_WRITE         0x0A70
#define M_DEPTH_TEST_ENABLE   0x0A74
#define M_ZSTENCIL_CLEAR      0x1D8C

/* ---- module state -------------------------------------------------------- */

typedef struct {
    u32 location, offset;
    u32 w, h;
    rsx_be_format fmt;
    u32 handle;
} eng_surface;

typedef struct {
    u32 location, offset;
    u32 w, h;
    u32 handle;
    u32 snapshot;          /* sampleable copy, or 0                        */
    int cleared;
    int had_write;
    int snapshot_valid;
    u32 packed;            /* RGBA8 copy as the D24S8 bytes read as colour */
    int packed_valid;
    u32 snap_w, snap_h;    /* the region each copy covers (see below)      */
    u32 packed_w, packed_h;
} eng_zdepth;

typedef struct {
    u32 location, offset, format, width, height, pitch, remap, cubemap;
    u32 handle;
    u64 content_hash;
    u32 last_hash_frame;
    u64 last_use_serial;
    u64 watch_stamp;       /* vm_watch_arm's stamp when the bytes were last hashed */
    u32 watched;           /* every page under it was protected then              */
} eng_texture;

struct eng_pipe_job;
typedef struct { u64 key; u32 handle; u8 fixed; u64 vp_fnv; u64 vp_hlsl; u64 fp_hlsl;
                 u8 pending; struct eng_pipe_job* job; } eng_pipeline;

typedef struct { u32 first, count; } eng_batch;

typedef struct {
    u32 location, offset, pitch, width, height;
    int valid;
} eng_display_buffer;

static struct {
    const rsx_draw_backend* be;
    int ready;
    int default_on;
    u32 width, height;

    rsx_dispatch rsx;

    eng_surface surfaces[ENG_MAX_SURFACES];
    u32 n_surfaces;
    eng_zdepth zdepths[ENG_MAX_ZDEPTHS];
    u32 n_zdepths;
    eng_texture textures[ENG_MAX_TEXTURES];
    u32 n_textures;
    eng_pipeline pipelines[ENG_MAX_PIPELINES];
    u32 n_pipelines;
    eng_display_buffer display_buffers[8];

    u32 frames;
    u64 texture_use_serial;
    u32 guest_draws;
    u32 last_guest_draws;
    u32 last_present_surface;
    u32 last_flip_buffer;
    u32 q_cur;               /* open occlusion query (backend counter), or 0 */
    u32 q_attempts, q_draws; /* draws the game issued / the backend counted */
    u32 q_unflushed;         /* reports handed to the backend since its last submit */
    u32 sink_flips;          /* flips decoded from the FIFO (0xE944), in draw order */

    /* staging: decoded texture levels, the constant blocks, the index list */
    u8* tex_staging;
    u32 tex_staging_cap;
    u8* vp_cb;
    u32 vp_cb_gen;           /* rsx.const_gen vp_cb was filled at        */
    int vp_cb_valid;
    int vp_cb_bound;         /* the backend's last staged block is vp_cb */
    u8* fp_cb;
    u32 fp_cb_cap;
    u32* indices;
    u32 index_cap;

    rsx_fp_constant_block fp_constants;
} g;

/* The decompilers' HLSL. The VP decompiler builds bodies up to 192 KB. */
static char s_vs_hlsl[256 * 1024];
static char s_ps_hlsl[256 * 1024];

/* The draw accumulator: one BEGIN_END group's batches, references and cuts. */
static struct {
    eng_batch arr[ENG_MAX_BATCHES];
    u32 n_arr;
    eng_batch idx[ENG_MAX_BATCHES];
    u32 n_idx;
    u32 n_packets;

    rsx_vertex_ref* refs;
    u32 n_refs, cap_refs, n_source_refs;
    int refs_remapped;
    rsx_vertex_remap ref_remap;

    u32* cuts;
    u32 n_cuts, cap_cuts;

    u8* verts;
    u64 verts_cap;
    u32 n_verts;
    const u8* out_verts;     /* what the draw uploads: verts, or a cached copy */

    rsx_vertex_layout_plan layout;
    rsx_vertex_fetch_plan fetch_plan;
    int fetch_ok;

    /* INLINE_ARRAY stream for this group, when the vertices came down the
     * FIFO rather than from a vertex array. */
    const u8* inl;
    u32 inl_bytes;
} dc;

/* ---- guest memory -------------------------------------------------------- */

/* rsx_dsp_* locations are RSX_LOCATION_LOCAL = 0 / _MAIN = 1;
 * cellGcmResolveLocated's argument is the opposite sense (1 = local), and
 * getting that backwards resolves every VRAM object through the IO table. */
static rsx_vertex_guest_ptr_fn s_guest_reader;
static void* s_guest_user;

void rsx_draw_engine_set_guest_memory(rsx_vertex_guest_ptr_fn reader, void* user)
{
    s_guest_reader = reader;
    s_guest_user = user;
}

static const u8* eng_guest_ptr(void* user, u32 location, u32 offset,
                               u32 min_bytes)
{
    (void)user;
    if (!min_bytes || min_bytes > (256u << 20)) return NULL;
    if (s_guest_reader) return s_guest_reader(s_guest_user, location, offset, min_bytes);
    if (!vm_base) return NULL;
    const u32 ea = cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset);
    if (!ea) return NULL;
    if (ppu_vm_size && (u64)ea + min_bytes > ppu_vm_size) return NULL;
    if (g_rsx_capture_on) rsx_capture_read(location, offset, min_bytes);
    return vm_base + ea;
}

/* tools/rsx_replay: start from a captured register file, transform program
 * and constants, as rsx_dispatch_seed_* lay them in. */
void rsx_draw_engine_seed_state(const u32* regs, u32 nregs, const u32* vp, u32 nvp,
                                const u32* constants, u32 nconst_words)
{
    if (regs) rsx_dispatch_seed_registers(&g.rsx, regs, nregs);
    if (vp) rsx_dispatch_seed_transform_program(&g.rsx, vp, nvp);
    if (constants) rsx_dispatch_seed_transform_constants(&g.rsx, constants, nconst_words);
}

static u64 eng_fnv1a(const void* data, u32 n, u64 hash)
{
    const u8* p = (const u8*)data;
    for (u32 i = 0; i < n; i++) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
}

/* ---- render state -------------------------------------------------------- */

void rsx_draw_engine_decode_render_state(const rsx_dispatch* rsx,
                                         rsx_be_render_state* rs)
{
    memset(rs, 0, sizeof(*rs));
    rs->alpha_test_enable = rsx_dsp_reg(rsx, M_ALPHA_TEST_ENABLE) & 1;
    rs->alpha_func   = rsx_dsp_reg(rsx, M_ALPHA_FUNC);
    rs->alpha_ref_raw = rsx_dsp_reg(rsx, M_ALPHA_REF);
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(rsx, &sf);
    rs->alpha_ref_format = sf.color_format;
    rs->rt_fp16 = sf.color_format == RSX_SURFACE_FMT_F_W16Z16Y16X16;
    rs->blend_enable = rsx_dsp_reg(rsx, M_BLEND_ENABLE) & 1;
    const u32 sfac = rsx_dsp_reg(rsx, M_BLEND_SFACTOR);
    const u32 dfac = rsx_dsp_reg(rsx, M_BLEND_DFACTOR);
    const u32 eq   = rsx_dsp_reg(rsx, M_BLEND_EQUATION);
    rs->sf_rgb = sfac & 0xFFFF; rs->sf_a = sfac >> 16;
    rs->df_rgb = dfac & 0xFFFF; rs->df_a = dfac >> 16;
    rs->eq_rgb = eq & 0xFFFF;   rs->eq_a = eq >> 16;
    rs->depth_test  = rsx_dsp_reg(rsx, M_DEPTH_TEST_ENABLE) & 1;
    rs->depth_write = rsx_dsp_reg(rsx, M_DEPTH_WRITE) & 1;
    rs->depth_func  = rsx_dsp_reg(rsx, M_DEPTH_FUNC);
    rs->cull_enable = rsx_dsp_reg(rsx, 0x183C) & 1;
    rs->cull_face   = rsx_dsp_reg(rsx, 0x1830);
    rs->front_face  = rsx_dsp_reg(rsx, 0x1834);
    /* The RAW register: a game-written 0 is a real "write no colour channel"
     * (a depth-prime pass), and rsx_dispatch_init seeds the nv40 reset value
     * so never-written reads as all-on. */
    rs->color_mask  = rsx_dsp_reg(rsx, M_COLOR_MASK);
    rs->stencil_enable    = rsx_dsp_reg(rsx, M_STENCIL_TEST_ENABLE) & 1;
    rs->stencil_two_sided = rsx_dsp_reg(rsx, M_TWO_SIDED_STENCIL) & 1;
    rs->s_func       = rsx_dsp_reg(rsx, M_STENCIL_FUNC);
    rs->s_func_mask  = rsx_dsp_reg(rsx, M_STENCIL_FUNC_MASK) & 0xFF;
    rs->s_write_mask = rsx_dsp_reg(rsx, M_STENCIL_MASK) & 0xFF;
    rs->s_fail       = rsx_dsp_reg(rsx, M_STENCIL_OP_FAIL);
    rs->s_zfail      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZFAIL);
    rs->s_zpass      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZPASS);
    rs->bs_func      = rsx_dsp_reg(rsx, M_BACK_STENCIL_FUNC);
    rs->bs_fail      = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_FAIL);
    rs->bs_zfail     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZFAIL);
    rs->bs_zpass     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZPASS);
    /* RSX_FORCE=<list>: experiment overrides. "nostencil" disables the
     * stencil test, "alpharef0" zeroes the alpha-test reference, "noalpha"
     * disables the alpha test. */
    { static int f = -1, ns = 0, ar = 0, na = 0;
      if (f < 0) { const char* e = getenv("RSX_FORCE"); f = e ? 1 : 0;
          if (e) { ns = strstr(e, "nostencil") != NULL; ar = strstr(e, "alpharef0") != NULL; na = strstr(e, "noalpha") != NULL; } }
      if (f) { if (ns) rs->stencil_enable = 0; if (ar) rs->alpha_ref_raw = 0; if (na) rs->alpha_test_enable = 0; } }
}

u64 rsx_draw_engine_hash_render_state(const rsx_be_render_state* rs, u64 hash)
{
#define ENG_HASH_FIELD(name) hash = eng_fnv1a(&rs->name, sizeof(rs->name), hash)
    /* The alpha REFERENCE is deliberately absent: it lives in the fragment
     * constant block, so a title animating a fade must not compile a new
     * pipeline per frame. Enable and compare mode still select a variant.
     * Every field is listed by name so struct padding can never be identity
     * (rsx_live_draw.c:3151-3185). */
    ENG_HASH_FIELD(alpha_test_enable);
    ENG_HASH_FIELD(alpha_func);
    ENG_HASH_FIELD(blend_enable);
    ENG_HASH_FIELD(sf_rgb);
    ENG_HASH_FIELD(df_rgb);
    ENG_HASH_FIELD(sf_a);
    ENG_HASH_FIELD(df_a);
    ENG_HASH_FIELD(eq_rgb);
    ENG_HASH_FIELD(eq_a);
    ENG_HASH_FIELD(depth_test);
    ENG_HASH_FIELD(depth_write);
    ENG_HASH_FIELD(depth_func);
    ENG_HASH_FIELD(cull_enable);
    ENG_HASH_FIELD(cull_face);
    ENG_HASH_FIELD(front_face);
    ENG_HASH_FIELD(color_mask);
    ENG_HASH_FIELD(rt_fp16);
    ENG_HASH_FIELD(stencil_enable);
    ENG_HASH_FIELD(stencil_two_sided);
    ENG_HASH_FIELD(s_func);
    ENG_HASH_FIELD(s_func_mask);
    ENG_HASH_FIELD(s_write_mask);
    ENG_HASH_FIELD(s_fail);
    ENG_HASH_FIELD(s_zfail);
    ENG_HASH_FIELD(s_zpass);
    ENG_HASH_FIELD(bs_func);
    ENG_HASH_FIELD(bs_fail);
    ENG_HASH_FIELD(bs_zfail);
    ENG_HASH_FIELD(bs_zpass);
#undef ENG_HASH_FIELD
    return hash;
}

/* ---- topology expansion -------------------------------------------------- */

/* Every expansion is bounded by the restart cuts, so a strip that the guest
 * broke with the sentinel index gets no connecting triangle across the break.
 * Missing that decode is what upstream's aa7fb63 records as "the exploded
 * spiky mesh that occluded the scene". */
u32 rsx_draw_engine_topology_index_count(u32 primitive, u32 source_refs,
                                         const u32* cuts, u32 cut_count)
{
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES:
        return source_refs - source_refs % 3u;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON: {
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 3) total += (count - 2) * 3;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUAD_STRIP: {
        /* Two vertices per quad after the first pair, so a segment of n
         * carries (n - 2) / 2 quads and an odd trailing vertex is dropped. */
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 4) total += ((count - 2u) / 2u) * 6u;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUADS:
        return (source_refs / 4u) * 6u;
    default:
        return 0;
    }
}

static u32 eng_topology_vertex(const u32* occurrence_to_unique, u32 occurrence)
{
    return occurrence_to_unique ? occurrence_to_unique[occurrence] : occurrence;
}

void rsx_draw_engine_write_topology_indices(u32 primitive, u32 source_refs,
                                            const u32* cuts, u32 cut_count,
                                            const u32* occurrence_to_unique,
                                            u32* indices)
{
    u32 write = 0;
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES: {
        const u32 count = source_refs - source_refs % 3u;
        for (u32 o = 0; o < count; o++)
            indices[write++] = eng_topology_vertex(occurrence_to_unique, o);
        break;
    }
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            /* Odd triangles swap their first two vertices, so the whole strip
             * keeps one winding. */
            for (u32 i = 0; i + 2 < count; i++) {
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 1u - (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 2u);
            }
        }
        break;
    /* A polygon is a convex fan around its own first vertex, which is what
     * both of this tree's other renderers make of one. */
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            for (u32 i = 1; i + 1 < count; i++) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i + 1u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUAD_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 4) continue;
            /* Every quad is split along the diagonal between the two pairs,
             * so each one keeps the strip's winding without alternating the
             * way a triangle strip has to; a cut simply starts the pairing
             * again from the segment's own first vertex. This is the
             * expansion the vtable path's emit_vertices performs. */
            for (u32 q = 0; q + 3 < count; q += 2) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 3u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUADS:
        for (u32 q = 0; q < source_refs / 4u; q++) {
            const u32 base = q * 4u;
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 1u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 3u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
        }
        break;
    default:
        break;
    }
}

/* Does this primitive become an indexed triangle list, and is the index buffer
 * needed at all? rsx_vertex_topology_plan answers that for the reference
 * engine, whose own expansion stops at quads. Quad strips and polygons are
 * this engine's, so their two cases are decided here rather than in the
 * shared helper, which the vendored rsx_live_draw.c also calls. */
static int eng_topology_rebuild(u32 primitive, int refs_remapped, int* indexed)
{
    if (primitive == RSX_PRIMITIVE_QUAD_STRIP ||
        primitive == RSX_PRIMITIVE_POLYGON) {
        (void)refs_remapped;
        *indexed = 1;
        return 1;
    }
    return rsx_vertex_topology_plan(primitive, refs_remapped, indexed);
}

/* ---- surfaces ------------------------------------------------------------ */

static rsx_be_format eng_surface_format(u32 color_format)
{
    switch (color_format & 0x1Fu) {
    case RSX_SURFACE_FMT_F_W16Z16Y16X16: return RSX_BE_FMT_R16G16B16A16F;
    case 0x0C: return RSX_BE_FMT_R32G32B32A32F;
    case 0x0D: return RSX_BE_FMT_R32F;
    default:   return RSX_BE_FMT_R8G8B8A8;
    }
}

/* Decode the guest's own bytes behind a colour surface into host R,G,B,A rows
 * so a freshly created target starts with what the title CPU-initialised it
 * to. Returns the staging pointer, or NULL when the bytes are unreadable or
 * the format is not one the decoder handles. */
static const void* eng_surface_seed(u32 location, u32 offset, u32 w, u32 h,
                                    rsx_be_format fmt, u32* out_row_bytes)
{
    *out_row_bytes = 0;
    if (fmt != RSX_BE_FMT_R8G8B8A8) return NULL;
    /* Only a surface in IO-mapped MAIN memory is seeded. RSX local memory is
     * reserved and not backed by this tree's guest VM -- runtime/memory/vm.h
     * maps the whole space PROT_NONE and commits main memory and the stacks,
     * leaving a runner to commit what its own title needs -- so reading a VRAM
     * surface's backing would fault long before it could help. Both halves are
     * checked: the DMA context says which space the offset is in, and the IO
     * table says the page is really mapped. */
    if (location != RSX_LOCATION_MAIN ||
        (!s_guest_reader && !cellGcmResolveIO(offset))) return NULL;
    rsx_tex_layout tl;
    /* A8R8G8B8 with the LN bit: a surface is linear, never swizzled. */
    rsx_texture_layout(0x85u | 0x20u, w, h, &tl);
    if (!tl.face_bytes || !tl.dst_row_bytes) return NULL;
    const u8* src = eng_guest_ptr(NULL, location, offset, tl.face_bytes);
    if (!src) return NULL;
    const u32 bytes = tl.dst_row_bytes * tl.rows;
    if (g.tex_staging_cap < bytes) {
        u8* n = (u8*)realloc(g.tex_staging, bytes);
        if (!n) return NULL;
        g.tex_staging = n;
        g.tex_staging_cap = bytes;
    }
    rsx_texture_decode(g.tex_staging, tl.dst_row_bytes, src, w, h, &tl,
                       rsx_texture_argb_is_rgba());
    *out_row_bytes = tl.dst_row_bytes;
    return g.tex_staging;
}

/* The colour target for this (location, offset), created or reallocated when
 * the size or format moved. Returns a slot index, or ENG_INVALID.
 * rsx_live_draw.c's surface_get (2605-2690). */
static u32 eng_surface_get(u32 location, u32 offset, u32 want_w, u32 want_h,
                          rsx_be_format want_fmt)
{
    if (!want_w) want_w = g.width;
    if (!want_h) want_h = g.height;

    u32 slot = ENG_MAX_SURFACES;
    for (u32 i = 0; i < g.n_surfaces; i++)
        if (g.surfaces[i].location == location && g.surfaces[i].offset == offset) {
            if (g.surfaces[i].w == want_w && g.surfaces[i].h == want_h &&
                g.surfaces[i].fmt == want_fmt)
                return i;
            slot = i;
            break;
        }

    /* Never destroy a usable render target because one malformed command
     * decoded a guest pointer as clip dimensions. */
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM) {
        static u32 logs = 0;
        if (logs++ < 8)
            fprintf(stderr, "[rsx engine] rejected implausible surface 0x%X %ux%u;"
                            " keeping the %s target\n", offset, want_w, want_h,
                    slot < ENG_MAX_SURFACES ? "existing" : "absent");
        return slot < ENG_MAX_SURFACES ? slot : ENG_INVALID;
    }
    if (slot == ENG_MAX_SURFACES) {
        if (g.n_surfaces >= ENG_MAX_SURFACES) return ENG_INVALID;
        slot = g.n_surfaces;
    }

    u32 seed_row = 0;
    const void* seed = eng_surface_seed(location, offset, want_w, want_h,
                                        want_fmt, &seed_row);
    const u32 handle = g.be->color_target_create(g.be->user, want_fmt,
                                                 want_w, want_h, seed, seed_row);
    if (!handle)
        return (slot < g.n_surfaces && g.surfaces[slot].handle) ? slot : ENG_INVALID;

    eng_surface* s = &g.surfaces[slot];
    if (s->handle) g.be->color_target_release(g.be->user, s->handle);
    s->location = location; s->offset = offset;
    s->w = want_w; s->h = want_h; s->fmt = want_fmt;
    s->handle = handle;
    if (slot == g.n_surfaces) g.n_surfaces++;
    { static u32 logs = 0; if (logs++ < 16)
        fprintf(stderr, "[rsx engine] surface %u:0x%08X %ux%u fmt %d%s\n",
                location, offset, want_w, want_h, (int)want_fmt,
                seed ? " (seeded from guest memory)" : ""); }
    return slot;
}

/* The colour targets SET_SURFACE_COLOR_TARGET names, as slots into
 * g.surfaces, with A first. One for an ordinary draw; an MRT set adds B, C
 * and D, each created at target A's size and format, which is the rule the
 * vtable path's target_for binds by. Resolution stops at the first member
 * that cannot be created, as that path stops at the first gap. Returns how
 * many were resolved, or 0. */
static u32 eng_current_target_set(u32 slots[RSX_BE_MAX_COLOR_TARGETS])
{
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(&g.rsx, &sf);
    /* SET_SURFACE_COLOR_TARGET 2 selects B alone; every other value starts at
     * A, and the MRT selectors add B, C and D on top of it. */
    const u32 sel = (sf.color_target == CELL_GCM_SURFACE_TARGET_1) ? 1u : 0u;
    const rsx_be_format fmt = eng_surface_format(sf.color_format);
    const u32 first = eng_surface_get(sf.color_location[sel], sf.color_offset[sel],
                                      sf.clip_w, sf.clip_h, fmt);
    if (first == ENG_INVALID) return 0;
    slots[0] = first;
    u32 n = 1;

    static const u32 mrt_from[3] = {
        CELL_GCM_SURFACE_TARGET_MRT1, CELL_GCM_SURFACE_TARGET_MRT2,
        CELL_GCM_SURFACE_TARGET_MRT3
    };
    for (u32 i = 0; i < 3; i++) {
        if (sf.color_target < mrt_from[i]) break;
        const u32 slot = eng_surface_get(sf.color_location[i + 1],
                                         sf.color_offset[i + 1],
                                         sf.clip_w, sf.clip_h, fmt);
        if (slot == ENG_INVALID) break;
        /* A set naming one buffer twice would attach the same target twice,
         * which no host API allows. */
        int seen = 0;
        for (u32 k = 0; k < n; k++) if (slots[k] == slot) seen = 1;
        if (seen) break;
        slots[n++] = slot;
    }
    return n;
}

static u32 eng_current_surface(void)
{
    u32 slots[RSX_BE_MAX_COLOR_TARGETS];
    return eng_current_target_set(slots) ? slots[0] : ENG_INVALID;
}

/* ---- per-zeta depth ------------------------------------------------------ */

/* One depth target per guest zeta address. Sharing a single resource across a
 * title's shadow, scene and post passes cross-contaminates later depth tests;
 * upstream's ef3271b records that as "the black player-character mass". */
static u32 eng_zdepth_get(u32 location, u32 offset, u32 rt_w, u32 rt_h)
{
    /* The attachment must cover the whole canvas: an early pass declaring a
     * smaller clip than the live viewport otherwise gets a target the host
     * API rejects when a later, larger pass binds it. */
    u32 want_w = rt_w > g.width ? rt_w : g.width;
    u32 want_h = rt_h > g.height ? rt_h : g.height;
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM)
        return ENG_INVALID;

    u32 slot = ENG_MAX_ZDEPTHS;
    for (u32 i = 0; i < g.n_zdepths; i++) {
        eng_zdepth* z = &g.zdepths[i];
        if (z->location == location && z->offset == offset) {
            if (z->w >= want_w && z->h >= want_h) return i;
            slot = i;
            if (want_w < z->w) want_w = z->w;
            if (want_h < z->h) want_h = z->h;
            break;
        }
    }
    if (slot == ENG_MAX_ZDEPTHS) {
        if (g.n_zdepths >= ENG_MAX_ZDEPTHS) return ENG_INVALID;
        slot = g.n_zdepths;
    }

    const u32 handle = g.be->depth_target_create(g.be->user, want_w, want_h);
    if (!handle)
        return (slot < g.n_zdepths && g.zdepths[slot].handle) ? slot : ENG_INVALID;

    eng_zdepth* z = &g.zdepths[slot];
    if (z->handle) g.be->depth_target_release(g.be->user, z->handle);
    if (z->snapshot && g.be->texture_release) g.be->texture_release(g.be->user, z->snapshot);
    if (z->packed && g.be->texture_release) g.be->texture_release(g.be->user, z->packed);
    z->packed = 0; z->packed_valid = 0;
    z->location = location; z->offset = offset;
    z->w = want_w; z->h = want_h;
    z->handle = handle;
    z->snapshot = 0;
    z->cleared = 0;
    z->had_write = 0;
    z->snapshot_valid = 0;
    if (slot == g.n_zdepths) g.n_zdepths++;
    return slot;
}

/* A depth target read back as a texture, but only once the pass has executed
 * a depth-WRITING draw: a write-enable bit alone does not prove the pass
 * produced a usable depth map, and a clear-only zeta falls through to guest
 * memory instead (rsx_live_draw.c:6016-6018, 6265-6269). */
/* A depth target is allocated at least as large as the screen
 * (eng_zdepth_get), so a 512x512 shadow map lives in the top-left corner of a
 * 1280x720 texture. A copy for sampling must be the size the title's texture
 * unit declares -- the region it addresses with 0..1 coordinates -- or every
 * lookup lands in the wrong texels: Drakengard 3's character shadow
 * projection read its 512x512 map as if it were 1280x720 and found nothing,
 * so no character cast a shadow. The copy is cropped texel for texel. */
static void eng_zdepth_copy_size(const eng_zdepth* z, u32 tw, u32 th, u32* w, u32* h)
{
    *w = (tw && tw < z->w) ? tw : z->w;
    *h = (th && th < z->h) ? th : z->h;
}

static u32 eng_zdepth_snapshot(u32 slot, u32 tw, u32 th)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->handle || !z->had_write) return 0;
    u32 cw, ch; eng_zdepth_copy_size(z, tw, th, &cw, &ch);
    if (z->snapshot_valid && z->snapshot && z->snap_w == cw && z->snap_h == ch) return z->snapshot;
    if (!g.be->depth_snapshot) return 0;
    /* The previous image of this zeta is stale (a clear invalidated it), so
     * hand it back before resolving a new one. It never was: every shadow-map
     * clear leaked one backend texture -- a frame's worth of them per frame --
     * until the backend's 4096-entry object table filled, a few minutes into
     * play, and from then on EVERY new texture failed to create. In
     * Drakengard 3 that is the in-game cutscene sampling three missing video
     * planes (solid green), and the player's body vanishing for good once its
     * textures were next re-uploaded. Draws already recorded against the old
     * snapshot keep it: the backend retires the handle and recycles it only
     * after the frame is encoded. */
    if (z->snapshot && g.be->texture_release) g.be->texture_release(g.be->user, z->snapshot);
    z->snapshot = 0;
    const u32 tex = g.be->depth_snapshot(g.be->user, z->handle, cw, ch);
    if (!tex) return 0;
    z->snapshot = tex;
    z->snapshot_valid = 1;
    z->snap_w = cw; z->snap_h = ch;
    return tex;
}

static u32 eng_zdepth_packed(u32 slot, u32 tw, u32 th)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->handle || !z->had_write) return 0;
    u32 cw, ch; eng_zdepth_copy_size(z, tw, th, &cw, &ch);
    if (z->packed_valid && z->packed && z->packed_w == cw && z->packed_h == ch) return z->packed;
    if (!g.be->depth_snapshot_rgba8) return 0;
    if (z->packed && g.be->texture_release) g.be->texture_release(g.be->user, z->packed);
    z->packed = 0;
    const u32 tex = g.be->depth_snapshot_rgba8(g.be->user, z->handle, cw, ch);
    if (!tex) return 0;
    z->packed = tex;
    z->packed_valid = 1;
    z->packed_w = cw; z->packed_h = ch;
    return tex;
}

/* ---- textures ------------------------------------------------------------ */

static rsx_be_format eng_texfmt(rsx_texfmt f)
{
    switch (f) {
    case RSX_TEXFMT_R8:              return RSX_BE_FMT_R8;
    case RSX_TEXFMT_R8G8:            return RSX_BE_FMT_R8G8;
    case RSX_TEXFMT_R8G8B8A8:        return RSX_BE_FMT_R8G8B8A8;
    case RSX_TEXFMT_BC1:             return RSX_BE_FMT_BC1;
    case RSX_TEXFMT_BC2:             return RSX_BE_FMT_BC2;
    case RSX_TEXFMT_BC3:             return RSX_BE_FMT_BC3;
    case RSX_TEXFMT_R16:             return RSX_BE_FMT_R16;
    case RSX_TEXFMT_R16G16:          return RSX_BE_FMT_R16G16;
    case RSX_TEXFMT_R16G16F:         return RSX_BE_FMT_R16G16F;
    case RSX_TEXFMT_R16G16B16A16F:   return RSX_BE_FMT_R16G16B16A16F;
    case RSX_TEXFMT_R32F:            return RSX_BE_FMT_R32F;
    case RSX_TEXFMT_R32G32B32A32F:   return RSX_BE_FMT_R32G32B32A32F;
    default:                         return RSX_BE_FMT_R8;
    }
}

/* How many guest bytes a texture occupies: the whole mip chain, times six for
 * a cube map. That is what the content hash has to cover, or a title
 * animating one face or one lower level keeps its stale upload. */
static u32 eng_texture_span(u32 fmt, u32 w, u32 h, u32 levels, u32 pitch, int cube)
{
    if (cube) return rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) * 6u;
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 n = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!n) return 0;
    return lv[n - 1].offset + lv[n - 1].tl.face_bytes;
}

/* One hash per cached texture per presented frame. FNV-style over the whole
 * span is deliberately cheap: this is a mutation detector, not a content id
 * (rsx_live_draw.c:2002-2026). Eight independent lanes, folded at the end: a
 * single lane is one dependent 64-bit multiply per 8 bytes, latency-bound at a
 * few GB/s, and it was 13% of the draw engine's time in Drakengard 3's battle
 * areas. The lanes run in parallel and cover every byte as before. */
static u64 eng_texture_content_hash(u32 location, u32 offset, u32 span,
                                    int* readable)
{
    const u8* src = span ? eng_guest_ptr(NULL, location, offset, span) : NULL;
    if (!src) { *readable = 0; return 0; }
    const u64 P = 1099511628211ull;
    u64 h[8];
    for (int k = 0; k < 8; k++) h[k] = 1469598103934665603ull + (u64)k;
    u32 i = 0;
    for (; i + 64 <= span; i += 64) {
        u64 w[8];
        memcpy(w, src + i, sizeof w);
        for (int k = 0; k < 8; k++) h[k] = (h[k] ^ w[k]) * P;
    }
    u64 hash = 1469598103934665603ull;
    for (int k = 0; k < 8; k++) hash = (hash ^ h[k]) * P;
    for (; i < span; i++) { hash ^= src[i]; hash *= P; }
    *readable = 1;
    return hash;
}

static int eng_staging_reserve(u32 bytes)
{
    if (g.tex_staging_cap >= bytes) return 1;
    u8* n = (u8*)realloc(g.tex_staging, bytes);
    if (!n) return 0;
    g.tex_staging = n;
    g.tex_staging_cap = bytes;
    return 1;
}

/* Decode a guest texture out of guest memory and hand every face and level to
 * the backend. Returns the backend handle, or 0. */
/* What the walker did in the current frame, for the stutter line in
 * eng_present (DOD3_STUTTER_MS). */
static struct { u32 tex; u64 tex_bytes; u32 vc_store; u64 vc_bytes; u32 pipes; double pipe_ms;
                u32 vc_evicts; double vc_evict_ms; u32 vc_evicted; u32 draws; double draw_ms; u32 tex_skipped; } s_fstat;
static double eng_now_ms(void)
{
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static u32 eng_texture_upload(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                              u32 levels, u32 pitch, int cube, u32 remap)
{
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 nlv = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!nlv || !lv[0].tl.face_bytes) return 0;
    const u32 faces = cube ? 6u : 1u;
    const u32 face_stride = cube
        ? rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) : 0u;
    const u32 span = eng_texture_span(fmt, w, h, levels, pitch, cube);
    const u8* src = eng_guest_ptr(NULL, location, offset, span);
    if (!src) return 0;
    s_fstat.tex++; s_fstat.tex_bytes += span;

    /* TEX_DUMP_DIR=<dir>: log every upload, and write each 8-bit (B8) upload at
     * least 256 wide -- a Bink video plane -- as a PPM, so "is the decoder
     * producing pixels" can be answered by looking at the image. */
    { static const char* dd = (const char*)1; static int ul = -1;
      if (dd == (const char*)1) dd = getenv("TEX_DUMP_DIR");
      if (ul < 0) ul = getenv("TEX_UP_LOG") ? 1 : 0;   /* the log line alone, no PPMs */
      if (dd || ul) {
          static int n = 0, saved = 0;
          if (n++ < 400 || (fmt & 0x9Fu) == 0x81u /* CELL_GCM_TEXTURE_B8, any layout flags */) {
              /* A B8 plane's mean, so "are the decoder's pixels there" is in
               * the log line itself: a Bink plane of a real picture averages
               * tens to a hundred-odd; all-zero planes shade the YUV->RGB
               * quad solid green (0,135,0). */
              unsigned long b8mean = 0;
              if ((fmt & 0x9Fu) == 0x81u && w && h) {
                  const u32 row = pitch ? pitch : w; unsigned long sum = 0;
                  for (u32 y = 0; y < h; y += 4) for (u32 x = 0; x < w; x += 4) sum += src[y * row + x];
                  b8mean = sum / (((h + 3) / 4) * ((w + 3) / 4)); }
              fprintf(stderr, "[tex-up] f%u fmt=0x%02X %ux%u pitch=%u levels=%u loc=%u off=0x%08X ea=0x%08X remap=0x%04X cube=%d mean=%lu\n",
                      g.frames, fmt, w, h, pitch, levels, location, offset,
                      cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset), remap, cube, b8mean);
          }
          static int b8n = 0;
          /* TEX_DUMP_FROM=<frame>: spend the PPM budget from that frame on. */
          static long dfrom = -1; if (dfrom < 0) { const char* e = getenv("TEX_DUMP_FROM"); dfrom = e ? atol(e) : 0; }
          if (dd && (long)g.frames >= dfrom && (fmt & 0x9Fu) == 0x81u /* CELL_GCM_TEXTURE_B8, any layout flags */ && w >= 256 &&
              (b8n++ % 150) == 0 && saved < 80) {   /* one plane every ~1.5 s of video */
              char path[512]; snprintf(path, sizeof path, "%s/b8_%03d_%ux%u_%08X.ppm", dd, saved++, w, h, offset);
              FILE* f = fopen(path, "wb");
              if (f) { fprintf(f, "P6\n%u %u\n255\n", w, h);
                  const u32 row = pitch ? pitch : w;
                  unsigned long sum = 0;
                  for (u32 y = 0; y < h; y++) for (u32 x = 0; x < w; x++) {
                      u8 v = src[y * row + x]; sum += v; fputc(v, f); fputc(v, f); fputc(v, f); }
                  fclose(f);
                  fprintf(stderr, "[tex-up] saved %s mean=%lu\n", path, sum / ((unsigned long)w * h)); }
          } } }

    const u32 handle = g.be->texture_create(g.be->user, eng_texfmt(lv[0].tl.fmt),
                                            w, h, nlv, faces, remap, fmt);
    if (!handle) return 0;

    for (u32 f = 0; f < faces; f++) {
        const u8* face = src + (size_t)f * face_stride;
        for (u32 m = 0; m < nlv; m++) {
            const rsx_tex_layout* tl = &lv[m].tl;
            if (!eng_staging_reserve(tl->dst_row_bytes * tl->rows)) {
                g.be->texture_release(g.be->user, handle);
                return 0;
            }
            rsx_texture_decode(g.tex_staging, tl->dst_row_bytes,
                               face + lv[m].offset, lv[m].w, lv[m].h, tl,
                               rsx_texture_argb_is_rgba());
            g.be->texture_upload(g.be->user, handle, f, m, lv[m].w, lv[m].h,
                                 g.tex_staging, tl->dst_row_bytes, tl->rows);
        }
    }
    return handle;
}

/* The cache slot for a texture unit's bytes, keyed on where they are and what
 * the registers say they are. An already-cached entry is re-hashed at most
 * once per presented frame and re-decoded when the guest changed it, which is
 * what makes an animated UI or a video texture update at all. A miss with the
 * cache full evicts the least recently used entry rather than returning
 * nothing: returning white "made the recovered orphanage render as flat
 * green/black geometry" (rsx_live_draw.c:2326-2334). */
/* The write-watch (ps3emu/vm_watch.h) in place of a hash per texture per
 * frame: when every page under a texture is protected and none was written
 * since the last hash, the bytes are known unchanged. RSX_TEX_WATCH=0, or a
 * host without it, hashes as before. -1 undecided; a host may switch it. */
/* RSX_AA: fxaa (1) -- FXAA on the presented frame (rsx_present_passes.h);
 * msaa2 / msaa4 / msaa8 (2, 4, 8) -- the 3D passes multisampled (the
 * engines' eng_ms_*). Read on first use; the port's settings page sets it
 * while running. */
int g_rsx_aa = -1;
int rsx_aa_mode(void)
{
    if (g_rsx_aa < 0) {
        const char* e = getenv("RSX_AA");
        g_rsx_aa = 0;
        if (e && (!strcmp(e, "fxaa") || !strcmp(e, "FXAA") || !strcmp(e, "1"))) g_rsx_aa = 1;
        else if (e && !strcmp(e, "msaa2")) g_rsx_aa = 2;
        else if (e && !strcmp(e, "msaa4")) g_rsx_aa = 4;
        else if (e && !strcmp(e, "msaa8")) g_rsx_aa = 8;
    }
    return g_rsx_aa;
}

/* Set (by the port's settings page) when RSX_VSYNC / RSX_DISPLAY /
 * RSX_WINDOW have changed; the backend applies them at its next message
 * pump and clears it. */
volatile int g_rsx_display_reload;
int g_eng_tex_watch = -1;
static unsigned long long s_tex_skips_total;   /* hashes the watch saved, for [frametime] */
static int eng_tex_watch_on(void)
{
    if (g_eng_tex_watch < 0) g_eng_tex_watch = vm_watch_available() && !s_guest_reader ? 1 : 0;
    return g_eng_tex_watch > 0;
}
/* Arm the watch over a texture's bytes; 1 when the stamp can be trusted. */
static int eng_tex_watch_arm(u32 location, u32 offset, u32 span, u64* stamp)
{
    const u32 ea = cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset);
    u32 unwatched = 1;
    if (!ea) { *stamp = 0; return 0; }
    *stamp = vm_watch_arm(ea, span, &unwatched);
    return unwatched == 0;
}

static u32 eng_texture_slot(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                            u32 levels, u32 pitch, int cube, u32 remap)
{
    if (!w || !h || w > ENG_MAX_TEX_DIM || h > ENG_MAX_TEX_DIM) return 0;
    const u32 span = eng_texture_span(fmt, w, h, levels, pitch, cube);
    if (!span) return 0;

    for (u32 i = 0; i < g.n_textures; i++) {
        eng_texture* e = &g.textures[i];
        if (e->location != location || e->offset != offset ||
            e->format != fmt || e->width != w || e->height != h ||
            e->pitch != pitch || e->remap != remap ||
            e->cubemap != (u32)(cube != 0))
            continue;
        if (e->handle && e->last_hash_frame != g.frames) {
            e->last_hash_frame = g.frames;
            int need_hash = 1;
            /* Not while capturing: the hash's read is what records the
             * texture's pages in the capture. */
            if (eng_tex_watch_on() && !g_rsx_capture_on) {
                /* Armed before the hash, so a write during or after it
                 * changes next frame's stamp. */
                u64 st = 0;
                const int watched = eng_tex_watch_arm(location, offset, span, &st);
                if (watched && e->watched && st == e->watch_stamp) need_hash = 0;
                e->watch_stamp = st; e->watched = (u32)watched;
                if (!need_hash) { s_fstat.tex_skipped++; s_tex_skips_total++; }
                /* RSX_TEX_WATCH_CHECK=1: hash anyway, and report a texture
                 * whose bytes changed while the watch said they had not -- a
                 * write it missed (a kernel write nobody touched, a page
                 * re-opened behind its back). The texture is still updated. */
                { static int chk = -1;
                  if (chk < 0) chk = getenv("RSX_TEX_WATCH_CHECK") ? 1 : 0;
                  if (chk && !need_hash) {
                      int rd = 0;
                      const u64 h2 = eng_texture_content_hash(location, offset, span, &rd);
                      static unsigned long long n_chk, n_bad;
                      n_chk++;
                      if (rd && h2 != e->content_hash) {
                          if (n_bad++ < 20)
                              fprintf(stderr, "[tex-watch] MISSED a write: loc %u off 0x%08X %ux%u fmt 0x%02X span %u (frame %u)\n",
                                      location, offset, w, h, fmt, span, g.frames);
                          need_hash = 1;
                      }
                      if ((n_chk & 0xFFFFF) == 0)
                          fprintf(stderr, "[tex-watch] check: %llu skipped hashes verified, %llu missed\n", n_chk, n_bad);
                  } }
            }
            int readable = 0;
            const u64 hash = need_hash ? eng_texture_content_hash(location, offset, span, &readable) : 0;
            if (need_hash && readable && hash != e->content_hash) {
                const u32 fresh = eng_texture_upload(location, offset, fmt, w, h,
                                                     levels, pitch, cube, remap);
                if (fresh) {
                    g.be->texture_release(g.be->user, e->handle);
                    e->handle = fresh;
                    e->content_hash = hash;
                }
            }
        }
        e->last_use_serial = ++g.texture_use_serial;
        return e->handle;
    }

    u32 index;
    u32 evicted = 0;
    if (g.n_textures < ENG_MAX_TEXTURES) {
        index = g.n_textures++;
    } else {
        index = 0;
        for (u32 i = 1; i < g.n_textures; i++)
            if (g.textures[i].last_use_serial < g.textures[index].last_use_serial)
                index = i;
        evicted = g.textures[index].handle;
    }

    eng_texture e;
    memset(&e, 0, sizeof(e));
    e.location = location; e.offset = offset; e.format = fmt;
    e.width = w; e.height = h; e.pitch = pitch;
    e.remap = remap; e.cubemap = (u32)(cube != 0);
    e.last_hash_frame = g.frames;
    e.last_use_serial = ++g.texture_use_serial;
    if (eng_tex_watch_on()) { u64 st = 0; e.watched = (u32)eng_tex_watch_arm(location, offset, span, &st); e.watch_stamp = st; }
    { int readable = 0;
      e.content_hash = eng_texture_content_hash(location, offset, span, &readable); }
    e.handle = eng_texture_upload(location, offset, fmt, w, h, levels, pitch,
                                  cube, remap);
    if (e.handle && evicted) g.be->texture_release(g.be->user, evicted);
    if (e.handle || !evicted) g.textures[index] = e;
    return e.handle;
}

/* SET_TEXTURE_FILTER's min field is [18:16] (1 NEAREST, 2 LINEAR, then 3..6,
 * the four combinations of a nearest/linear minification with a
 * nearest/linear mip filter) and mag [26:24]; SET_TEXTURE_CONTROL0 carries
 * max LOD at [18:7] and min LOD at [30:19], both 4.8 fixed point. RSX's LOD
 * bias, SET_TEXTURE_FILTER [12:0], is not applied: the reference engine
 * leaves it at zero too. */
static void eng_decode_sampler(u32 filter, u32 wrap, u32 control0,
                               rsx_be_sampler_desc* out)
{
    const u32 minf = (filter >> 16) & 7u;
    const u32 magf = (filter >> 24) & 7u;
    out->min_linear  = (u8)(minf == 2 || minf == 4 || minf == 6);
    out->mag_linear  = (u8)(magf == 2);
    out->mip_present = (u8)(minf >= 3);
    out->mip_linear  = (u8)(minf == 5 || minf == 6);
    out->wrap_s = (u8)(wrap & 0xFu);
    out->wrap_t = (u8)((wrap >> 8) & 0xFu);
    out->wrap_r = (u8)((wrap >> 16) & 0xFu);
    out->min_lod = (float)((control0 >> 19) & 0xFFFu) / 256.0f;
    out->max_lod = out->mip_present ? (float)((control0 >> 7) & 0xFFFu) / 256.0f
                                    : 0.0f;
    if (out->max_lod < out->min_lod) out->max_lod = out->min_lod;
}

/* Texel conversions per unit for rsx_fp_set_texel_ops: TEXTURE_ADDRESS gamma
 * (bits 20-23, R G B A) and UNSIGNED_REMAP_BIASED expansion (bits 12-15 == 1),
 * on the formats the RSX applies them to (RPCS3 get_format_features: gamma
 * and expansion on the 8-bit and compressed colour formats, expansion alone
 * on depth, X16, Y16_X16 and HILO8, neither on float formats). Expansion
 * skips channels the remap fills with a constant and channels gamma already
 * converts. RSX_NO_TEXEL_OPS=1 turns both off, RSX_NO_TEX_GAMMA=1 gamma only. */
static u32 eng_texel_ops(u32 ops[16])
{
    static int off = -1, no_gamma = -1;
    if (off < 0) { off = getenv("RSX_NO_TEXEL_OPS") ? 1 : 0; no_gamma = getenv("RSX_NO_TEX_GAMMA") ? 1 : 0; }
    u32 mask = 0;
    memset(ops, 0, 16 * sizeof ops[0]);
    if (off) return 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        if (!t.enabled) continue;
        const u32 base = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        int can_gamma = 0, can_expand = 0, wide = 0;
        switch (base) {
        case 0x81: case 0x82: case 0x83: case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x8B: case 0x8D: case 0x8E: case 0x8F: case 0x97: case 0x9D: case 0x9E:
            can_gamma = can_expand = 1; break;
        case 0x90: case 0x91: case 0x92: case 0x93: case 0x98:
            can_expand = 1; break;
        case 0x94: case 0x95:
            can_expand = 1; wide = 1; break;
        default: break;
        }
        u32 gamma = can_gamma && !no_gamma ? (t.wrap >> 20) & 0xFu : 0u;
        u32 expand = 0;
        if (can_expand && ((t.wrap >> 12) & 0xFu) == 1u) {
            const u32 r = t.remap & 0xFFFFu;
            const u32 op_r = (r >> 10) & 3u, op_g = (r >> 12) & 3u, op_b = (r >> 14) & 3u, op_a = (r >> 8) & 3u;
            expand = (op_r == 2u ? 1u : 0u) | (op_g == 2u ? 2u : 0u) | (op_b == 2u ? 4u : 0u) | (op_a == 2u ? 8u : 0u);
            expand &= ~gamma;
        }
        if (!gamma && !expand) continue;
        ops[u] = gamma | (expand << 4) | (wide ? 0x100u : 0u);
        mask |= 1u << u;
    }
    return mask;
}

/* A fragment program's identity across draws: its structure without the
 * inline constants (which titles patch per draw), as the pipeline log prints
 * it ("fp-struct="). Per-title corrections name programs by it. */
static u64 eng_fp_struct_id(const u8* fp_uc, u32 fp_size)
{
    return fp_uc ? rsx_fp_structural_hash(fp_uc, fp_size, 1469598103934665603ull) : 0;
}

/* RSX_FP_SAT_ALPHA=<fp-struct>[,<fp-struct>...]: clamp those programs' first
 * colour export alpha to [0, 1]. A per-title correction, set by the title's
 * runner: Drakengard 3's point-light shaft mask writes
 * max(distance / radius, behind-the-light)^4 unclamped into an FP16 target,
 * and its composite multiplies the scene by about 0.3 + 1.05 * mask^2 -- the
 * smoky doorway and the windows of the village interior came out 15-70x too
 * bright. Nothing in the microcode or the FP16 path bounds it, and RPCS3 does
 * not either; clamped, the room matches the original by eye. */
static int eng_fp_sat_alpha(const u8* fp_uc, u32 fp_size)
{
    static u64 ids[16]; static int n = -1;
    if (n < 0) {
        n = 0;
        const char* e = getenv("RSX_FP_SAT_ALPHA");
        while (e && *e && n < 16) {
            char* end; const u64 v = strtoull(e, &end, 16);
            if (end == e) break;
            ids[n++] = v;
            e = (*end == ',') ? end + 1 : end;
        }
    }
    if (!n) return 0;
    const u64 id = eng_fp_struct_id(fp_uc, fp_size);
    for (int i = 0; i < n; i++) if (ids[i] == id) return 1;
    return 0;
}

/* ---- pipelines ----------------------------------------------------------- */

static u32 eng_vtex_mask(void)
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_VERTEX_TEXTURES; u++) {
        rsx_dsp_vertex_texture vt;
        rsx_dsp_get_vertex_texture(&g.rsx, u, &vt);
        if (vt.enabled && vt.width && vt.height) mask |= 1u << u;
    }
    return mask;
}

/* Shadow-map units: an enabled depth-format texture (DEPTH24_D8, DEPTH16 and
 * their float variants) whose TEXTURE_ADDRESS carries a compare function
 * (bits 28-31, CELL_GCM_TEXTURE_ZFUNC_*; 0 = NEVER = no comparison). Such a
 * unit returns the comparison result, not the depth (see
 * rsx_fp_set_shadow_units). RSX_NO_SHADOW_CMP=1 turns it off. */
static u32 eng_shadow_units(u8 funcs[16])
{
    static int off = -1; if (off < 0) off = getenv("RSX_NO_SHADOW_CMP") ? 1 : 0;
    u32 mask = 0;
    memset(funcs, 0, 16);
    if (off) return 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        if (!t.enabled) continue;
        const u32 base = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        if (base < 0x90u || base > 0x93u) continue;
        const u32 zf = (t.wrap >> 28) & 0xFu;
        if (!zf || zf > 7u) continue;
        mask |= 1u << u; funcs[u] = (u8)zf;
    }
    return mask;
}

static u32 eng_cube_mask(void)
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        /* A cube face is square by construction, so a non-square image
         * cannot be one; the fragment program is compiled against this mask
         * and a texturecube slot filled with a 2D texture is a validation
         * failure, not a wrong pixel. */
        if (t.enabled && t.cubemap && t.width && t.width == t.height)
            mask |= 1u << u;
    }
    return mask;
}

/* Are both of the guest's own programs resident? One answer, used by the
 * layout and by the pipeline, because they must not disagree: the built-in
 * program declares all sixteen inputs, so narrowing the layout for it would
 * leave the pipeline's vertex descriptor short of what the shader reads. */
/* What every draw would otherwise work out again from the resident vertex
 * program -- its size, the inputs it reads, its hash -- for as long as no
 * program word is uploaded (rsx.vp_gen) and the start slot stays put.
 * RSX_VP_INFO_NOCACHE=1 recomputes them on every draw. */
static struct {
    int valid;
    u32 gen, start, instrs;
    int mask_valid; u32 input_mask;
    int hash_valid; u64 hash;
} s_vpi;

static int eng_vp_info_current(u32 start)
{
    static int off = -1;
    if (off < 0) off = getenv("RSX_VP_INFO_NOCACHE") ? 1 : 0;
    if (!off && s_vpi.valid && s_vpi.gen == g.rsx.vp_gen && s_vpi.start == start)
        return 1;
    s_vpi.valid = 0;
    s_vpi.mask_valid = s_vpi.hash_valid = 0;
    return 0;
}

static int eng_guest_programs(const u8** out_vp, u32* out_vp_instrs,
                              const u8** out_fp, u32* out_fp_size)
{
    const u8* vp_uc = NULL;
    u32 vp_instrs = 0;
    const u32 start = rsx_dsp_vp_start(&g.rsx);
    if (start < RSX_DSP_VP_INSTR) {
        vp_uc = (const u8*)(g.rsx.vp + start * 4);
        if (eng_vp_info_current(start)) {
            vp_instrs = s_vpi.instrs;
        } else {
            vp_instrs = rsx_vp_program_size_instrs(
                vp_uc, (RSX_DSP_VP_INSTR - start) * 16u);
            s_vpi.valid = 1;
            s_vpi.gen = g.rsx.vp_gen;
            s_vpi.start = start;
            s_vpi.instrs = vp_instrs;
        }
    }

    /* A zero SET_SHADER_PROGRAM is "none", not "the program at offset 0": the
     * register carries the location in its low two bits, so a real program can
     * never read back as zero. */
    const u8* fp_uc = NULL;
    u32 fp_size = 0;
    if (rsx_dsp_reg(&g.rsx, 0x08E4)) {
        u32 fp_loc = 0;
        const u32 fp_off = rsx_dsp_fragment_program(&g.rsx, &fp_loc);
        fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, 16);
        fp_size = fp_uc ? rsx_fp_program_size(fp_uc, ENG_FP_MAX_BYTES) : 0;
        if (fp_size) fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, fp_size);
        if (!fp_uc) fp_size = 0;
    }

    if (out_vp) *out_vp = vp_uc;
    if (out_vp_instrs) *out_vp_instrs = vp_instrs;
    if (out_fp) *out_fp = fp_uc;
    if (out_fp_size) *out_fp_size = fp_size;
    return vp_instrs && fp_size;
}

/* The layout the active vertex program actually reads. An uncertain analysis
 * falls back to all sixteen registers (rsx_live_draw.c:3625-3643). */
static void eng_vertex_layout(rsx_vertex_layout_plan* layout)
{
    u32 mask = 0xFFFFu;
    const u8* uc = NULL;
    u32 instrs = 0;
    if (eng_guest_programs(&uc, &instrs, NULL, NULL)) {
        if (s_vpi.valid && s_vpi.mask_valid) {
            mask = s_vpi.input_mask;
        } else {
            rsx_vp_input_analysis analysis = { 0xFFFFu, 0 };
            if (rsx_vp_analyze_inputs(uc, instrs * 16u, &analysis) == (int)instrs &&
                analysis.exact && analysis.input_mask)
                mask = analysis.input_mask;
            if (s_vpi.valid) { s_vpi.input_mask = mask; s_vpi.mask_valid = 1; }
        }
    }
    rsx_vertex_layout_plan_init(layout, mask);
}

/* What a draw runs before a title has loaded programs of its own.
 *
 * The reference engine has no fallback and drops such a draw, because a title
 * always has both programs resident by the time it draws anything. A general
 * toolkit does not get to assume that: the host harness's fixed-function modes
 * are exactly this case, and so is the first frame of a title that clears and
 * flips before its first SET_TRANSFORM_PROGRAM. It is written in the same HLSL
 * shape the decompilers emit -- ATTRn inputs, the COLOR0/TEXCOORD varyings, the
 * VPConst block and the viewport epilogue -- so it goes through the one
 * pipeline_create the backend already implements rather than needing a second
 * entry point for a built-in shader.
 *
 * Vertex constant rows 0..3 are the transform when the guest programmed them
 * and identity when it did not: an all-zero matrix would collapse every vertex
 * onto the origin. */
static const char* const kEngFixedVS =
"struct VSInput {\n"
"    float4 a0:ATTR0;  float4 a1:ATTR1;  float4 a2:ATTR2;  float4 a3:ATTR3;\n"
"    float4 a4:ATTR4;  float4 a5:ATTR5;  float4 a6:ATTR6;  float4 a7:ATTR7;\n"
"    float4 a8:ATTR8;  float4 a9:ATTR9;  float4 a10:ATTR10; float4 a11:ATTR11;\n"
"    float4 a12:ATTR12; float4 a13:ATTR13; float4 a14:ATTR14; float4 a15:ATTR15;\n"
"};\n"
"struct VSOutput {\n"
"    float4 pos:SV_Position; float4 col0:COLOR0; float4 col1:COLOR1;\n"
"    float4 fog:FOG;\n"
"    float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;\n"
"    float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7;\n"
"    float4 t8:TEXCOORD8; float4 t9:TEXCOORD9;\n"
"};\n"
"cbuffer VPConst : register(b0) {\n"
"    float4 vp_c[512];\n"
"    float4 vp_posscale;\n"
"    float4 vp_posoffset;\n"
"};\n"
"VSOutput main(VSInput input) {\n"
"    float4 _p = float4(dot(vp_c[0], input.a0), dot(vp_c[1], input.a0),\n"
"                       dot(vp_c[2], input.a0), dot(vp_c[3], input.a0));\n"
"    float _m = dot(abs(vp_c[0]), 1.0) + dot(abs(vp_c[1]), 1.0)\n"
"             + dot(abs(vp_c[2]), 1.0) + dot(abs(vp_c[3]), 1.0);\n"
"    if (_m == 0.0) _p = input.a0;\n"
"    VSOutput Out;\n"
"    Out.pos = float4(_p.xyz * vp_posscale.xyz + _p.w * vp_posoffset.xyz, _p.w);\n"
"    Out.col0 = input.a3; Out.col1 = float4(0,0,0,1); Out.fog = (float4)0;\n"
"    Out.t0 = input.a8; Out.t1 = (float4)0; Out.t2 = (float4)0; Out.t3 = (float4)0;\n"
"    Out.t4 = (float4)0; Out.t5 = (float4)0; Out.t6 = (float4)0; Out.t7 = (float4)0;\n"
"    Out.t8 = (float4)0; Out.t9 = (float4)0;\n"
"    return Out;\n"
"}\n";

static const char* const kEngFixedPS =
"struct PSInput {\n"
"    float4 position : SV_POSITION; float4 col0 : COLOR0; float4 col1 : COLOR1;\n"
"    float4 fog : FOG;\n"
"    float4 tc0:TEXCOORD0; float4 tc1:TEXCOORD1; float4 tc2:TEXCOORD2; float4 tc3:TEXCOORD3;\n"
"    float4 tc4:TEXCOORD4; float4 tc5:TEXCOORD5; float4 tc6:TEXCOORD6; float4 tc7:TEXCOORD7;\n"
"    float4 tc8:TEXCOORD8; float4 tc9:TEXCOORD9;\n"
"};\n"
"float4 main(PSInput input) : SV_TARGET { return input.col0; }\n";

/* The pipeline for this draw, built once per distinct key. A negative result
 * is cached too (handle 0), so a program pair that will not translate is not
 * retried on every draw of every frame. */
/* ---- asynchronous pipeline builds ------------------------------------------
 *
 * A pipeline the cache has not seen means translating its programs (HLSL ->
 * SPIR-V -> MSL) and compiling them, and the first time on a machine that is
 * 200-450 ms a pipeline: Drakengard 3 froze for 0.7 s entering an area with a
 * handful of new effects. The walker now hands the build to a worker and
 * waits for it out of a budget of RSX_ASYNC_WAIT_MS a frame (default 8) -- a
 * cached shader is ready well inside that, so nothing changes for it -- and
 * a slow one is skipped until it is ready: the effect appears a few frames
 * late, the first time, instead of the game stopping. RSX_ASYNC_SHADERS=0 builds on the walker as
 * before. Metal and D3D12: both backends' pipeline_create are thread-safe. */
#if defined(__APPLE__) || defined(_WIN32)
#if defined(_WIN32)
/* The pthread names the worker was written with, over Win32: an SRW lock,
 * condition variables and CreateThread. The timed wait takes a deadline the
 * caller computed with timespec_get, so it is converted to a span here. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef SRWLOCK pj_mutex_t;
typedef CONDITION_VARIABLE pj_cond_t;
typedef HANDLE pj_thread_t;
#define PJ_MUTEX_INIT SRWLOCK_INIT
#define PJ_COND_INIT  CONDITION_VARIABLE_INIT
static void pj_lock(pj_mutex_t* m)   { AcquireSRWLockExclusive(m); }
static void pj_unlock(pj_mutex_t* m) { ReleaseSRWLockExclusive(m); }
static void pj_wait(pj_cond_t* c, pj_mutex_t* m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static int  pj_timedwait(pj_cond_t* c, pj_mutex_t* m, const struct timespec* dl)
{
    struct timespec now; timespec_get(&now, TIME_UTC);
    long long ms = (long long)(dl->tv_sec - now.tv_sec) * 1000LL + (dl->tv_nsec - now.tv_nsec) / 1000000LL;
    if (ms <= 0) return 1;
    return SleepConditionVariableSRW(c, m, (DWORD)ms, 0) ? 0 : 1;
}
static void pj_signal(pj_cond_t* c)    { WakeConditionVariable(c); }
static void pj_broadcast(pj_cond_t* c) { WakeAllConditionVariable(c); }
#else
#include <pthread.h>
typedef pthread_mutex_t pj_mutex_t;
typedef pthread_cond_t  pj_cond_t;
typedef pthread_t       pj_thread_t;
#define PJ_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define PJ_COND_INIT  PTHREAD_COND_INITIALIZER
static void pj_lock(pj_mutex_t* m)   { pthread_mutex_lock(m); }
static void pj_unlock(pj_mutex_t* m) { pthread_mutex_unlock(m); }
static void pj_wait(pj_cond_t* c, pj_mutex_t* m) { pthread_cond_wait(c, m); }
static int  pj_timedwait(pj_cond_t* c, pj_mutex_t* m, const struct timespec* dl) { return pthread_cond_timedwait(c, m, dl); }
static void pj_signal(pj_cond_t* c)    { pthread_cond_signal(c); }
static void pj_broadcast(pj_cond_t* c) { pthread_cond_broadcast(c); }
#endif
typedef struct eng_pipe_job {
    char* vs; char* ps;
    rsx_be_render_state rs;
    rsx_vertex_layout_plan layout;
    u32 stride; rsx_be_format rt_fmt; u32 rt_count;
    u32 handle; int done;
    u32 frame;
    struct eng_pipe_job* next;
} eng_pipe_job;
static pj_mutex_t s_pj_mu = PJ_MUTEX_INIT;
static pj_cond_t  s_pj_work = PJ_COND_INIT, s_pj_done = PJ_COND_INIT;
static eng_pipe_job *s_pj_head, *s_pj_tail;
static int s_pj_started, s_pj_quit;
static pj_thread_t s_pj_thread;

static int eng_async_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_ASYNC_SHADERS"); on = !(e && e[0] == '0'); }
    return on;
}
#if defined(_WIN32)
static DWORD WINAPI eng_pipe_worker(LPVOID arg)
#else
static void* eng_pipe_worker(void* arg)
#endif
{
    (void)arg;
#if defined(__APPLE__)
    pthread_setname_np("rsx pipeline build");
#endif
    for (;;) {
        pj_lock(&s_pj_mu);
        while (!s_pj_head && !s_pj_quit) pj_wait(&s_pj_work, &s_pj_mu);
        if (!s_pj_head) { pj_unlock(&s_pj_mu); return 0; }
        eng_pipe_job* j = s_pj_head;
        s_pj_head = j->next;
        if (!s_pj_head) s_pj_tail = NULL;
        pj_unlock(&s_pj_mu);
        struct timespec t0, t1;
        timespec_get(&t0, TIME_UTC);
        const u32 h = g.be->pipeline_create(g.be->user, j->vs, j->ps, &j->rs,
                                            &j->layout, j->stride, j->rt_fmt, j->rt_count);
        timespec_get(&t1, TIME_UTC);
        const double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (ms >= 20.0)
            fprintf(stderr, "[pipe-slow] async build took %.1f ms (asked for in frame %u, ready in frame %u)%s\n",
                    ms, j->frame, g.frames, h ? "" : " -- FAILED");
        free(j->vs); free(j->ps); j->vs = j->ps = NULL;
        pj_lock(&s_pj_mu);
        j->handle = h;
        __atomic_store_n(&j->done, 1, __ATOMIC_RELEASE);
        pj_broadcast(&s_pj_done);
        pj_unlock(&s_pj_mu);
    }
}
/* Queue a build and wait for it a little; the job is the entry's until done. */
static eng_pipe_job* eng_pipe_submit(const char* vs, const char* ps, const rsx_be_render_state* rs,
                                     const rsx_vertex_layout_plan* layout, u32 stride,
                                     rsx_be_format rt_fmt, u32 rt_count)
{
    eng_pipe_job* j = (eng_pipe_job*)calloc(1, sizeof *j);
    if (!j) return NULL;
    j->vs = strdup(vs); j->ps = strdup(ps);
    if (!j->vs || !j->ps) { free(j->vs); free(j->ps); free(j); return NULL; }
    j->rs = *rs; j->layout = *layout; j->stride = stride; j->rt_fmt = rt_fmt; j->rt_count = rt_count;
    j->frame = g.frames;
    pj_lock(&s_pj_mu);
    if (!s_pj_started) {
        s_pj_started = 1; s_pj_quit = 0;
#if defined(_WIN32)
        s_pj_thread = CreateThread(NULL, 64u << 20, eng_pipe_worker, NULL, 0, NULL);
        if (!s_pj_thread) s_pj_started = -1;
#else
        pthread_attr_t at; pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 64u << 20);   /* glslang and spirv-opt recurse deeply */
        if (pthread_create(&s_pj_thread, &at, eng_pipe_worker, NULL) != 0) s_pj_started = -1;
        pthread_attr_destroy(&at);
#endif
    }
    if (s_pj_started < 0) { pj_unlock(&s_pj_mu); free(j->vs); free(j->ps); free(j); return NULL; }
    if (s_pj_tail) s_pj_tail->next = j; else s_pj_head = j;
    s_pj_tail = j;
    pj_signal(&s_pj_work);
    /* The wait is a budget for the frame, not for each build: 38 new
     * effects at once had waited 8 ms apiece, 300 ms in one frame. */
    static long wait_ms = -1;
    static u32 budget_frame;
    static long left_us;
    if (wait_ms < 0) { const char* e = getenv("RSX_ASYNC_WAIT_MS"); wait_ms = e ? atol(e) : 8; }
    if (budget_frame != g.frames) { budget_frame = g.frames; left_us = wait_ms * 1000L; }
    if (left_us > 0) {
        struct timespec t0, dl; timespec_get(&t0, TIME_UTC);
        dl = t0;
        dl.tv_nsec += (left_us % 1000000L) * 1000L; dl.tv_sec += left_us / 1000000L + dl.tv_nsec / 1000000000L;
        dl.tv_nsec %= 1000000000L;
        while (!j->done) if (pj_timedwait(&s_pj_done, &s_pj_mu, &dl) != 0) break;
        struct timespec t1; timespec_get(&t1, TIME_UTC);
        left_us -= (long)((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000L);
    }
    pj_unlock(&s_pj_mu);
    return j;
}
static void eng_pipe_stop(void)
{
    pj_lock(&s_pj_mu);
    const int started = s_pj_started > 0;
    s_pj_quit = 1;
    pj_broadcast(&s_pj_work);
    pj_unlock(&s_pj_mu);
#if defined(_WIN32)
    if (started) { WaitForSingleObject(s_pj_thread, INFINITE); CloseHandle(s_pj_thread); s_pj_thread = NULL; }
#else
    if (started) pthread_join(s_pj_thread, NULL);
#endif
    s_pj_started = 0;
}
/* An entry whose build is in flight: its handle once done (and the job
 * freed), else 0. */
static u32 eng_pipe_poll(eng_pipeline* p)
{
    if (!p->pending) return p->handle;
    if (!__atomic_load_n(&p->job->done, __ATOMIC_ACQUIRE)) return 0;
    p->handle = p->job->handle;
    p->pending = 0;
    free(p->job); p->job = NULL;
    return p->handle;
}
#else
static int eng_async_on(void) { return 0; }
static void eng_pipe_stop(void) {}
static u32 eng_pipe_poll(eng_pipeline* p) { return p->handle; }
#endif

static u32 eng_pipeline_get(const rsx_vertex_layout_plan* layout,
                            const rsx_be_render_state* rs,
                            rsx_be_format rt_fmt, u32 rt_count, int* out_fixed)
{
    *out_fixed = 1;
    const u8* vp_uc = NULL;
    const u8* fp_uc = NULL;
    u32 vp_instrs = 0, fp_size = 0;
    /* No resident program pair means the built-in one; see kEngFixedVS. */
    const int fixed = !eng_guest_programs(&vp_uc, &vp_instrs, &fp_uc, &fp_size);

    const u32 fp_ctrl  = rsx_dsp_shader_control(&g.rsx);
    const u32 cube_mask = eng_cube_mask();
    const u32 vtex_mask = eng_vtex_mask();
    u8 shadow_funcs[16];
    const u32 shadow_mask = eng_shadow_units(shadow_funcs);
    u32 texel_ops[16];
    const u32 texop_mask = eng_texel_ops(texel_ops);
    /* Texel-addressed units (RSX_TEX_FMT_UNNORM): the decompiler scales their
     * coordinates by 1/size. The D3D12 path patched this in; this engine never
     * did, so every post-process pass that samples a render target in texels
     * -- downsampling, blur, bloom, the luminance chain -- read coordinates
     * hundreds of times out of range and got one clamped edge texel. The
     * near-black luminance then drove Drakengard 3's exposure up until the
     * village interior washed out to white. RSX_NO_UNNORM=1 turns it off. */
    u32 unnorm_mask = 0, unnorm_dim[16][2];
    memset(unnorm_dim, 0, sizeof unnorm_dim);
    { static int off = -1; if (off < 0) off = getenv("RSX_NO_UNNORM") ? 1 : 0;
      for (u32 u = 0; !off && u < RSX_DSP_NUM_TEXTURES && u < 16; u++) {
          rsx_dsp_texture t; rsx_dsp_get_texture(&g.rsx, u, &t);
          if (!t.enabled || !(t.format & RSX_TEX_FMT_UNNORM) || ((cube_mask >> u) & 1u)) continue;
          if (!t.width || !t.height) continue;
          unnorm_mask |= 1u << u; unnorm_dim[u][0] = t.width; unnorm_dim[u][1] = t.height;
      } }

    memset(&g.fp_constants, 0, sizeof g.fp_constants);
    if (!fixed && rsx_fp_collect_constants(fp_uc, fp_size, &g.fp_constants) < 0)
        return 0;

    /* Identity: the vertex program's own bytes, the fragment program's
     * STRUCTURE (its inline constants are hoisted into the buffered block, so
     * a constant change must not be a new pipeline), the export-width bit of
     * SHADER_CONTROL, the cube and vertex-texture masks, the input layout,
     * and the structural render state. */
    u64 key = 1469598103934665603ull;
    if (fixed) {
        static const u32 fixed_tag = 0x4E464958u;   /* "XIFN" */
        key = eng_fnv1a(&fixed_tag, sizeof fixed_tag, key);
    } else {
        if (s_vpi.valid && s_vpi.hash_valid) {
            key = s_vpi.hash;
        } else {
            key = eng_fnv1a(vp_uc, vp_instrs * 16u, key);
            if (s_vpi.valid) { s_vpi.hash = key; s_vpi.hash_valid = 1; }
        }
        key = rsx_fp_structural_hash(fp_uc, fp_size, key);
        if (!key) return 0;
        const u32 fp_ctrl_key = fp_ctrl & 0x40u;
        key = eng_fnv1a(&fp_ctrl_key, sizeof fp_ctrl_key, key);
        key = eng_fnv1a(&cube_mask, sizeof cube_mask, key);
        key = eng_fnv1a(&vtex_mask, sizeof vtex_mask, key);
        if (shadow_mask) {
            key = eng_fnv1a(&shadow_mask, sizeof shadow_mask, key);
            key = eng_fnv1a(shadow_funcs, sizeof shadow_funcs, key);
        }
        if (unnorm_mask) {
            key = eng_fnv1a(&unnorm_mask, sizeof unnorm_mask, key);
            key = eng_fnv1a(unnorm_dim, sizeof unnorm_dim, key);
        }
        if (texop_mask) {
            key = eng_fnv1a(&texop_mask, sizeof texop_mask, key);
            key = eng_fnv1a(texel_ops, sizeof texel_ops, key);
        }
    }
    key = eng_fnv1a(&layout->mask, sizeof layout->mask, key);
    key = eng_fnv1a(&layout->stride, sizeof layout->stride, key);
    key = eng_fnv1a(&rt_fmt, sizeof rt_fmt, key);
    /* How many colour attachments the pass will have is pipeline identity:
     * a host API matches the two against each other. */
    key = eng_fnv1a(&rt_count, sizeof rt_count, key);
    key = rsx_draw_engine_hash_render_state(rs, key);

    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].key == key) {
            *out_fixed = g.pipelines[i].fixed;
            return eng_pipe_poll(&g.pipelines[i]);
        }
    if (g.n_pipelines >= ENG_MAX_PIPELINES) return 0;

    u32 handle = 0;
    int vi = 1, fi = 1;
    u32 nconst = 0;
    struct timespec t_create0, t_create1;
    timespec_get(&t_create0, TIME_UTC);
    if (fixed) {
        snprintf(s_vs_hlsl, sizeof s_vs_hlsl, "%s", kEngFixedVS);
        snprintf(s_ps_hlsl, sizeof s_ps_hlsl, "%s", kEngFixedPS);
    } else {
        vi = rsx_vp_decompile_compact_ex(vp_uc, vp_instrs * 16u, vtex_mask,
                                         layout->mask, s_vs_hlsl,
                                         sizeof s_vs_hlsl);
        rsx_fp_set_shadow_units(shadow_mask, shadow_funcs);
        rsx_fp_set_unnorm_units(unnorm_mask, unnorm_dim);
        rsx_fp_set_texel_ops(texop_mask ? texel_ops : NULL);
        rsx_fp_set_saturate_alpha(eng_fp_sat_alpha(fp_uc, fp_size));
        fi = rsx_fp_decompile_buffered_ex(fp_uc, fp_size, fp_ctrl, cube_mask,
                                          s_ps_hlsl, sizeof s_ps_hlsl, &nconst);
        rsx_fp_set_shadow_units(0, NULL);
        rsx_fp_set_unnorm_units(0, NULL);
        rsx_fp_set_texel_ops(NULL);
        rsx_fp_set_saturate_alpha(0);
        if (unnorm_mask) { static int n = 0; if (n++ < 6)
            fprintf(stderr, "[rsx engine] unnormalised units 0x%X (unit %u %ux%u)\n", unnorm_mask,
                    __builtin_ctz(unnorm_mask), unnorm_dim[__builtin_ctz(unnorm_mask)][0], unnorm_dim[__builtin_ctz(unnorm_mask)][1]); }
        if (shadow_mask) { static int n = 0; if (n++ < 6)
            fprintf(stderr, "[rsx engine] shadow-compare units 0x%X (func of first: %u)\n",
                    shadow_mask, shadow_funcs[__builtin_ctz(shadow_mask)]); }
        if (fi > 0 && nconst != g.fp_constants.count) fi = -1;
        if (fi > 0 && rs->alpha_test_enable &&
            rsx_fp_apply_alpha_test_buffered(s_ps_hlsl, sizeof s_ps_hlsl,
                                             rs->alpha_func) < 0)
            fi = -1;
    }
    struct eng_pipe_job* job = NULL;
#if defined(__APPLE__) || defined(_WIN32)
    if (vi > 0 && fi > 0 && !fixed && eng_async_on()) {
        job = eng_pipe_submit(s_vs_hlsl, s_ps_hlsl, rs, layout, layout->stride, rt_fmt, rt_count);
        if (job && __atomic_load_n(&job->done, __ATOMIC_ACQUIRE)) {
            handle = job->handle; free(job); job = NULL;
        } else if (job) {
            handle = 0;                          /* not ready: this draw is skipped */
        }
    }
    if (!job && handle == 0)
#endif
    if (vi > 0 && fi > 0)
        handle = g.be->pipeline_create(g.be->user, s_vs_hlsl, s_ps_hlsl, rs,
                                       layout, layout->stride, rt_fmt, rt_count);
    timespec_get(&t_create1, TIME_UTC);
    /* RSX_PIPE_LOG=1 lifts the cap: the draw trace names pipelines by handle,
     * and the fragment program's FNV hash is what the shader dump files carry. */
    { static u32 logs = 0; static int uncapped = -1;
      if (uncapped < 0) uncapped = getenv("RSX_PIPE_LOG") ? 1 : 0;
      if (uncapped || logs++ < 32) {
        unsigned long long fh = 1469598103934665603ull;
        for (u32 i = 0; fp_uc && i < fp_size; i++) fh = (fh ^ fp_uc[i]) * 1099511628211ull;
        /* The HLSL hashes are the names the shader dump files carry
         * (PS3RECOMP_METAL_SHADER_DUMP writes vp_<hash>.msl / fp_<hash>.msl). */
        unsigned long long vh = 1469598103934665603ull, ph = 1469598103934665603ull;
        for (const char* c = s_vs_hlsl; *c; c++) vh = (vh ^ (unsigned char)*c) * 1099511628211ull;
        for (const char* c = s_ps_hlsl; *c; c++) ph = (ph ^ (unsigned char)*c) * 1099511628211ull;
        fprintf(stderr, "[rsx engine] pipeline %016llx: %s vp %d, fp %d,"
                        " %u constants -> %s handle=%u fp-hash=%016llx fp-size=%u blend=%u(%X/%X) vp-hlsl=%016llx fp-hlsl=%016llx vp-start=%u vp-instrs=%u"
                        " prec(h/x12/x9)=%u/%u/%u exptex=%u fp-struct=%016llx\n",
                (unsigned long long)key, fixed ? "built-in" : "guest",
                vi, fi, nconst, handle ? "ok" : job ? "compiling (draw skipped until it is)" : "FAILED (draw dropped)",
                handle, fh, fp_size,
                rs->blend_enable, rs->sf_rgb, rs->df_rgb, vh, ph, rsx_dsp_vp_start(&g.rsx), vp_instrs,
                g_rsx_fp_stats.prec[1], g_rsx_fp_stats.prec[2], g_rsx_fp_stats.prec[3], g_rsx_fp_stats.exp_tex,
                (unsigned long long)eng_fp_struct_id(fp_uc, fp_size));
        /* RSX_VP_HEX=<vp-hlsl hash>: that program's microcode, four words per
         * instruction, for decoding a field by hand against the decompiler. */
        { static unsigned long long want = 0; static int got = -1;
          if (got < 0) { const char* e = getenv("RSX_VP_HEX"); want = e ? strtoull(e, 0, 16) : 0; got = 0; }
          if (want && vh == want && !got && vp_uc) { got = 1;
              fprintf(stderr, "[vp-hex] vp-hlsl=%016llx start=%u instrs=%u\n", vh, rsx_dsp_vp_start(&g.rsx), vp_instrs);
              for (u32 i = 0; i < vp_instrs; i++) {
                  const u32* w = (const u32*)(vp_uc + i * 16);
                  fprintf(stderr, "[vp-hex] %3u: %08X %08X %08X %08X\n", i, w[0], w[1], w[2], w[3]); } } } } }

    g.pipelines[g.n_pipelines].key = key;
    g.pipelines[g.n_pipelines].handle = handle;
    g.pipelines[g.n_pipelines].fixed = (u8)fixed;
    g.pipelines[g.n_pipelines].pending = job ? 1 : 0;
    g.pipelines[g.n_pipelines].job = job;
    { u64 vf = 1469598103934665603ull, vh2 = 1469598103934665603ull;
      for (u32 i = 0; vp_uc && i < vp_instrs * 16u; i++) vf = (vf ^ vp_uc[i]) * 1099511628211ull;
      for (const char* c = s_vs_hlsl; *c; c++) vh2 = (vh2 ^ (unsigned char)*c) * 1099511628211ull;
      g.pipelines[g.n_pipelines].vp_fnv = vf;
      g.pipelines[g.n_pipelines].vp_hlsl = fixed ? 0 : vh2;
      u64 fh2 = 1469598103934665603ull;
      for (const char* c = s_ps_hlsl; *c; c++) fh2 = (fh2 ^ (unsigned char)*c) * 1099511628211ull;
      g.pipelines[g.n_pipelines].fp_hlsl = fixed ? 0 : fh2; }
    /* A pipeline built mid-game stalls the FIFO walker for as long as it
     * takes (decompile, translate, Metal compile): name every slow one. */
    { static double total_ms = 0;
      const double ms = (double)(t_create1.tv_sec - t_create0.tv_sec) * 1e3 +
                        (double)(t_create1.tv_nsec - t_create0.tv_nsec) / 1e6;
      total_ms += ms;
      s_fstat.pipes++; s_fstat.pipe_ms += ms;
      if (ms >= 20.0)
          fprintf(stderr, "[pipe-slow] frame %u: new pipeline took %.1f ms (vp-hlsl %016llx fp-hlsl %016llx); %u built, %.0f ms in all\n",
                  g.frames, ms, (unsigned long long)g.pipelines[g.n_pipelines].vp_hlsl,
                  (unsigned long long)g.pipelines[g.n_pipelines].fp_hlsl, g.n_pipelines + 1, total_ms); }
    g.n_pipelines++;
    *out_fixed = fixed;
    return handle;
}

/* ---- draw accumulation --------------------------------------------------- */

static void dc_reset(void)
{
    dc.n_arr = dc.n_idx = dc.n_packets = 0;
    dc.n_refs = dc.n_source_refs = dc.n_verts = dc.n_cuts = 0;
    dc.out_verts = NULL;
    dc.refs_remapped = 0;
    dc.fetch_ok = 1;
    dc.inl = NULL;
    dc.inl_bytes = 0;
}

static int dc_push_ref(u32 vertex_id, u32 base_index)
{
    if (dc.n_refs >= dc.cap_refs) {
        const u32 next = dc.cap_refs ? dc.cap_refs * 2u : 4096u;
        rsx_vertex_ref* r = (rsx_vertex_ref*)realloc(dc.refs,
                                                     (size_t)next * sizeof(*r));
        if (!r) return 0;
        dc.refs = r;
        dc.cap_refs = next;
    }
    dc.refs[dc.n_refs].vertex_id = vertex_id;
    dc.refs[dc.n_refs].base_index = base_index;
    dc.n_refs++;
    return 1;
}

static int dc_reserve_verts(u64 bytes)
{
    if (bytes <= dc.verts_cap) return 1;
    u64 next = dc.verts_cap ? dc.verts_cap : (1u << 20);
    while (next < bytes) next *= 2u;
    u8* v = (u8*)realloc(dc.verts, (size_t)next);
    if (!v) return 0;
    dc.verts = v;
    dc.verts_cap = next;
    return 1;
}

static int dc_reserve_indices(u32 count)
{
    if (count <= g.index_cap) return 1;
    u32 next = g.index_cap ? g.index_cap : 4096u;
    while (next < count) next *= 2u;
    u32* p = (u32*)realloc(g.indices, (size_t)next * sizeof(u32));
    if (!p) return 0;
    g.indices = p;
    g.index_cap = next;
    return 1;
}

/* Turn this group's batches into a reference list, recording a CUT rather
 * than fetching a phantom vertex wherever the restart sentinel appears, then
 * collapse repeated references and fetch each unique vertex once.
 * rsx_live_draw.c's fetch_batches_hoisted (4165-4292). */
/* RSX_VFETCH_CHECK totals, printed at exit. */
static unsigned long *s_vfc_draws, *s_vfc_bad, *s_vfc_bytes;
static void eng_vfetch_check_summary(void)
{
    if (s_vfc_draws)
        fprintf(stderr, "[vfetch-check] total: %lu draws compared, %lu mismatched (%lu bytes)\n",
                *s_vfc_draws, *s_vfc_bad, *s_vfc_bytes);
}

/* ---- vertex cache -------------------------------------------------------- *
 *
 * Most of what a frame draws is the same static meshes from the same guest
 * vertex and index buffers, frame after frame, and converting them again was
 * most of the draw engine's time: a Drakengard 3 battle frame decodes ~280k
 * vertices (19 MB) from ~560k index references. A draw is keyed on everything
 * that decides its conversion other than memory -- the vertex layout, every
 * fetched attribute's descriptor and default, the index array and the batch
 * ranges -- and an entry is reused only when a hash of the bytes it read (the
 * index ranges, then each attribute's vertex span) still matches. The hash is
 * taken BEFORE the conversion reads memory, so a guest write racing it can
 * only cause a miss later, never a stale hit. RSX_VCACHE=0 turns it off;
 * RSX_VCACHE_CHECK=1 converts every hit again and compares. */
typedef struct { u32 location, start, len; } EngVCSpan;
typedef struct {
    u64 key, content;                       /* key 0: empty slot            */
    u32 last_frame;
    u32 n_verts, stride, n_source_refs, n_draw;
    int indexed;
    u32 n_spans;
    EngVCSpan span[RSX_DSP_NUM_VERTEX_ATTR];
    u8* verts;
    u32* indices;
    size_t bytes;
    u32 buf;                                /* backend buffer, or 0          */
    u32 ib_off;                             /* the indices' offset in it     */
    u64 watch_stamp;                        /* write-watch stamp at the last hash */
    int watched;                            /* 1: every page was watched then */
} EngVCEntry;
typedef struct {
    int ok;
    u64 content;
    u32 n_spans;
    EngVCSpan span[RSX_DSP_NUM_VERTEX_ATTR];
} EngVCFill;
#define ENG_VC_SLOTS  8192u                 /* power of two                 */
#define ENG_VC_BUDGET ((size_t)384 << 20)
#define ENG_VC_GPU_MIN ((size_t)64 << 10)    /* entries this big get a buffer */
static EngVCEntry* s_vc;
static u32 s_vc_count;
static size_t s_vc_bytes;
static struct { unsigned long long hit, miss, stored, uncacheable, check_bad, stale, hit_watch; } s_vcstat;

static inline u64 eng_vc_mum(u64 a, u64 b)
{
    const __uint128_t r = (__uint128_t)a * b;
    return (u64)r ^ (u64)(r >> 64);
}

/* Four independent 128-bit-multiply lanes over 64-byte blocks; every byte
 * reaches every bit of the result, unlike the texture mutation hash. */
static u64 eng_vc_hash(const u8* p, size_t n, u64 seed)
{
    const u64 s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull,
              s2 = 0x8ebc6af09c88c6e3ull, s3 = 0x589965cc75374cc3ull;
    u64 a = seed ^ s0, b = seed ^ s1, c = seed ^ s2, d = seed ^ s3;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        u64 w[8];
        memcpy(w, p + i, sizeof w);
        a = eng_vc_mum(w[0] ^ s0, w[1] ^ a);
        b = eng_vc_mum(w[2] ^ s1, w[3] ^ b);
        c = eng_vc_mum(w[4] ^ s2, w[5] ^ c);
        d = eng_vc_mum(w[6] ^ s3, w[7] ^ d);
    }
    for (; i + 8 <= n; i += 8) {
        u64 w; memcpy(&w, p + i, 8);
        a = eng_vc_mum(w ^ s1, a ^ s2);
    }
    if (i < n) {
        u64 w = 0; memcpy(&w, p + i, n - i);
        b = eng_vc_mum(w ^ s3, b ^ s0);
    }
    const u64 h = eng_vc_mum(a ^ s1, b ^ s2) ^ eng_vc_mum(c ^ s3, d ^ s0);
    return eng_vc_mum(h ^ (u64)n, s1 ^ seed);
}

static int eng_vc_enabled(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_VCACHE"); on = !(e && e[0] == '0'); }
    return on;
}
static unsigned long long s_vc_checked, s_vc_check_bad_total;
static void eng_vc_check_summary(void)
{
    fprintf(stderr, "[vcache-check] %llu hits converted again, %llu mismatched\n",
            s_vc_checked, s_vc_check_bad_total);
}
static int eng_vc_checking(void)
{
    static int on = -1;
    if (on < 0) { on = getenv("RSX_VCACHE_CHECK") ? 1 : 0; if (on) atexit(eng_vc_check_summary); }
    return on;
}

/* Everything other than memory that decides the draw's conversion; 0 for a
 * draw the cache does not take (an inline stream, or no batches). */
static u64 eng_vc_key(const rsx_vertex_layout_plan* layout, u32 prim, int rebuild)
{
    /* A capture records the guest pages a draw reads (eng_guest_ptr); a
     * cache hit reads none, so while one runs every draw converts afresh. */
    if (!eng_vc_enabled() || g_rsx_capture_on || dc.inl || dc.inl_bytes) return 0;
    if (!dc.n_arr && !dc.n_idx) return 0;
    rsx_vertex_fetch_plan plan;
    rsx_vertex_fetch_plan_init(&plan, &g.rsx, layout, eng_guest_ptr, NULL);
    u32 m[8 + RSX_DSP_NUM_VERTEX_ATTR * 13 + 2 * ENG_MAX_BATCHES * 2 + 16];
    u32 n = 0;
    m[n++] = prim; m[n++] = (u32)rebuild;
    m[n++] = layout->mask; m[n++] = layout->stride; m[n++] = layout->count;
    m[n++] = plan.base_offset; m[n++] = plan.divider_mask;
    for (u32 slot = 0; slot < layout->count; slot++) {
        const u32 attr = layout->attrs[slot];
        const rsx_vertex_fetch_attr* f = &plan.attr[attr];
        m[n++] = attr;
        m[n++] = f->desc.type; m[n++] = f->desc.size; m[n++] = f->desc.stride;
        m[n++] = f->desc.frequency; m[n++] = f->desc.offset; m[n++] = f->desc.location;
        m[n++] = f->elem_size; m[n++] = f->stride;
        memcpy(&m[n], f->default_value, 16); n += 4;
    }
    m[n++] = dc.n_arr;
    for (u32 b = 0; b < dc.n_arr; b++) { m[n++] = dc.arr[b].first; m[n++] = dc.arr[b].count; }
    m[n++] = dc.n_idx;
    for (u32 b = 0; b < dc.n_idx; b++) { m[n++] = dc.idx[b].first; m[n++] = dc.idx[b].count; }
    if (dc.n_idx) {
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(&g.rsx, &ia);
        m[n++] = ia.offset; m[n++] = ia.location; m[n++] = ia.is_u32;
        m[n++] = rsx_dsp_vertex_data_base_index(&g.rsx);
        m[n++] = (u32)rsx_dsp_restart_index_enabled(&g.rsx, ia.is_u32);
        m[n++] = rsx_dsp_restart_index(&g.rsx);
    }
    const u64 k = eng_vc_hash((const u8*)m, (size_t)n * 4u, 0x5643414348450001ull);
    return k ? k : 1;
}

/* The index bytes the draw reads, batch by batch; 0 when a batch is not one
 * contiguous guest range. */
static int eng_vc_hash_indices(u64* h)
{
    if (!dc.n_idx) return 1;
    rsx_dsp_index_array ia;
    rsx_dsp_get_index_array(&g.rsx, &ia);
    const u32 esz = ia.is_u32 ? 4u : 2u;
    const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
    for (u32 b = 0; b < dc.n_idx; b++) {
        const u64 start = (u64)ia.offset + (u64)dc.idx[b].first * esz;
        const u64 bytes = (u64)dc.idx[b].count * esz;
        if (!bytes) continue;
        if (start + bytes > 0x100000000ull) return 0;
        const u8* p = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
        if (!p) return 0;
        *h = eng_vc_hash(p, (size_t)bytes, *h);
    }
    return 1;
}

static int eng_vc_hash_spans(const EngVCSpan* sp, u32 n, u64* h)
{
    for (u32 i = 0; i < n; i++) {
        const u8* p = eng_guest_ptr(NULL, sp[i].location, sp[i].start, sp[i].len);
        if (!p) return 0;
        *h = eng_vc_hash(p, sp[i].len, *h);
    }
    return 1;
}

static EngVCEntry* eng_vc_slot(u64 key)
{
    if (!s_vc) {
        s_vc = (EngVCEntry*)calloc(ENG_VC_SLOTS, sizeof *s_vc);
        if (!s_vc) return NULL;
    }
    u32 i = (u32)key & (ENG_VC_SLOTS - 1u);
    for (u32 probe = 0; probe < ENG_VC_SLOTS; probe++) {
        EngVCEntry* e = &s_vc[i];
        if (!e->key || e->key == key) return e;
        i = (i + 1u) & (ENG_VC_SLOTS - 1u);
    }
    return NULL;
}

/* Arm the texture write-watch (vm_watch) over everything validating an entry
 * would hash -- its vertex spans and this draw's index ranges -- and return
 * the combined stamp; *trusted is 1 when every page is watched. Armed before
 * any hash, so a write during or after one changes the next stamp. */
int g_eng_vc_watch = 1;   /* DOD3_AB=vcwatch switches it in a run */
static u64 eng_vc_arm(const EngVCEntry* e, int* trusted)
{
    *trusted = 0;
    if (!g_eng_vc_watch || !eng_tex_watch_on() || g_rsx_capture_on) return 0;
    int ok = 1;
    u64 st = 0x5643574154434801ull;
    for (u32 i = 0; i < e->n_spans; i++) {
        u64 x = 0;
        if (!eng_tex_watch_arm(e->span[i].location, e->span[i].start, e->span[i].len, &x)) ok = 0;
        st = eng_vc_mum(st ^ x, 0x9E3779B97F4A7C15ull) + e->span[i].start;
    }
    if (dc.n_idx) {
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(&g.rsx, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;
        for (u32 b = 0; b < dc.n_idx; b++) {
            const u64 start = (u64)ia.offset + (u64)dc.idx[b].first * esz;
            const u64 bytes = (u64)dc.idx[b].count * esz;
            if (!bytes) continue;
            if (start + bytes > 0x100000000ull) return 0;
            u64 x = 0;
            if (!eng_tex_watch_arm(ia_loc, (u32)start, (u32)bytes, &x)) ok = 0;
            st = eng_vc_mum(st ^ x, 0x9E3779B97F4A7C15ull) + start;
        }
    }
    *trusted = ok;
    return st;
}

/* A valid entry for the key: present, and the bytes it read unchanged --
 * known from the write-watch when every page it reads is watched and none
 * was written since the last check, otherwise by hashing them again (the
 * texture cache does the same; RSX_TEX_WATCH=0 turns both off). */
static EngVCEntry* eng_vc_lookup(u64 key)
{
    EngVCEntry* e = eng_vc_slot(key);
    if (!e || e->key != key) return NULL;
    int trusted = 0;
    const u64 st = eng_vc_arm(e, &trusted);
    if (trusted && e->watched && st == e->watch_stamp) {
        /* RSX_VC_WATCH_CHECK=1: hash anyway and count the hits the watch got
         * wrong (a write it never saw) as check mismatches. */
        static int chk = -1;
        if (chk < 0) chk = getenv("RSX_VC_WATCH_CHECK") ? 1 : 0;
        if (chk) {
            u64 hc = key;
            if (!eng_vc_hash_indices(&hc) || !eng_vc_hash_spans(e->span, e->n_spans, &hc) || hc != e->content) {
                s_vcstat.check_bad++;
                e->watched = 0;
                return NULL;
            }
        }
        s_vcstat.hit_watch++;
        e->last_frame = g.frames;
        return e;
    }
    u64 h = key;
    if (!eng_vc_hash_indices(&h) || !eng_vc_hash_spans(e->span, e->n_spans, &h) ||
        h != e->content) {
        s_vcstat.stale++;
        e->watched = 0;
        return NULL;
    }
    e->watch_stamp = st;
    e->watched = trusted;
    e->last_frame = g.frames;
    return e;
}

static void eng_vc_free(EngVCEntry* e)
{
    if (e->buf) g.be->buffer_release(g.be->user, e->buf);   /* frees later */
    else { free(e->verts); free(e->indices); }
    s_vc_bytes -= e->bytes;
    memset(e, 0, sizeof *e);
}

/* Over budget, or the table filling up: keep what the last two frames drew,
 * re-inserted so the probe chains stay intact, and drop the rest -- or
 * everything, when even the recent draws would leave it nearly full again. */
static void eng_vc_evict(void)
{
    const double t0 = eng_now_ms();
    const u32 before = s_vc_count;
    size_t recent = 0;
    u32 n_recent = 0;
    for (u32 i = 0; i < ENG_VC_SLOTS; i++)
        if (s_vc[i].key && s_vc[i].last_frame + 1u >= g.frames) { recent += s_vc[i].bytes; n_recent++; }
    const int keep_none = recent > ENG_VC_BUDGET / 2u || n_recent > ENG_VC_SLOTS / 2u;
    EngVCEntry* old = s_vc;
    s_vc = (EngVCEntry*)calloc(ENG_VC_SLOTS, sizeof *s_vc);
    if (!s_vc) { s_vc = old; return; }
    s_vc_count = 0;
    for (u32 i = 0; i < ENG_VC_SLOTS; i++) {
        EngVCEntry* e = &old[i];
        if (!e->key) continue;
        if (keep_none || e->last_frame + 1u < g.frames) { eng_vc_free(e); continue; }
        EngVCEntry* d = eng_vc_slot(e->key);
        if (!d) { eng_vc_free(e); continue; }
        *d = *e;
        s_vc_count++;
    }
    free(old);
    s_fstat.vc_evicts++; s_fstat.vc_evict_ms += eng_now_ms() - t0; s_fstat.vc_evicted += before - s_vc_count;
}

static void eng_vc_store(u64 key, const EngVCFill* f, const u8* verts,
                         u32 n_verts, u32 stride, u32 n_source_refs,
                         int indexed, const u32* indices, u32 n_draw)
{
    if (s_vc_bytes > ENG_VC_BUDGET || s_vc_count >= ENG_VC_SLOTS * 3u / 4u)
        eng_vc_evict();
    EngVCEntry* e = eng_vc_slot(key);
    if (!e) return;
    if (e->key) { eng_vc_free(e); s_vc_count--; }
    const size_t vb = (size_t)n_verts * stride;
    const size_t ib = indexed ? (size_t)n_draw * sizeof(u32) : 0;
    u8* v = NULL;
    u32* x = NULL;
    u32 buf = 0, ib_off = 0;
    /* A large entry lives in a buffer of its own that the GPU reads in place,
     * where a small one is copied into the per-submit arena with the rest. */
    /* RSX_VCACHE_GPU_MIN=<bytes>: the smallest entry given a buffer of its own. */
    static size_t gpu_min = 0;
    if (!gpu_min) { const char* e = getenv("RSX_VCACHE_GPU_MIN");
                    gpu_min = e ? (size_t)strtoull(e, 0, 0) : ENG_VC_GPU_MIN; if (!gpu_min) gpu_min = 1; }
    if (vb + ib >= gpu_min && g.be->buffer_wrap && g.be->draw_buffer &&
        g.be->buffer_release) {
        /* Page-aligned, a whole number of pages: what buffer_wrap takes. On
         * Windows the memory comes from _aligned_malloc and the backend frees
         * it with _aligned_free; elsewhere posix_memalign and free. */
#ifdef _WIN32
        const size_t page = 4096u;
#else
        const size_t page = (size_t)getpagesize();
#endif
        ib_off = (u32)((vb + 255u) & ~(size_t)255u);
        const size_t total = ((size_t)ib_off + ib + page - 1u) & ~(page - 1u);
        void* mem = NULL;
#ifdef _WIN32
        if (total <= 0xFFFFFFFFu) mem = _aligned_malloc(total, page);
        if (mem) {
#else
        if (total <= 0xFFFFFFFFu && posix_memalign(&mem, page, total) == 0) {
#endif
            memcpy(mem, verts, vb);
            if (ib) memcpy((u8*)mem + ib_off, indices, ib);
            buf = g.be->buffer_wrap(g.be->user, mem, (u32)total);
            if (buf) { v = (u8*)mem; x = ib ? (u32*)((u8*)mem + ib_off) : NULL; }
#ifdef _WIN32
            else _aligned_free(mem);
#else
            else free(mem);
#endif
        }
    }
    if (!buf) {
        ib_off = 0;
        v = (u8*)malloc(vb ? vb : 1);
        x = ib ? (u32*)malloc(ib) : NULL;
        if (!v || (ib && !x)) { free(v); free(x); return; }
        memcpy(v, verts, vb);
        if (ib) memcpy(x, indices, ib);
    }
    e->key = key;
    e->content = f->content;
    e->watch_stamp = 0; e->watched = 0;   /* validated by hashing until a lookup arms the watch */
    e->last_frame = g.frames;
    e->n_verts = n_verts; e->stride = stride;
    e->n_source_refs = n_source_refs;
    e->indexed = indexed; e->n_draw = n_draw;
    e->n_spans = f->n_spans;
    memcpy(e->span, f->span, sizeof e->span);
    e->verts = v; e->indices = x;
    e->buf = buf; e->ib_off = ib_off;
    e->bytes = vb + ib;
    s_vc_bytes += e->bytes;
    s_vc_count++;
    s_vcstat.stored++;
    s_fstat.vc_store++; s_fstat.vc_bytes += vb + ib;
}

/* The prepared plan's vertex spans, merged where interleaved attributes
 * overlap, hashed into the fill. 0 when an attribute has no single span. */
static int eng_vc_fill_spans(EngVCFill* f, const rsx_vertex_fetch_plan* plan)
{
    EngVCSpan sp[RSX_DSP_NUM_VERTEX_ATTR];
    u32 n = 0;
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const rsx_vertex_fetch_attr* a = &plan->attr[plan->layout.attrs[slot]];
        if (!a->desc.type || !a->desc.size) continue;
        if (!a->stable_base || !a->elem_size) return 0;
        const u64 prefix = (u64)plan->base_offset + a->desc.offset;
        const u64 start = prefix + (u64)a->stable_first * a->stride;
        const u64 end = prefix + (u64)a->stable_last * a->stride + a->elem_size;
        if (end <= start || end > 0x100000000ull) return 0;
        EngVCSpan s = { a->desc.location, (u32)start, (u32)(end - start) };
        u32 k = n;
        while (k > 0 && (sp[k - 1].location > s.location ||
                         (sp[k - 1].location == s.location && sp[k - 1].start > s.start))) {
            sp[k] = sp[k - 1]; k--;
        }
        sp[k] = s; n++;
    }
    f->n_spans = 0;
    for (u32 i = 0; i < n; i++) {
        EngVCSpan* last = f->n_spans ? &f->span[f->n_spans - 1] : NULL;
        if (last && last->location == sp[i].location &&
            sp[i].start <= last->start + last->len) {
            const u32 end = sp[i].start + sp[i].len;
            if (end > last->start + last->len) last->len = end - last->start;
        } else {
            f->span[f->n_spans++] = sp[i];
        }
    }
    return eng_vc_hash_spans(f->span, f->n_spans, &f->content);
}

static void dc_fetch(const rsx_vertex_layout_plan* layout, int allow_remap,
                     EngVCFill* vc_fill)
{
    /* The index bytes are hashed before they are read; see the cache note. */
    if (vc_fill && vc_fill->ok && !eng_vc_hash_indices(&vc_fill->content))
        vc_fill->ok = 0;

    for (u32 b = 0; b < dc.n_arr && dc.fetch_ok; b++)
        for (u32 i = 0; i < dc.arr[b].count && dc.fetch_ok; i++)
            if (!dc_push_ref(dc.arr[b].first + i, 0)) dc.fetch_ok = 0;

    if (dc.n_idx && dc.fetch_ok) {
        const u32 base_index = rsx_dsp_vertex_data_base_index(&g.rsx);
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(&g.rsx, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const int restart_en = rsx_dsp_restart_index_enabled(&g.rsx, ia.is_u32);
        const u32 restart_val = rsx_dsp_restart_index(&g.rsx);
        /* The index array's location field is a DMA selector, not the
         * rsx_dsp location enum: 0 selects local memory. */
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;

        for (u32 b = 0; b < dc.n_idx && dc.fetch_ok; b++) {
            const u32 first = dc.idx[b].first, count = dc.idx[b].count;
            const u64 start = (u64)ia.offset + (u64)first * esz;
            const u64 bytes = (u64)count * esz;
            const u8* run = NULL;
            if (count && start <= 0xFFFFFFFFull && bytes <= 0xFFFFFFFFull &&
                start + bytes <= 0x100000000ull)
                run = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
            for (u32 i = 0; i < count && dc.fetch_ok; i++) {
                const u8* p = run ? run + (size_t)i * esz
                    : eng_guest_ptr(NULL, ia_loc, ia.offset + (first + i) * esz, esz);
                if (!p) { dc.fetch_ok = 0; break; }
                const u32 index = ia.is_u32
                    ? (((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3])
                    : (u32)((p[0] << 8) | p[1]);
                if (restart_en && index == restart_val) {
                    if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts,
                                              dc.n_refs))
                        dc.fetch_ok = 0;
                    continue;
                }
                if (!dc_push_ref(index, base_index)) dc.fetch_ok = 0;
            }
        }
    }
    if (!dc.fetch_ok) return;

    dc.n_source_refs = dc.n_refs;
    /* Only a topology this engine rebuilds through an index buffer may share
     * repeated references. A line strip or a point list is drawn in the order
     * it arrived, so collapsing duplicates would reorder it. */
    if (allow_remap && dc.n_refs > 1) {
        u32 unique = dc.n_refs;
        if (rsx_vertex_remap_build(&dc.ref_remap, dc.refs, dc.n_refs, &unique)) {
            dc.refs_remapped = unique < dc.n_refs;
            dc.n_refs = unique;
        }
    }

    dc.layout = *layout;
    rsx_vertex_fetch_plan_init(&dc.fetch_plan, &g.rsx, layout, eng_guest_ptr, NULL);
    rsx_vertex_fetch_plan_set_inline(&dc.fetch_plan, &g.rsx, dc.inl, dc.inl_bytes);
    rsx_vertex_fetch_plan_prepare(&dc.fetch_plan, dc.refs, dc.n_refs);
    if (vc_fill && vc_fill->ok && !eng_vc_fill_spans(vc_fill, &dc.fetch_plan))
        vc_fill->ok = 0;
    if (!dc_reserve_verts((u64)dc.n_refs * layout->stride)) {
        dc.fetch_ok = 0;
        return;
    }
    /* rsx_vertex_fetch_all decodes attribute by attribute (one format
     * dispatch per attribute, not per component per vertex); it was the
     * largest share of the draw engine's time. RSX_VFETCH_ONE=1 keeps the
     * per-vertex rsx_vertex_fetch_one path; RSX_VFETCH_CHECK=1 runs both on
     * every draw and reports any byte that differs. */
    static int vf_mode = -1;
    if (vf_mode < 0) vf_mode = getenv("RSX_VFETCH_ONE") ? 1 : getenv("RSX_VFETCH_CHECK") ? 2 : 0;
    if (vf_mode == 1) {
        for (u32 i = 0; i < dc.n_refs; i++) {
            u8* dst = layout->stride ? dc.verts + (u64)i * layout->stride : NULL;
            if (!rsx_vertex_fetch_one(&dc.fetch_plan, &dc.refs[i], dst)) {
                dc.fetch_ok = 0;
                return;
            }
        }
    } else {
        const int ok = rsx_vertex_fetch_all(&dc.fetch_plan, dc.refs, dc.n_refs, dc.verts);
        if (vf_mode == 2) {
            static u8* ref_buf; static u64 ref_cap;
            static unsigned long draws, bad_draws, bad_bytes;
            static int summary_armed = 0;
            if (!summary_armed) { summary_armed = 1; atexit(eng_vfetch_check_summary); }
            s_vfc_draws = &draws; s_vfc_bad = &bad_draws; s_vfc_bytes = &bad_bytes;
            const u64 bytes = (u64)dc.n_refs * layout->stride;
            int ref_ok = 1;
            if (bytes > ref_cap) { free(ref_buf); ref_buf = (u8*)malloc(bytes); ref_cap = ref_buf ? bytes : 0; }
            if (ref_buf || !bytes) {
                for (u32 i = 0; i < dc.n_refs && ref_ok; i++)
                    if (!rsx_vertex_fetch_one(&dc.fetch_plan, &dc.refs[i],
                                              layout->stride ? ref_buf + (u64)i * layout->stride : NULL))
                        ref_ok = 0;
                draws++;
                unsigned long nb = 0;
                if (ok == ref_ok && ok)
                    for (u64 k = 0; k < bytes; k++) nb += dc.verts[k] != ref_buf[k];
                if (ok != ref_ok || nb) {
                    bad_draws++; bad_bytes += nb;
                    if (bad_draws <= 8)
                        fprintf(stderr, "[vfetch-check] f%u MISMATCH: fetch_all ok=%d fetch_one ok=%d, %lu of %llu bytes differ (verts=%u stride=%u)\n",
                                g.frames, ok, ref_ok, nb, (unsigned long long)bytes, dc.n_refs, layout->stride);
                }
                if ((draws % 20000) == 0)
                    fprintf(stderr, "[vfetch-check] %lu draws compared, %lu mismatched (%lu bytes)\n", draws, bad_draws, bad_bytes);
            }
        }
        if (!ok) {
            dc.fetch_ok = 0;
            return;
        }
    }
    dc.n_verts = dc.n_refs;
}

/* ---- sink ---------------------------------------------------------------- */

static void sink_begin(void* user, const rsx_dispatch* r, u32 prim)
{
    (void)user; (void)r; (void)prim;
    dc_reset();
}

static void sink_draw_arrays(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dc.n_packets++;
    if (dc.n_arr >= ENG_MAX_BATCHES) return;
    dc.arr[dc.n_arr].first = first;
    dc.arr[dc.n_arr].count = count;
    dc.n_arr++;
}

/* INLINE_ARRAY delivers one finished stream rather than a (first, count)
 * range; queue it as an ordinary consecutive batch and let the fetch plan
 * read the vertices out of it (rsx_vertex_fetch_plan_set_inline). */
static void sink_inline_array(void* user, const rsx_dispatch* r,
                              const u8* data, u32 bytes)
{
    (void)user;
    const u32 stride = rsx_vertex_inline_layout(r, NULL);
    if (!stride || bytes < stride) return;
    dc.inl = data;
    dc.inl_bytes = bytes;
    dc.n_packets++;
    if (dc.n_arr >= ENG_MAX_BATCHES) return;
    dc.arr[dc.n_arr].first = 0;
    dc.arr[dc.n_arr].count = bytes / stride;
    dc.n_arr++;
}

static void sink_draw_index(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dc.n_packets++;
    if (dc.n_idx >= ENG_MAX_BATCHES) return;
    dc.idx[dc.n_idx].first = first;
    dc.idx[dc.n_idx].count = count;
    dc.n_idx++;
}

/* Which texture units this draw binds, and from where: a unit naming a
 * registered colour surface samples that live target rather than guest
 * memory (every post-process, shadow and reflection in a PS3 title is a
 * render-to-texture read back through a texture unit), a unit naming a
 * tracked zeta in DEPTH24_D8 samples its snapshot, and everything else is a
 * guest upload (rsx_live_draw.c:5986-6062). */
static int s_zeta_alias_log;   /* print the next pipeline's fragment constants */
static void eng_surface_dump_now(void);   /* every registered surface, now */
static const char* s_dump_tag = "";        /* file-name suffix for the dumps above */
/* RSX_SNAP_STATS=1: every 300 frames, how many draws sampled the target they
 * write (each a full copy of it), how many texture-cache invalidations
 * (NV4097 0x1FD8) the title sent, and how many of those copies came with no
 * invalidation since the previous copy of the same target. */
static int s_snap_stats = -1;
static unsigned long s_snap_n, s_snap_noinval, s_inval_n[4], s_snap_unused;
static u32 s_snap_last_surface = ~0u; static unsigned long s_inval_at_last_snap = ~0ul, s_inval_total;
static void snap_stats_frame(void)
{
    static u32 last;
    if (g.frames - last < 300) return;
    last = g.frames;
    fprintf(stderr, "[snap-stats] 300 frames: %lu own-target copies (%lu with no texture-cache invalidation since the last), "
            "%lu own-target units the program does not sample (no copy); invalidations arg1 %lu arg2 %lu arg3 %lu other %lu\n",
            s_snap_n, s_snap_noinval, s_snap_unused, s_inval_n[1], s_inval_n[2], s_inval_n[3], s_inval_n[0]);
    s_snap_n = s_snap_noinval = s_snap_unused = 0; s_inval_n[0] = s_inval_n[1] = s_inval_n[2] = s_inval_n[3] = 0;
}
static u32 sink_bind_textures(const u32* target_slots, u32 n_targets,
                              u32 current_zslot,
                              u32 textures[RSX_BE_MAX_TEXTURES],
                              rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES])
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        if (!t.enabled) continue;
        mask |= 1u << u;
        eng_decode_sampler(t.filter, t.wrap, t.control0, &samplers[u]);

        int sampled = -1, unused_own = 0;
        for (u32 i = 0; i < g.n_surfaces; i++) {
            if (!g.surfaces[i].handle || g.surfaces[i].location != t.location ||
                g.surfaces[i].offset != t.offset)
                continue;
            /* Not one this draw writes: reading a target a pass is writing
             * is undefined on every API, and an MRT set writes more than
             * one of them. */
            int own = 0;
            for (u32 k = 0; k < n_targets; k++) if (target_slots[k] == i) own = 1;
            if (own) {
                /* A unit left enabled from an earlier pass that this
                 * program never samples: nothing to copy, and a full copy
                 * of a 4K target is ~0.1 ms of GPU. (Drakengard 3's ~95
                 * own-target copies a frame in its destruction scenes are
                 * not this case -- each of those programs samples the scene
                 * at an offset, heat haze, and needs the copy.)
                 * RSX_SNAP_ALL=1 copies regardless. */
                { static int all = -1; if (all < 0) all = getenv("RSX_SNAP_ALL") ? 1 : 0;
                  const u8* vpu; u32 vpn; const u8* fpu = NULL; u32 fpn = 0;
                  const int got = !all && eng_guest_programs(&vpu, &vpn, &fpu, &fpn);
                  if (got && fpu &&
                      !((rsx_fp_texture_mask(fpu, fpn) >> u) & 1u)) {
                      if (s_snap_stats > 0) s_snap_unused++;
                      unused_own = 1;
                      break;
                  } }
                /* The draw samples the target it writes. Drakengard 3's
                 * post-process runs in place in the back buffer it then
                 * flips; binding the live target is undefined, and the old
                 * fallback -- the guest's stale bytes -- came out white on
                 * every such frame. Sample a copy taken at this point. */
                if (s_snap_stats < 0) s_snap_stats = getenv("RSX_SNAP_STATS") ? 1 : 0;
                if (s_snap_stats) {
                    s_snap_n++;
                    if (s_snap_last_surface == i && s_inval_at_last_snap == s_inval_total) s_snap_noinval++;
                    s_snap_last_surface = i; s_inval_at_last_snap = s_inval_total;
                    snap_stats_frame();
                }
                const u32 snap = g.be->color_snapshot
                    ? g.be->color_snapshot(g.be->user, g.surfaces[i].handle) : 0;
                if (snap) {
                    const u32 view = g.be->surface_view
                        ? g.be->surface_view(g.be->user, snap, t.remap & 0xFFFFu, t.format) : 0;
                    textures[u] = view ? view : snap;
                    { static int _n = 0; if (_n++ < 4)
                        fprintf(stderr, "[rsx engine] unit %u samples its own target s%u: bound a copy\n", u, i); }
                }
                break;
            }
            sampled = (int)i;
            break;
        }
        if (unused_own) continue;   /* bound to nothing: the program never reads it */
        if (textures[u]) continue;
        if (sampled >= 0) {
            const u32 view = g.be->surface_view
                ? g.be->surface_view(g.be->user, g.surfaces[sampled].handle,
                                     t.remap & 0xFFFFu, t.format)
                : 0;
            textures[u] = view ? view : g.surfaces[sampled].handle;
            continue;
        }
        const u32 base_fmt = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        /* A non-depth format over a tracked zeta: the title reads its depth
         * buffer's bytes as colour (UE3 decodes D24 from A8R8G8B8 with a dot
         * product). Report each (offset, format) once. */
        if (base_fmt != RSX_TEX_FMT_DEPTH24_D8 && base_fmt != 0x92u) {
            for (u32 i = 0; i < g.n_zdepths; i++)
                if (g.zdepths[i].location == t.location && g.zdepths[i].offset == t.offset) {
                    static u32 seen[16][2]; static u32 nseen = 0; int dup = 0;
                    for (u32 k = 0; k < nseen; k++) if (seen[k][0] == t.offset && seen[k][1] == t.format) dup = 1;
                    /* Bind the depth packed as those bytes, through the unit's
                     * own remap, when it is A8R8G8B8. The packed image is a
                     * copy, so this also works for the zeta the draw has bound:
                     * UE3's shadow projection keeps the scene depth attached
                     * for its stencil test while it reads that depth back,
                     * and skipping the zeta-in-use case handed it the guest
                     * bytes (all zero), so it rebuilt every receiver at the
                     * near plane and no character cast a shadow.
                     * RSX_NO_DEPTH_AS_COLOR=1 leaves the guest bytes. */
                    { static int off = -1; if (off < 0) off = getenv("RSX_NO_DEPTH_AS_COLOR") ? 1 : 0;
                      if (!off && base_fmt == 0x85u) {
                          const u32 pk = eng_zdepth_packed(i, t.width, t.height);
                          if (pk) {
                              const u32 view = g.be->surface_view
                                  ? g.be->surface_view(g.be->user, pk, t.remap & 0xFFFFu, t.format) : 0;
                              textures[u] = view ? view : pk;
                          }
                      } }
                    if (!dup && nseen < 16) { seen[nseen][0] = t.offset; seen[nseen][1] = t.format; nseen++;
                        fprintf(stderr, "[rsx engine] unit %u reads zeta %u (loc %u off 0x%08X %ux%u) as format 0x%02X %ux%u remap 0x%04X -- depth as colour\n",
                                u, i, t.location, t.offset, g.zdepths[i].w, g.zdepths[i].h, t.format, t.width, t.height, t.remap & 0xFFFFu);
                        s_zeta_alias_log = 1; }
                    break;
                }
        }
        if (textures[u]) continue;
        if (base_fmt == RSX_TEX_FMT_DEPTH24_D8) {
            int found = 0;
            for (u32 i = 0; i < g.n_zdepths; i++)
                if (g.zdepths[i].location == t.location &&
                    g.zdepths[i].offset == t.offset && current_zslot != i) {
                    found = 1;
                    const u32 snap = eng_zdepth_snapshot(i, t.width, t.height);
                    if (snap) textures[u] = snap;
                    else { static int n = 0; if (n++ < 6)
                        fprintf(stderr, "[rsx engine] depth texture unit %u (loc %u off 0x%08X %ux%u): tracked zeta %u has no snapshot (handle %u had_write %u)\n",
                                u, t.location, t.offset, t.width, t.height, i, g.zdepths[i].handle, g.zdepths[i].had_write); }
                    break;
                }
            if (!found) { static int n = 0; if (n++ < 6) {
                fprintf(stderr, "[rsx engine] depth texture unit %u (loc %u off 0x%08X %ux%u zfunc %u): no tracked zeta matches; zetas:",
                        u, t.location, t.offset, t.width, t.height, (t.wrap >> 28) & 0xFu);
                for (u32 i = 0; i < g.n_zdepths; i++) fprintf(stderr, " %u:%u/0x%08X/%ux%u%s", i, g.zdepths[i].location, g.zdepths[i].offset, g.zdepths[i].w, g.zdepths[i].h, current_zslot == i ? "(bound)" : "");
                fputc('\n', stderr); } }
            if (textures[u]) continue;
        }
        textures[u] = eng_texture_slot(t.location, t.offset, t.format,
                                       t.width, t.height, t.mipmaps, t.pitch,
                                       (int)t.cubemap, t.remap & 0xFFFFu);
    }
    return mask;
}

static u32 sink_bind_vertex_textures(
    u32 vtex_mask, u32 textures[RSX_BE_MAX_VERTEX_TEXTURES],
    rsx_be_sampler_desc samplers[RSX_BE_MAX_VERTEX_TEXTURES])
{
    u32 bound = 0;
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        if (!((vtex_mask >> u) & 1u)) continue;
        rsx_dsp_vertex_texture vt;
        rsx_dsp_get_vertex_texture(&g.rsx, u, &vt);
        eng_decode_sampler(vt.filter, vt.wrap, vt.control0, &samplers[u]);
        /* A vertex unit has no crossbar, so its remap is the identity. */
        textures[u] = eng_texture_slot(vt.location, vt.offset, vt.format,
                                       vt.width, vt.height, vt.mipmaps,
                                       vt.pitch, 0, 0xAAE4u);
        if (textures[u]) bound |= 1u << u;
    }
    return bound;
}

/* RSX_DRAW_STATS=1: every 120 frames, how many draws reached the backend and
 * how many were dropped at each silent return below (a dropped draw is an
 * object missing from the frame, and until now nothing said so).
 * RSX_DRAW_TRACE_FRAME=<n>: every draw of frame n in full -- primitive,
 * counts, the enabled vertex attributes (type/size/stride/location/offset),
 * the index array, textures and the outcome. */
static struct { unsigned long issued, drop_topo, drop_fetch, drop_targets, drop_pipeline, drop_empty;
                unsigned long long refs, verts, vbytes; } s_dstat;
static unsigned s_surf_draws[ENG_MAX_SURFACES];   /* draws per surface since the last present */
/* GPU-skinned colour draws since the last present (bone indices in attribute
 * 7 as unnormalised bytes, colour writes on) and their vertex total: whether a
 * character that vanished from the picture is still being submitted. */
static unsigned s_skin_draws, s_skin_verts;
static int s_dstat_on = -1; static long s_dtrace_frame = -2;
static void eng_draw_stats_tick(void)
{
    if (s_dstat_on < 0) s_dstat_on = getenv("RSX_DRAW_STATS") ? 1 : 0;
    if (s_dtrace_frame == -2) { const char* e = getenv("RSX_DRAW_TRACE_FRAME"); s_dtrace_frame = e ? atol(e) : -1; }
}
uint32_t g_rsx_engine_frame = 0;   /* the present count, for frame-gated logs elsewhere */
uint32_t g_rsx_engine_hitches = 0; /* presents more than 25 ms after the one before */
/* Set by the host: called with a frame's time when RSX_HITCH_LOG reports it
 * (main.cpp hands it to the sampling profiler's slow-frame report). */
void (*g_rsx_hitch_hook)(double frame_ms) = 0;
double g_rsx_frame_gpu_wait_ms = 0.0;   /* a backend's blocking GPU waits since the last present */
double g_rsx_frame_pso_ms = 0.0;        /* a backend's pipeline states built on the walker since then */
static void eng_draw_stats_report(void)
{
    if (s_dstat_on != 1 || (g.frames % 120u) != 0u) return;
    fprintf(stderr, "[draw-stats] frame %u: issued %lu, dropped topo=%lu fetch=%lu targets=%lu pipeline=%lu empty=%lu; "
            "refs %llu, vertices %llu (%llu KB); vcache hit %llu (%llu by the write-watch) miss %llu uncacheable %llu stored %llu (stale %llu), %u entries %zu MB, check mismatches %llu\n",
            g.frames, s_dstat.issued, s_dstat.drop_topo, s_dstat.drop_fetch, s_dstat.drop_targets,
            s_dstat.drop_pipeline, s_dstat.drop_empty, s_dstat.refs, s_dstat.verts, s_dstat.vbytes >> 10,
            s_vcstat.hit, s_vcstat.hit_watch, s_vcstat.miss, s_vcstat.uncacheable, s_vcstat.stored, s_vcstat.stale, s_vc_count, s_vc_bytes >> 20,
            s_vcstat.check_bad);
    memset(&s_vcstat, 0, sizeof s_vcstat);
    memset(&s_dstat, 0, sizeof s_dstat);
}
/* Retroactive trace: with RSX_TRACE_ON_WHITE set, every draw and clear is
 * also formatted into a ring of the last 512 records, and a present that
 * reads back white dumps the ring -- the draws that PRODUCED the white
 * buffer, which a trace armed after the fact can never show. */
#define ENG_RING_N 512
static char s_ring[ENG_RING_N][400];
static u32  s_ring_head = 0, s_ring_used = 0;
static int  s_ring_on = -1;
static void eng_ring_push(const char* line)
{
    size_t n = strlen(line); if (n >= sizeof s_ring[0]) n = sizeof s_ring[0] - 1;
    memcpy(s_ring[s_ring_head], line, n); s_ring[s_ring_head][n] = 0;
    s_ring_head = (s_ring_head + 1) % ENG_RING_N; if (s_ring_used < ENG_RING_N) s_ring_used++;
}
static void eng_ring_dump(const char* why)
{
    fprintf(stderr, "[ring] ---- %s: last %u records ----\n", why, s_ring_used);
    for (u32 i = 0; i < s_ring_used; i++)
        fprintf(stderr, "[ring] %s\n", s_ring[(s_ring_head + ENG_RING_N - s_ring_used + i) % ENG_RING_N]);
    fprintf(stderr, "[ring] ---- end ----\n");
}
static void eng_ring_draw(const char* outcome, u32 prim, int indexed, u32 n_draw, u32 pipeline, u32 tex_mask)
{
    if (s_ring_on < 0) s_ring_on = (getenv("RSX_TRACE_ON_WHITE") || getenv("RSX_TRACE_ON_COLOR")) ? 1 : 0;
    if (!s_ring_on) return;
    rsx_be_render_state trs; rsx_draw_engine_decode_render_state(&g.rsx, &trs);
    u32 tt[RSX_BE_MAX_COLOR_TARGETS]; const u32 tn = eng_current_target_set(tt);
    rsx_dsp_surface tsf; rsx_dsp_get_surface(&g.rsx, &tsf);
    char line[400]; int o = snprintf(line, sizeof line,
        "f%u %s prim=%u verts=%u idx=%d n=%u pipe=%u tex=0x%X target=s%d(fmt%u) zeta=0x%08X clip=%ux%u blend=%u(%X/%X) alpha=%u(f%X ref=0x%X fmt%u -> %.4g) depth=%u/%u/%X cull=%u mask=0x%X",
        g.frames, outcome, prim, dc.n_verts, indexed, n_draw, pipeline, tex_mask,
        tn ? (int)tt[0] : -1, tn ? (u32)g.surfaces[tt[0]].fmt : 0, tsf.zeta_offset, tsf.clip_w, tsf.clip_h,
        trs.blend_enable, trs.sf_rgb, trs.df_rgb, trs.alpha_test_enable, trs.alpha_func, trs.alpha_ref_raw, trs.alpha_ref_format, (double)rsx_fp_alpha_ref(trs.alpha_ref_raw, trs.alpha_ref_format), trs.depth_test, trs.depth_write, trs.depth_func,
        trs.cull_enable, trs.color_mask);
    if (dc.n_verts && dc.n_verts <= 4 && dc.out_verts && dc.layout.stride && o > 0 && o < (int)sizeof line - 40) {
        const float* v = (const float*)dc.out_verts;
        for (u32 k = 0; k < 8 && k < dc.layout.stride / 4 && o < (int)sizeof line - 12; k++)
            o += snprintf(line + o, sizeof line - o, "%s%.3g", k ? " " : " v0=", v[k]);
    }
    eng_ring_push(line);
}
static void eng_draw_trace(const char* outcome, u32 prim, int indexed, u32 n_draw, u32 pipeline, u32 tex_mask)
{
    eng_ring_draw(outcome, prim, indexed, n_draw, pipeline, tex_mask);
    /* RSX_TRACE_SKINNED_FROM=<frame>: from that frame on, the first 40 draws
     * into a full-size target that carry an unnormalised-byte attribute (a
     * GPU-skinned mesh's bone indices), whatever frame they fall in. */
    int skinned_pick = 0;
    { static long from = -2; static int left = 40;
      if (from == -2) { const char* e = getenv("RSX_TRACE_SKINNED_FROM"); from = e ? atol(e) : -1; }
      if (from >= 0 && (long)g.frames >= from && left > 0) {
          u32 tt[RSX_BE_MAX_COLOR_TARGETS]; const u32 tn = eng_current_target_set(tt);
          if (tn && g.surfaces[tt[0]].w >= 1280) {
              /* Bone indices ride in attribute 7 as unnormalised bytes; a rigid
               * mesh's packed tangents are unnormalised bytes too, in 2 and 5. */
              rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(&g.rsx, 7, &a);
              if (a.type == 7) { skinned_pick = 1; left--; } } } }
    /* Four consecutive frames: a title that presents each rendered frame
     * twice has every other frame empty of draws. */
    if (!skinned_pick && (s_dtrace_frame < 0 || (long)g.frames < s_dtrace_frame || (long)g.frames >= s_dtrace_frame + 4)) return;
    rsx_dsp_index_array ia; rsx_dsp_get_index_array(&g.rsx, &ia);
    rsx_be_render_state trs; rsx_draw_engine_decode_render_state(&g.rsx, &trs);
    u32 ttargets[RSX_BE_MAX_COLOR_TARGETS]; const u32 tnt = eng_current_target_set(ttargets);
    fprintf(stderr, "[draw-trace] f%u %s prim=%u packets=%u refs=%u verts=%u indexed=%d n=%u pipe=%u tex=0x%X idx(loc=%u off=0x%08X u32=%d)"
            " target=s%d(%ux%u fmt%u) blend=%u(%X/%X eq%X) alpha=%u(f%X ref=0x%X fmt%u -> %.4g) depth=%u/%u cull=%u mask=0x%X",
            g.frames, outcome, prim, dc.n_packets, dc.n_source_refs, dc.n_verts, indexed, n_draw, pipeline, tex_mask,
            ia.location, ia.offset, ia.is_u32,
            tnt ? (int)ttargets[0] : -1, tnt ? g.surfaces[ttargets[0]].w : 0, tnt ? g.surfaces[ttargets[0]].h : 0,
            tnt ? (u32)g.surfaces[ttargets[0]].fmt : 0,
            trs.blend_enable, trs.sf_rgb, trs.df_rgb, trs.eq_rgb, trs.alpha_test_enable, trs.alpha_func, trs.alpha_ref_raw, trs.alpha_ref_format, (double)rsx_fp_alpha_ref(trs.alpha_ref_raw, trs.alpha_ref_format), trs.depth_test, trs.depth_write,
            trs.cull_enable, trs.color_mask);
    /* The first decoded vertex of a small draw: a full-screen quad's position
     * and its vertex colour are what decide whether it covers the frame. */
    if (dc.n_verts && dc.n_verts <= 8 && dc.out_verts && dc.layout.stride) {
        /* Every vertex of a small draw, first 8 floats each: a quad's four
         * corners and colours say whether it covers the frame and with what. */
        for (u32 n = 0; n < dc.n_verts; n++) {
            const float* v = (const float*)(dc.out_verts + (size_t)n * dc.layout.stride);
            fprintf(stderr, " v%u=[", n);
            for (u32 k = 0; k < dc.layout.stride / 4 && k < 8; k++) fprintf(stderr, "%s%.3g", k ? " " : "", v[k]);
            fprintf(stderr, "]");
        }
    }
    fprintf(stderr, " attrs:");
    for (u32 i = 0; i < 16; i++) {
        rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(&g.rsx, i, &a);
        if (!a.type) continue;
        fprintf(stderr, " %u:t%u/%u/s%u/L%u/0x%08X", i, a.type, a.size, a.stride, a.location, a.offset);
    }
    /* For a quad-sized draw (a movie, a post-process or UI pass), each bound
     * texture: location, offset, format, size, and the mean of its first
     * row -- whether the planes a video quad samples hold any picture. */
    static int tex_all = -1; if (tex_all < 0) tex_all = getenv("RSX_TRACE_TEXUNITS") ? 1 : 0;
    if (dc.n_verts && (dc.n_verts <= 8 || tex_all)) {
        fprintf(stderr, " texunits:");
        for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
            if (!(tex_mask & (1u << u))) continue;
            rsx_dsp_texture t; rsx_dsp_get_texture(&g.rsx, u, &t);
            const u32 row = t.pitch ? t.pitch : t.width;
            const u8* p = (t.width && row) ? eng_guest_ptr(NULL, t.location, t.offset, row) : NULL;
            unsigned long sum = 0; for (u32 k = 0; p && k < row && k < 4096; k++) sum += p[k];
            int surf = -1, zeta = -1;
            for (u32 k = 0; k < g.n_surfaces; k++) if (g.surfaces[k].handle && g.surfaces[k].location == t.location && g.surfaces[k].offset == t.offset) surf = (int)k;
            for (u32 k = 0; k < g.n_zdepths; k++) if (g.zdepths[k].handle && g.zdepths[k].location == t.location && g.zdepths[k].offset == t.offset) zeta = (int)k;
            if (surf >= 0) fprintf(stderr, " [surface s%d fmt%u]", surf, (u32)g.surfaces[surf].fmt);
            if (zeta >= 0) fprintf(stderr, " [zeta z%d]", zeta);
            fprintf(stderr, " %u:L%u/0x%08X/f%02X/%ux%u/p%u/mips%u/filt%08X/ctl%08X/remap%04X/addr%08X/ea%08X/m%lu", u, t.location, t.offset, t.format, t.width, t.height, t.pitch, t.mipmaps, t.filter, t.control0, t.remap & 0xFFFFu, t.wrap,
                    cellGcmResolveLocated(t.location == RSX_LOCATION_LOCAL, t.offset),
                    p ? sum / (row < 4096 ? row : 4096) : 9999ul);
        }
    }
    fputc('\n', stderr);
    /* A GPU-skinned draw (an unnormalised-byte attribute carries its bone
     * indices): the constants Drakengard 3's skinning program reads -- bone
     * rows from c207, the unpack factors in c463, position scale/offset in
     * c465/c466 -- as the dispatcher holds them for this draw. */
    { int skinned = 0;
      for (u32 i = 0; i < 16 && !skinned; i++) { rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(&g.rsx, i, &a); if (a.type == 7) skinned = 1; }
      if (skinned) {
          static const u32 slots[] = { 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 429, 430, 431, 432, 462, 463, 464, 465, 466, 467 };
          { const u8* vu = NULL; const u8* fu = NULL; u32 vi2 = 0, fs2 = 0;
            u64 vf = 1469598103934665603ull;
            if (eng_guest_programs(&vu, &vi2, &fu, &fs2))
                for (u32 i = 0; i < vi2 * 16u; i++) vf = (vf ^ vu[i]) * 1099511628211ull;
            u64 pf = 0, ph2 = 0;
            for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { pf = g.pipelines[i].vp_fnv; ph2 = g.pipelines[i].vp_hlsl; break; }
            fprintf(stderr, "[draw-trace]   vp resident=%016llx (start=%u instrs=%u) pipeline's=%016llx %s vp-hlsl=%016llx\n",
                    (unsigned long long)vf, rsx_dsp_vp_start(&g.rsx), vi2, (unsigned long long)pf, vf == pf ? "same" : "DIFFERENT",
                    (unsigned long long)ph2); }
          fprintf(stderr, "[draw-trace]   constants:");
          for (u32 k = 0; k < sizeof slots / sizeof slots[0]; k++) {
              const float* c = (const float*)g.rsx.constants[slots[k]];
              fprintf(stderr, " c%u=(%.3g %.3g %.3g %.3g)", slots[k], c[0], c[1], c[2], c[3]); }
          fputc('\n', stderr); } }
}
static void sink_end_impl(void* user, const rsx_dispatch* r);
/* Timed, for the slow-frame report: how much of a frame the walker spent
 * issuing draws (conversion, hashing, cache, records). */
static void sink_end(void* user, const rsx_dispatch* r)
{
    const double t0 = eng_now_ms();
    sink_end_impl(user, r);
    s_fstat.draws++; s_fstat.draw_ms += eng_now_ms() - t0;
}
static void sink_end_impl(void* user, const rsx_dispatch* r)
{
    (void)user; (void)r;
    if (!g.ready || !dc.n_packets) return;
    eng_draw_stats_tick();

    const u32 prim = g.rsx.current_primitive;
    /* Everything that becomes triangles -- lists, strips, fans, quads, quad
     * strips and polygons -- is REBUILT into one triangle list through an
     * index buffer. A host strip topology cannot express a restart cut, which
     * is the whole reason the reference engine rebuilds them, and rebuilding
     * is also what lets repeated references share an uploaded vertex. Points
     * and lines pass through as the guest issued them, and anything left that
     * would need an expansion this engine does not have is dropped rather
     * than drawn as something else. */
    rsx_topology topology = RSX_TOPOLOGY_TRIANGLES;
    int scratch = 0;
    const int rebuild = eng_topology_rebuild(prim, 0, &scratch) != 0;
    if (!rebuild) {
        if (rsx_primitive_needs_expansion(prim)) { s_dstat.drop_topo++; eng_draw_trace("DROP-topology", prim, 0, 0, 0, 0); return; }
        topology = rsx_primitive_topology(prim);
        if (topology == RSX_TOPOLOGY_UNSUPPORTED) { s_dstat.drop_topo++; eng_draw_trace("DROP-topology", prim, 0, 0, 0, 0); return; }
    }

    rsx_vertex_layout_plan layout;
    eng_vertex_layout(&layout);

    /* A cache hit supplies the converted vertices and the rebuilt index list;
     * a miss converts and, once the indices are written, stores both. Under
     * RSX_VCACHE_CHECK a hit is converted anyway and compared at that point. */
    const u64 vc_key = eng_vc_key(&layout, prim, rebuild);
    EngVCEntry* vce = vc_key ? eng_vc_lookup(vc_key) : NULL;
    EngVCEntry* vc_check = NULL;
    if (vce && eng_vc_checking()) { vc_check = vce; vce = NULL; }
    EngVCFill vcf;
    vcf.ok = vc_key != 0;
    vcf.content = vc_key;
    vcf.n_spans = 0;
    const u8* draw_verts = NULL;
    const u32* draw_indices = NULL;
    int indexed = 0;
    u32 n_draw = 0;
    if (vce) {
        s_vcstat.hit++;
        dc.fetch_ok = 1;
        dc.n_verts = vce->n_verts;
        dc.n_source_refs = vce->n_source_refs;
        indexed = vce->indexed;
        n_draw = vce->n_draw;
        draw_verts = vce->verts;
        draw_indices = vce->indices;
        dc.out_verts = draw_verts;
        dc.layout = layout;
    } else {
        if (vc_key) s_vcstat.miss++; else s_vcstat.uncacheable++;
        /* RSX_VCACHE_LOG_MISS=<frame>: every miss of that frame. */
        { static long lf = -2; if (lf == -2) { const char* e = getenv("RSX_VCACHE_LOG_MISS"); lf = e ? atol(e) : -1; }
          if (lf >= 0 && (long)g.frames == lf) {
              rsx_dsp_index_array ia; rsx_dsp_get_index_array(&g.rsx, &ia);
              char buf[512]; int o = snprintf(buf, sizeof buf, "[vc-miss] prim %u arr %u idx %u", prim, dc.n_arr, dc.n_idx);
              if (dc.n_idx) o += snprintf(buf + o, sizeof buf - o, " ia %u:%08X first %u count %u base %u", ia.location, ia.offset, dc.idx[0].first, dc.idx[0].count, rsx_dsp_vertex_data_base_index(&g.rsx));
              if (dc.n_arr) o += snprintf(buf + o, sizeof buf - o, " arr first %u count %u", dc.arr[0].first, dc.arr[0].count);
              o += snprintf(buf + o, sizeof buf - o, " mask %04X baseoff %08X", layout.mask, rsx_dsp_vertex_data_base_offset(&g.rsx));
              for (u32 a = 0; a < 16 && o < (int)sizeof buf - 40; a++) if ((layout.mask >> a) & 1u) {
                  rsx_dsp_vertex_attr va; rsx_dsp_get_vertex_attr(&g.rsx, a, &va);
                  o += snprintf(buf + o, sizeof buf - o, " a%u=%u:%08X/%u/t%u", a, va.location, va.offset, va.stride, va.type); }
              fprintf(stderr, "%s\n", buf); } }
        dc_fetch(&layout, rebuild, vc_key ? &vcf : NULL);
        draw_verts = dc.verts;
        dc.out_verts = draw_verts;
    }
    if (s_dstat_on == 1 && dc.fetch_ok) {
        s_dstat.refs += dc.n_source_refs; s_dstat.verts += dc.n_verts;
        s_dstat.vbytes += (unsigned long long)dc.n_verts * layout.stride;
    }
    if (!dc.n_verts || !dc.fetch_ok) { s_dstat.drop_fetch++; eng_draw_trace(dc.fetch_ok ? "DROP-noverts" : "DROP-fetch", prim, 0, 0, 0, 0); return; }

    if (!vce) {
        n_draw = dc.n_source_refs;
        if (rebuild) {
            if (!eng_topology_rebuild(prim, dc.refs_remapped, &indexed)) { s_dstat.drop_topo++; eng_draw_trace("DROP-rebuild", prim, 0, 0, 0, 0); return; }
            n_draw = indexed
                ? rsx_draw_engine_topology_index_count(prim, dc.n_source_refs,
                                                       dc.cuts, dc.n_cuts)
                : dc.n_source_refs - dc.n_source_refs % 3u;
        }
    }
    if (!n_draw) { s_dstat.drop_empty++; eng_draw_trace("DROP-empty", prim, indexed, 0, 0, 0); return; }
    if (g.q_cur) g.q_attempts++;

    rsx_be_render_state rs;
    rsx_draw_engine_decode_render_state(&g.rsx, &rs);
    /* RSX_SKIP_BLEND=<sf>/<df> (hex blend factors): drop every blended draw
     * using that pair -- an experiment switch to tell whether one blend mode's
     * draws are what wrecks a frame. */
    { static long sk_sf = -2, sk_df = 0;
      if (sk_sf == -2) { const char* e = getenv("RSX_SKIP_BLEND"); sk_sf = -1;
          if (e) { char* d; sk_sf = strtol(e, &d, 16); if (*d == '/') sk_df = strtol(d + 1, 0, 16); } }
      if (sk_sf >= 0 && rs.blend_enable && (long)rs.sf_rgb == sk_sf && (long)rs.df_rgb == sk_df) {
          static unsigned long nskip = 0; if (++nskip <= 4 || (nskip % 1000) == 0)
              fprintf(stderr, "[rsx engine] RSX_SKIP_BLEND: dropped draw #%lu (blend %lX/%lX)\n", nskip, sk_sf, sk_df);
          return; } }

    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    const u32 n_targets = eng_current_target_set(targets);
    if (!n_targets) { s_dstat.drop_targets++; eng_draw_trace("DROP-targets", prim, indexed, n_draw, 0, 0); return; }
    const u32 target = targets[0];

    rsx_dsp_surface sf;
    rsx_dsp_viewport vp;
    rsx_dsp_get_surface(&g.rsx, &sf);
    rsx_dsp_get_viewport(&g.rsx, &vp);
    const u32 zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                                     sf.clip_w, sf.clip_h);

    /* Textures before any pipeline or target binding: a cache refresh can
     * replace a resource, and the backend is free to submit while doing it. */
    u32 textures[RSX_BE_MAX_TEXTURES];
    rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES];
    u32 vtextures[RSX_BE_MAX_VERTEX_TEXTURES];
    rsx_be_sampler_desc vsamplers[RSX_BE_MAX_VERTEX_TEXTURES];
    memset(textures, 0, sizeof textures);
    memset(samplers, 0, sizeof samplers);
    memset(vtextures, 0, sizeof vtextures);
    memset(vsamplers, 0, sizeof vsamplers);
    const u32 tex_mask = sink_bind_textures(targets, n_targets, zslot,
                                            textures, samplers);
    const u32 vtex_mask = sink_bind_vertex_textures(eng_vtex_mask(), vtextures,
                                                    vsamplers);

    /* A guest program pair that will not translate has no fallback: the draw
     * is dropped, as the reference engine drops it. Substituting the built-in
     * program would draw the geometry in the wrong colours, which is harder to
     * see than a missing object and hides the translation failure. */
    int pipeline_is_fixed = 1;
    /* The pipeline's colour attachment format must be the format of the
     * texture that is really bound, which is the one the surface was
     * registered with. A pass that declares another format for the same
     * offset -- Drakengard 3's depth pre-pass binds a display buffer -- would
     * otherwise get a pipeline whose attachment format disagrees with the
     * attachment, and what Metal then renders is undefined. */
    rsx_be_format rt_fmt = eng_surface_format(sf.color_format);
    if (g.surfaces[target].fmt != RSX_BE_FMT_NONE && g.surfaces[target].fmt != rt_fmt) {
        static unsigned long n_mismatch = 0;
        if (++n_mismatch <= 12 || (n_mismatch % 5000) == 0)
            fprintf(stderr, "[rsx engine] surface s%u (off=0x%08X) declared fmt %u but registered as %u -- using the registered one (#%lu, mask=0x%X)\n",
                    target, g.surfaces[target].offset, (unsigned)rt_fmt, (unsigned)g.surfaces[target].fmt,
                    n_mismatch, rs.color_mask);
        rt_fmt = g.surfaces[target].fmt;
    }
    const u32 pipeline = eng_pipeline_get(&layout, &rs, rt_fmt,
                                          n_targets, &pipeline_is_fixed);
    if (!pipeline) { s_dstat.drop_pipeline++; eng_draw_trace("DROP-pipeline", prim, indexed, n_draw, 0, tex_mask); return; }
    /* RSX_SURF_DUMP_AT_FP=<hash>: the first time (from RSX_SURF_DUMP_FROM) a
     * draw with that fragment program is about to run, finish the GPU work
     * recorded so far and dump every surface -- what the draw will read. */
    /* RSX_SURF_DUMP_AT_FP_EVERY=1: once per frame instead of once per run. */
    { static int armed = -1, every = 0; static unsigned long long want = 0; static u32 last_frame = ~0u;
      if (armed < 0) { const char* e = getenv("RSX_SURF_DUMP_AT_FP"); want = e ? strtoull(e, 0, 16) : 0; armed = want ? 1 : 0;
                       every = getenv("RSX_SURF_DUMP_AT_FP_EVERY") ? 1 : 0; }
      if (armed == 1 && (!every || last_frame != g.frames)) {
          const char* e = getenv("RSX_SURF_DUMP_FROM"); const long from = e ? atol(e) : 0;
          u64 fh = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
          if (fh == want && (long)g.frames >= from) {
              if (every) last_frame = g.frames; else armed = 0;
              if (g.be->submit_and_wait) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
              fprintf(stderr, "[surf-dump] before the first draw with fp-hlsl %016llx in frame %u\n", want, g.frames);
              s_dump_tag = "_pre"; eng_surface_dump_now(); s_dump_tag = "";
          } } }
    /* RSX_SKIP_FP=<hash>[,<hash>...]: drop every draw whose fragment program
     * translates to that HLSL hash (the fp-hlsl the pipeline log prints) --
     * to find which pass produces an artefact. */
    { static int n = -1; static unsigned long long want[16];
      if (n < 0) { n = 0; const char* e = getenv("RSX_SKIP_FP");
          while (e && *e && n < 16) { char* d; want[n++] = strtoull(e, &d, 16); e = (*d == ',') ? d + 1 : NULL; } }
      if (n) { u64 fh = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
          for (int k = 0; k < n; k++) if (fh && fh == want[k]) return; } }

    if (indexed && !vce) {
        if (!dc_reserve_indices(n_draw)) return;
        rsx_draw_engine_write_topology_indices(
            prim, dc.n_source_refs, dc.cuts, dc.n_cuts,
            dc.refs_remapped ? dc.ref_remap.occurrence_to_unique : NULL,
            g.indices);
        draw_indices = g.indices;
    }
    if (vc_check) {
        const int same = vc_check->n_verts == dc.n_verts && vc_check->stride == layout.stride &&
            vc_check->indexed == indexed && vc_check->n_draw == n_draw &&
            !memcmp(vc_check->verts, dc.verts, (size_t)dc.n_verts * layout.stride) &&
            (!indexed || !memcmp(vc_check->indices, g.indices, (size_t)n_draw * sizeof(u32)));
        s_vc_checked++;
        if (!same) s_vc_check_bad_total++;
        if (!same && ++s_vcstat.check_bad <= 8)
            fprintf(stderr, "[vcache-check] f%u MISMATCH: cached %u verts/%u draw (indexed %d), converted %u/%u (%d)\n",
                    g.frames, vc_check->n_verts, vc_check->n_draw, vc_check->indexed,
                    dc.n_verts, n_draw, indexed);
    } else if (!vce && vcf.ok && dc.fetch_ok) {
        eng_vc_store(vc_key, &vcf, dc.verts, dc.n_verts, layout.stride,
                     dc.n_source_refs, indexed, g.indices, n_draw);
    }

    /* The transform constant block, then the viewport epilogue the vertex
     * program's tail multiplies by: RSX window coordinates mapped into the
     * host's clip space (rsx_live_draw.c:6143-6156). */
    const float W = sf.clip_w ? (float)sf.clip_w : (float)g.width;
    const float H = sf.clip_h ? (float)sf.clip_h : (float)g.height;
    float xf[8] = { 1, 1, 1, 0, 0, 0, 0, 0 };
    if (vp.scale[0] != 0.0f || vp.translate[0] != 0.0f) {
        xf[0] = vp.scale[0] / (W * 0.5f);
        xf[1] = -(vp.scale[1] / (H * 0.5f));
        xf[2] = vp.scale[2];
        xf[4] = (vp.translate[0] - W * 0.5f) / (W * 0.5f);
        xf[5] = -((vp.translate[1] - H * 0.5f) / (H * 0.5f));
        xf[6] = vp.translate[2];
    }
    /* The 8 KB block is copied and staged again only when a constant or the
     * viewport changed since the last draw; otherwise the backend rebinds
     * what it already staged. RSX_VS_CB_ALWAYS=1 stages it every draw. */
    static int vs_cb_always = -1;
    if (vs_cb_always < 0) vs_cb_always = getenv("RSX_VS_CB_ALWAYS") ? 1 : 0;
    const int vs_cb_same = !vs_cb_always && g.vp_cb_valid &&
        g.vp_cb_gen == g.rsx.const_gen &&
        !memcmp(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, xf, sizeof xf);
    if (!vs_cb_same) {
        memcpy(g.vp_cb, g.rsx.constants, RSX_DSP_NUM_CONSTANTS * 16u);
        memcpy(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, xf, sizeof xf);
        g.vp_cb_gen = g.rsx.const_gen;
        g.vp_cb_valid = 1;
        g.vp_cb_bound = 0;
    }

    /* The buffered fragment constants, then fp_alpha: the layout
     * rsx_fp_decompile_buffered_ex compiled the shader against. */
    const u32 nslots = g.fp_constants.count ? g.fp_constants.count : 1u;
    const u32 fp_bytes = (nslots + 1u) * 16u;
    if (g.fp_cb_cap < fp_bytes) {
        u8* n = (u8*)realloc(g.fp_cb, fp_bytes);
        if (!n) return;
        g.fp_cb = n;
        g.fp_cb_cap = fp_bytes;
    }
    memset(g.fp_cb, 0, fp_bytes);
    if (g.fp_constants.count)
        memcpy(g.fp_cb, g.fp_constants.values, g.fp_constants.count * 16u);
    /* RSX_FPCONST_AT_FP=<hash>: the fragment constants of the first 3 draws
     * (from RSX_SURF_DUMP_FROM) whose program translates to that hash. */
    { static int left = -1; static unsigned long long want = 0;
      if (left < 0) { const char* e = getenv("RSX_FPCONST_AT_FP"); want = e ? strtoull(e, 0, 16) : 0; left = want ? 3 : 0; }
      if (left > 0) {
          const char* e = getenv("RSX_SURF_DUMP_FROM"); const long from = e ? atol(e) : 0;
          u64 fh = 0;
          for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
          if (fh == want && (long)g.frames >= from) { left--;
              fprintf(stderr, "[fpconst] f%u fp %016llx:", g.frames, want);
              for (u32 k = 0; k < g.fp_constants.count; k++) {
                  const float* c = (const float*)g.fp_constants.values[k];
                  fprintf(stderr, " c%u=(%.6g %.6g %.6g %.6g)", k, c[0], c[1], c[2], c[3]); }
              fputc('\n', stderr); } } }
    if (s_zeta_alias_log) { s_zeta_alias_log = 0;
        fprintf(stderr, "[rsx engine]   its fragment constants:");
        for (u32 k = 0; k < g.fp_constants.count && k < 6; k++) {
            const float* c = (const float*)g.fp_constants.values[k];
            fprintf(stderr, " c%u=(%.6g %.6g %.6g %.6g)", k, c[0], c[1], c[2], c[3]); }
        fputc('\n', stderr); }
    { float* alpha = (float*)(g.fp_cb + nslots * 16u);
      alpha[0] = rsx_fp_alpha_ref(rs.alpha_ref_raw, rs.alpha_ref_format);
      alpha[1] = alpha[2] = alpha[3] = 0.0f; }

    /* The depth attachment is cleared the first time it is bound, so a title
     * that never clears its shadow zeta still tests against something. */
    const u32 depth_handle = (zslot != ENG_INVALID) ? g.zdepths[zslot].handle : 0;
    if (zslot != ENG_INVALID && !g.zdepths[zslot].cleared) {
        g.be->clear_depth_stencil(g.be->user, depth_handle,
                                  RSX_BE_CLEAR_DEPTH | RSX_BE_CLEAR_STENCIL,
                                  1.0f, 0);
        g.zdepths[zslot].cleared = 1;
        g.zdepths[zslot].had_write = 0;
    }

    u32 handles[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i < n_targets; i++) handles[i] = g.surfaces[targets[i]].handle;
    g.be->bind_targets(g.be->user, handles, n_targets, depth_handle);
    g.be->bind_pipeline(g.be->user, pipeline);
    if (vs_cb_always || !g.vp_cb_bound || !g.be->reuse_vs_constants ||
        !g.be->reuse_vs_constants(g.be->user))
        g.be->bind_vs_constants(g.be->user, g.vp_cb, ENG_VP_CB_BYTES);
    g.vp_cb_bound = 1;
    g.be->bind_ps_constants(g.be->user, g.fp_cb, fp_bytes);
    g.be->bind_textures(g.be->user, textures, samplers, tex_mask);
    g.be->bind_vertex_textures(g.be->user, vtextures, vsamplers, vtex_mask);
    g.be->set_viewport(g.be->user, 0.0f, 0.0f, W, H);

    /* The guest scissor intersected with the surface. The nv40 reset is a
     * full 4096x4096 window and a never-written register reads 0 here, which
     * the w == 0 test treats as no scissor, so only a real game scissor
     * narrows anything (rsx_live_draw.c:6180-6197). */
    {
        u32 sx = 0, sy = 0;
        u32 sw = g.surfaces[target].w, sh = g.surfaces[target].h;
        const u32 h = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
        const u32 v = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
        const u32 gx = h & 0xFFFFu, gw = h >> 16;
        const u32 gy = v & 0xFFFFu, gh = v >> 16;
        if (gw > 0 && gh > 0) {
            u32 right = sx + sw, bottom = sy + sh;
            if (gx > sx) sx = gx;
            if (gy > sy) sy = gy;
            if (gx + gw < right)  right = gx + gw;
            if (gy + gh < bottom) bottom = gy + gh;
            sw = right > sx ? right - sx : 0;
            sh = bottom > sy ? bottom - sy : 0;
        }
        g.be->set_scissor(g.be->user, sx, sy, sw, sh);
    }
    /* Dynamic, and deliberately outside the pipeline key. */
    g.be->set_stencil_ref(g.be->user,
                          rsx_dsp_reg(&g.rsx, M_STENCIL_FUNC_REF) & 0xFFu);

    const u32 uploaded = indexed ? dc.n_verts : n_draw;
    if (g.q_cur && g.be->query_set) { g.be->query_set(g.be->user, g.q_cur); g.q_draws++; }
    if (vce && vce->buf)
        g.be->draw_buffer(g.be->user, topology, vce->buf, 0, uploaded,
                          layout.stride, vce->ib_off, indexed ? n_draw : 0);
    else
        g.be->draw(g.be->user, topology, draw_verts, uploaded, layout.stride,
                   indexed ? draw_indices : NULL, indexed ? n_draw : 0);
    /* Counted only when the draw ran the guest's OWN programs, which is what
     * the test hook means: a draw through the built-in pair is a draw, not
     * evidence that the decompile-translate-compile path worked. */
    if (!pipeline_is_fixed) g.guest_draws++;
    s_dstat.issued++;
    if (target < ENG_MAX_SURFACES) s_surf_draws[target]++;
    if (rs.color_mask) { rsx_dsp_vertex_attr a7; rsx_dsp_get_vertex_attr(&g.rsx, 7, &a7);
        if (a7.type == 7) { s_skin_draws++; s_skin_verts += dc.n_verts; } }
    eng_draw_trace(pipeline_is_fixed ? "OK-fixed" : "OK", prim, indexed, n_draw, pipeline, tex_mask);
    /* RSX_PICK=<x>,<y>,<frame>: after every draw of that frame into a
     * full-size target, finish the GPU work and read pixel (x, y) of the
     * draw's first target back -- which draws touch a pixel, and what each
     * one leaves there. Slow (a GPU sync per draw); for tools/rsx_replay. */
    { static int px = -2, py = 0; static long pf = -1; static u32 last[4]; static u32 last_surf = ENG_INVALID; static u32 idx;
      if (px == -2) { const char* e = getenv("RSX_PICK"); px = -1;
          if (e && sscanf(e, "%d,%d,%ld", &px, &py, &pf) != 3) px = -1; }
      if (px >= 0 && (long)g.frames == pf && target < g.n_surfaces) {
          const eng_surface* ts = &g.surfaces[target];
          idx++;
          /* A smaller target (a downsample) is read at the same place in
           * the picture: the pick scaled to its size. */
          const u32 qx = ts->w >= 1280 ? (u32)px : (u32)((u64)px * ts->w / 1280u);
          const u32 qy = ts->h >= 720 ? (u32)py : (u32)((u64)py * ts->h / 720u);
          if (qx < ts->w && qy < ts->h) {
              if (g.be->submit_and_wait) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
              u8 b[16]; memset(b, 0, sizeof b);
              const u32 bpp = ts->fmt == RSX_BE_FMT_R16G16B16A16F ? 8u : 4u;
              g.be->readback(g.be->user, ts->handle, qx, qy, 1, 1, b, bpp);
              u32 cur[4] = {0, 0, 0, 0}; memcpy(cur, b, bpp);
              if (target != last_surf || memcmp(cur, last, sizeof cur)) {
                  float c[4] = {0, 0, 0, 0};
                  if (bpp == 8) for (int k = 0; k < 4; k++) {
                      const u16 v = ((const u16*)b)[k]; const u32 ex = (v >> 10) & 0x1F, mant = v & 0x3FF;
                      float o = ex == 0 ? mant / 1024.0f / 16384.0f : ex == 31 ? 65504.0f : (1.0f + mant / 1024.0f) * (float)pow(2.0, (int)ex - 15);
                      c[k] = (v & 0x8000) ? -o : o; }
                  else for (int k = 0; k < 4; k++) c[k] = b[k] / 255.0f;
                  u64 fh = 0;
                  for (u32 i = 0; i < g.n_pipelines; i++) if (g.pipelines[i].handle == pipeline) { fh = g.pipelines[i].fp_hlsl; break; }
                  rsx_dsp_surface psf; rsx_dsp_get_surface(&g.rsx, &psf);
                  fprintf(stderr, "[pick] f%u draw %u pipe %u fp %016llx -> s%u(rsxfmt 0x%X engfmt %u) = (%.4g %.4g %.4g %.4g) blend=%u(%X/%X) depth=%u/%u/%X mask=0x%X tex=0x%X verts=%u\n",
                          g.frames, idx, pipeline, (unsigned long long)fh, target, psf.color_format, (u32)ts->fmt, c[0], c[1], c[2], c[3],
                          rs.blend_enable, rs.sf_rgb, rs.df_rgb, rs.depth_test, rs.depth_write, rs.depth_func, rs.color_mask, tex_mask, dc.n_verts);
                  memcpy(last, cur, sizeof cur); last_surf = target;
              }
          }
      } }

    if (zslot != ENG_INVALID && rs.depth_test && rs.depth_write) {
        /* New depth: any snapshot taken before this draw is stale. Only a
         * depth CLEAR used to invalidate them, so a title that never clears
         * a depth buffer between reads (Drakengard 3's battle areas) read a
         * snapshot from long before: the character shadow projection
         * rebuilt every receiver at the near plane, landed outside its
         * shadow map, and no character cast a shadow. */
        g.zdepths[zslot].had_write = 1;
        g.zdepths[zslot].snapshot_valid = 0;
        g.zdepths[zslot].packed_valid = 0;
    }
}

static void sink_clear(void* user, const rsx_dispatch* r, u32 mask)
{
    (void)user; (void)r;
    if (!g.ready) return;
    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    const u32 n_targets = eng_current_target_set(targets);
    if (s_ring_on < 0) s_ring_on = (getenv("RSX_TRACE_ON_WHITE") || getenv("RSX_TRACE_ON_COLOR")) ? 1 : 0;
    if (s_ring_on) { char line[160];
        snprintf(line, sizeof line, "f%u CLEAR mask=0x%X color=0x%08X target=s%d", g.frames, mask,
                 rsx_dsp_clear_color(&g.rsx), n_targets ? (int)targets[0] : -1);
        eng_ring_push(line); }
    if (!n_targets) return;

    if (mask & (RSX_CLEAR_COLOR_R | RSX_CLEAR_COLOR_G |
                RSX_CLEAR_COLOR_B | RSX_CLEAR_COLOR_A)) {
        const u32 c = rsx_dsp_clear_color(&g.rsx);
        const float rgba[4] = {
            (float)((c >> 16) & 0xFF) / 255.0f,
            (float)((c >>  8) & 0xFF) / 255.0f,
            (float)( c        & 0xFF) / 255.0f,
            (float)((c >> 24) & 0xFF) / 255.0f,
        };
        /* Every target the set names, not only A: a deferred pass clears its
         * whole G-buffer in one CLEAR_SURFACE, and B, C and D would otherwise
         * keep whatever the last frame left in them. */
        for (u32 i = 0; i < n_targets; i++)
            g.be->clear_color(g.be->user, g.surfaces[targets[i]].handle, rgba);
    }
    if (mask & (RSX_CLEAR_DEPTH | RSX_CLEAR_STENCIL)) {
        rsx_dsp_surface sf;
        rsx_dsp_get_surface(&g.rsx, &sf);
        const u32 zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                                         sf.clip_w, sf.clip_h);
        if (zslot == ENG_INVALID) return;
        u32 flags = 0;
        if (mask & RSX_CLEAR_DEPTH)   flags |= RSX_BE_CLEAR_DEPTH;
        if (mask & RSX_CLEAR_STENCIL) flags |= RSX_BE_CLEAR_STENCIL;
        /* ZSTENCIL_CLEAR_VALUE is Z24S8: depth24 << 8 | stencil8, with the
         * nv40 reset 0xFFFFFF00 seeded by rsx_dispatch_init, so a stream that
         * never writes it still clears to 1.0 / 0. */
        const u32 zs = rsx_dsp_reg(&g.rsx, M_ZSTENCIL_CLEAR);
        const float zval = zs ? (float)(zs >> 8) / 16777215.0f : 1.0f;
        /* CLEAR_SURFACE honours the scissor. Drakengard 3 packs its shadow
         * maps into one 512x512 depth target and clears each region just
         * before drawing it; cleared whole, the last region's clear wiped the
         * character shadows drawn earlier in the frame whenever the
         * projections came after it, and they flickered in and out. The
         * scissor is the draws' (guest scissor within the surface);
         * RSX_CLEAR_NO_SCISSOR=1 clears the whole target as before. */
        int partial = 0;
        u32 cx = 0, cy = 0, cw = 0, ch = 0;
        { static int off = -1; if (off < 0) off = getenv("RSX_CLEAR_NO_SCISSOR") ? 1 : 0;
          const u32 h = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
          const u32 v = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
          const u32 gx = h & 0xFFFFu, gw = h >> 16, gy = v & 0xFFFFu, gh = v >> 16;
          const u32 sw = sf.clip_w ? sf.clip_w : g.zdepths[zslot].w;
          const u32 sh = sf.clip_h ? sf.clip_h : g.zdepths[zslot].h;
          if (!off && gw > 0 && gh > 0 && g.be->clear_depth_stencil_rect) {
              u32 right = sw, bottom = sh;
              cx = gx; cy = gy;
              if (gx + gw < right)  right = gx + gw;
              if (gy + gh < bottom) bottom = gy + gh;
              cw = right > cx ? right - cx : 0;
              ch = bottom > cy ? bottom - cy : 0;
              /* Whole-target clears keep the load-action path. */
              partial = !(cx == 0 && cy == 0 && cw >= g.zdepths[zslot].w && ch >= g.zdepths[zslot].h);
              if (partial && (!cw || !ch)) return;   /* scissored away entirely */
          } }
        if (partial)
            g.be->clear_depth_stencil_rect(g.be->user, g.zdepths[zslot].handle, flags,
                                           zval, (u8)(zs & 0xFFu), cx, cy, cw, ch);
        else
            g.be->clear_depth_stencil(g.be->user, g.zdepths[zslot].handle, flags,
                                      zval, (u8)(zs & 0xFFu));
        g.zdepths[zslot].cleared = 1;
        if (mask & RSX_CLEAR_DEPTH) {
            /* A clear invalidates the older published depth image; the next
             * texture consumer resolves the newly written pass exactly once.
             * A partial clear leaves the rest of the target's depth in place,
             * so it still counts as written. */
            if (!partial) g.zdepths[zslot].had_write = 0;
            g.zdepths[zslot].snapshot_valid = 0;
            g.zdepths[zslot].packed_valid = 0;
        }
    }
}

/* Resolve a flip's buffer id to a REGISTERED surface rather than whatever is
 * currently bound: the live target at that instant is often an offscreen
 * shadow or post-process surface, and copying it presents black despite the
 * scene's draws having executed (rsx_live_draw.c:7557-7570). */
static u32 eng_present_surface(u32 buffer_id)
{
    if (buffer_id < 8 && g.display_buffers[buffer_id].valid) {
        const eng_display_buffer* d = &g.display_buffers[buffer_id];
        for (u32 i = 0; i < g.n_surfaces; i++)
            if (g.surfaces[i].handle && g.surfaces[i].location == d->location &&
                g.surfaces[i].offset == d->offset)
                return i;
    }
    return eng_current_surface();
}

/* RSX_SURF_DUMP_DIR=<dir> [RSX_SURF_DUMP_FROM=<frame> RSX_SURF_DUMP_EVERY=<n>
 * RSX_SURF_DUMP_COUNT=<k>]: after a present, read every registered colour
 * surface back and write it as a PPM (an FP16 target clamped to 0..1), with a
 * log line carrying its mean and maximum. The whole post-process chain of one
 * frame side by side: which stage goes white is then a matter of looking. */
static void eng_surface_dump_frame(void)
{
    static const char* dir = (const char*)1; static long from, every, count, done;
    if (dir == (const char*)1) {
        dir = getenv("RSX_SURF_DUMP_DIR");
        const char* e;
        from  = (e = getenv("RSX_SURF_DUMP_FROM"))  ? atol(e) : 0;
        every = (e = getenv("RSX_SURF_DUMP_EVERY")) ? atol(e) : 240; if (every <= 0) every = 240;
        count = (e = getenv("RSX_SURF_DUMP_COUNT")) ? atol(e) : 8;
    }
    /* RSX_SURF_DUMP_BRIGHT=<mean>: instead of a frame schedule, dump when the
     * presented frame's mean brightness (0-255) is at least <mean>, from
     * RSX_SURF_DUMP_FROM on, at least RSX_SURF_DUMP_EVERY frames apart -- the
     * scene you are after, wherever the run's timing put it. */
    static long bright = -2, last_dump = -1000000;
    if (bright == -2) { const char* e = getenv("RSX_SURF_DUMP_BRIGHT"); bright = e ? atol(e) : -1; }
    if (!dir || !*dir || done >= count || (long)g.frames < from) return;
    if (bright >= 0) {
        if ((long)g.frames - last_dump < every) return;
        const eng_surface* ps = &g.surfaces[g.last_present_surface];
        if (ps->fmt != RSX_BE_FMT_R8G8B8A8 || !ps->w || !ps->h) return;
        u8* buf = (u8*)malloc((size_t)ps->w * ps->h * 4);
        if (!buf) return;
        g.be->readback(g.be->user, ps->handle, 0, 0, ps->w, ps->h, buf, ps->w * 4);
        unsigned long long sum = 0; u32 n = 0;
        for (u32 q = 0; q < ps->w * ps->h; q += 61) { sum += buf[q * 4] + buf[q * 4 + 1] + buf[q * 4 + 2]; n += 3; }
        free(buf);
        if (!n || (long)(sum / n) < bright) return;
        last_dump = (long)g.frames;
        fprintf(stderr, "[surf-dump] frame %u presented mean %llu >= %ld: dumping\n", g.frames, sum / n, bright);
    } else if (((long)g.frames - from) % every) return;
    done++;
    eng_surface_dump_now();
}

static void eng_surface_dump_now(void)
{
    static const char* dir = (const char*)1;
    if (dir == (const char*)1) dir = getenv("RSX_SURF_DUMP_DIR");
    if (!dir || !*dir) return;
    for (u32 i = 0; i < g.n_surfaces; i++) {
        const eng_surface* sf = &g.surfaces[i];
        if (!sf->handle || !sf->w || !sf->h) continue;
        const int fp16 = sf->fmt == RSX_BE_FMT_R16G16B16A16F;
        const int f32  = sf->fmt == RSX_BE_FMT_R32F;
        if (!fp16 && !f32 && sf->fmt != RSX_BE_FMT_R8G8B8A8) continue;
        const u32 bpp = fp16 ? 8u : 4u;
        u8* buf = (u8*)malloc((size_t)sf->w * sf->h * bpp);
        if (!buf) continue;
        memset(buf, 0, (size_t)sf->w * sf->h * bpp);
        g.be->readback(g.be->user, sf->handle, 0, 0, sf->w, sf->h, buf, sf->w * bpp);
        double sum[3] = {0, 0, 0}, mx[3] = {0, 0, 0}; unsigned long nan = 0;
        double asum = 0, amin = 1e30, amax = -1e30;
        char path[1024];
        snprintf(path, sizeof path, "%s/f%06u%s_s%02u_%ux%u_%s.ppm", dir, g.frames, s_dump_tag, i, sf->w, sf->h,
                 fp16 ? "fp16" : f32 ? "r32f" : "rgba8");
        FILE* f = fopen(path, "wb");
        if (f) fprintf(f, "P6\n%u %u\n255\n", sf->w, sf->h);
        for (u32 p = 0; p < sf->w * sf->h; p++) {
            float c[3];
            if (fp16) {
                const u16* h = (const u16*)(buf + (size_t)p * 8);
                float a4[4];
                for (int k = 0; k < 4; k++) {
                    u16 v = h[k]; u32 sgn = (v >> 15) & 1, ex = (v >> 10) & 0x1F, mant = v & 0x3FF; float out;
                    if (ex == 0) out = (float)mant / 1024.0f / 16384.0f;
                    else if (ex == 31) { out = mant ? 0.0f : 1e30f; if (mant) nan++; }
                    else out = (1.0f + mant / 1024.0f) * (float)pow(2.0, (int)ex - 15);
                    a4[k] = sgn ? -out : out;
                }
                c[0] = a4[0]; c[1] = a4[1]; c[2] = a4[2];
                asum += a4[3]; if (a4[3] < amin) amin = a4[3]; if (a4[3] > amax) amax = a4[3];
            } else if (f32) {
                float v; memcpy(&v, buf + (size_t)p * 4, 4); c[0] = c[1] = c[2] = v;
            } else {
                c[0] = buf[p * 4] / 255.0f; c[1] = buf[p * 4 + 1] / 255.0f; c[2] = buf[p * 4 + 2] / 255.0f;
            }
            u8 rgb[3];
            for (int k = 0; k < 3; k++) {
                if (c[k] == c[k]) { sum[k] += c[k]; if (c[k] > mx[k]) mx[k] = c[k]; }
                float t = c[k] < 0 ? 0 : c[k] > 1 ? 1 : c[k]; rgb[k] = (u8)(t * 255.0f + 0.5f);
            }
            if (f) fwrite(rgb, 1, 3, f);
        }
        if (f) fclose(f);
        const double n = (double)sf->w * sf->h;
        fprintf(stderr, "[surf-dump] f%u s%u %ux%u %s off=0x%08X mean=(%.3f %.3f %.3f) max=(%.2f %.2f %.2f) nan=%lu",
                g.frames, i, sf->w, sf->h, fp16 ? "fp16" : f32 ? "r32f" : "rgba8", sf->offset,
                sum[0] / n, sum[1] / n, sum[2] / n, mx[0], mx[1], mx[2], nan);
        if (fp16) fprintf(stderr, " alpha mean=%.4g min=%.4g max=%.4g", asum / n, amin, amax);
        fprintf(stderr, " -> %s\n", path);
        /* An FP16 target's alpha as its own image, scaled to its maximum:
         * UE3 on PS3 keeps scene depth there. */
        if (fp16 && amax > 0) {
            char apath[1100]; snprintf(apath, sizeof apath, "%s.alpha.ppm", path);
            FILE* af = fopen(apath, "wb");
            if (af) { fprintf(af, "P6\n%u %u\n255\n", sf->w, sf->h);
                for (u32 p = 0; p < sf->w * sf->h; p++) {
                    const u16 v = ((const u16*)(buf + (size_t)p * 8))[3];
                    u32 ex = (v >> 10) & 0x1F, mant = v & 0x3FF; float out;
                    if (ex == 0) out = (float)mant / 1024.0f / 16384.0f; else if (ex == 31) out = 0; else out = (1.0f + mant / 1024.0f) * (float)pow(2.0, (int)ex - 15);
                    if (v & 0x8000) out = -out;
                    float t = out / (float)amax; t = t < 0 ? 0 : t > 1 ? 1 : t; u8 g8 = (u8)(t * 255.0f + 0.5f); u8 px3[3] = { g8, g8, g8 };
                    fwrite(px3, 1, 3, af); }
                fclose(af); } }
        free(buf);
    }
}
static const char* s_present_src = "host";   /* "fifo": the title's own flip method */
static void eng_present(u32 buffer_id)
{
    if (!g.ready) return;
    const u32 target = eng_present_surface(buffer_id);
    if (target == ENG_INVALID) return;
    if (getenv("PS3RECOMP_METAL_FRAME_DUMP") && g.frames % 120 == 0) {
        u32 current = eng_current_surface();
        fprintf(stderr, "[rsx capture] frame=%u buffer=%u selected=%u current=%u draws=%u surfaces=%u\n",
                g.frames, buffer_id, target, current, g.guest_draws, g.n_surfaces);
        for (u32 i = 0; i < g.n_surfaces; ++i)
            fprintf(stderr, "[rsx capture] surface %u loc=%u offset=%08X size=%ux%u handle=%u\n",
                    i, g.surfaces[i].location, g.surfaces[i].offset,
                    g.surfaces[i].w, g.surfaces[i].h, g.surfaces[i].handle);
    }
    g.last_present_surface = target;
    g.be->present(g.be->user, g.surfaces[target].handle);
    /* DOD3_STUTTER_MS=<n>: a frame longer than n ms says what the walker did
     * in it -- texture uploads, vertex conversions stored, pipelines built --
     * beside the guest threads' report ([stutter], ppu_hle.cpp). */
    { static int thr = -1; static double last;
      if (thr < 0) { const char* e = getenv("DOD3_STUTTER_MS"); thr = e ? atoi(e) : 0; }
      if (thr > 0) {
          struct timespec ts; timespec_get(&ts, TIME_UTC);
          const double now = (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
          if (last > 0 && now - last > thr)
              { extern double g_jc_walk_ms; extern unsigned g_jc_walks;
              fprintf(stderr, "[stutter-rsx] frame %u took %.0f ms: %u textures uploaded (%llu KB), "
                      "%u vertex conversions stored (%llu KB), %u pipelines built (%.0f ms), "
                      "%u cache evictions (%u entries, %.1f ms), %u draws in %.1f ms of walker; "
                      "%u job-chain walks, %.1f ms, on the render thread\n",
                      g.frames, now - last, s_fstat.tex, (unsigned long long)(s_fstat.tex_bytes >> 10),
                      s_fstat.vc_store, (unsigned long long)(s_fstat.vc_bytes >> 10),
                      s_fstat.pipes, s_fstat.pipe_ms, s_fstat.vc_evicts, s_fstat.vc_evicted, s_fstat.vc_evict_ms,
                      s_fstat.draws, s_fstat.draw_ms, g_jc_walks, g_jc_walk_ms); }
          last = now;
          memset(&s_fstat, 0, sizeof s_fstat);
          { extern double g_jc_walk_ms; extern unsigned g_jc_walks; g_jc_walk_ms = 0; g_jc_walks = 0; }
      } }
    /* Every 5 s: presents per second and frame times, worst included -- the
     * numbers a player feels. RSX_FRAMETIME=0 turns the line off. */
    { static int on = -1; static double win_start, last, worst, first; static u32 n, slow;
      if (on < 0) { const char* e = getenv("RSX_FRAMETIME"); on = !(e && e[0] == '0'); }
      if (on) {
          struct timespec ts; timespec_get(&ts, TIME_UTC);
          const double now = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
          if (last > 0.0) {
              const double ft = (now - last) * 1000.0;
              if (ft > worst) worst = ft;
              if (ft > 34.0) slow++;
              if (ft > 25.0) g_rsx_engine_hitches++;
              /* RSX_HITCH_LOG=<ms>: each present that came later than that,
               * with its frame number -- nothing else switched on, so the
               * log does not make hitches of its own. */
              { static double hl = -1.0;
                if (hl < 0.0) { const char* e = getenv("RSX_HITCH_LOG"); hl = e ? atof(e) : 0.0; }
                if (hl > 0.0 && ft > hl && g_rsx_hitch_hook) g_rsx_hitch_hook(ft);
                /* With the walker's blocking GPU waits in the frame (the
                 * D3D12 backend adds them up; 0 elsewhere). */
                if (hl > 0.0 && ft > hl)
                    fprintf(stderr, "[hitch] frame %u: %.1f ms (walker blocked on the GPU %.1f ms of it; "
                            "%u textures uploaded %llu KB, %u vertex conversions %llu KB, %u pipelines built %.1f ms, "
                            "%u draws %.1f ms, pipeline states %.1f ms)\n",
                            g.frames, ft, g_rsx_frame_gpu_wait_ms,
                            s_fstat.tex, (unsigned long long)(s_fstat.tex_bytes >> 10),
                            s_fstat.vc_store, (unsigned long long)(s_fstat.vc_bytes >> 10),
                            s_fstat.pipes, s_fstat.pipe_ms, s_fstat.draws, s_fstat.draw_ms, g_rsx_frame_pso_ms);
                g_rsx_frame_gpu_wait_ms = 0.0; g_rsx_frame_pso_ms = 0.0;
                /* The stutter report resets these per frame; without it, here. */
                { static int st = -1; if (st < 0) st = getenv("DOD3_STUTTER_MS") ? 1 : 0;
                  if (!st) memset(&s_fstat, 0, sizeof s_fstat); } }
              n++;
          } else {
              win_start = first = now;
          }
          last = now;
          if (now - win_start >= 5.0 && n) {
              const double span = now - win_start;
              fprintf(stderr, "[frametime] t=%.0fs %.1f fps, mean %.1f ms, worst %.1f ms, %u of %u frames over 34 ms",
                      now - first, n / span, span * 1000.0 / n, worst, slow, n);
              /* The texture write-watch's work in the window, when it runs. */
              if (eng_tex_watch_on()) {
                  static unsigned long long f0, s0;
                  unsigned long long f = 0, pp = 0;
                  vm_watch_stats(&f, &pp);
                  fprintf(stderr, "; watch: %llu faults, %llu hashes saved, %llu pages protected in all",
                          f - f0, s_tex_skips_total - s0, pp);
                  f0 = f; s0 = s_tex_skips_total;
              }
              fputc('\n', stderr);
              win_start = now; worst = 0.0; n = 0; slow = 0;
          }
      } }
    g.q_unflushed = 0;   /* the present's command buffer carries them */
    eng_surface_dump_frame();
    /* RSX_PRESENT_LOG=<from frame>: one line per present from that frame on:
     * the guest's buffer id, the surface it resolved to, the draws each
     * 1280x720 colour surface received since the last present, and the mean
     * brightness of what was presented. The dump strides of 120 and 240
     * frames are multiples of a three-buffer rotation, so they could not
     * tell "one buffer in three is white" from "a white episode". */
    { static long from = -2; if (from == -2) { const char* e = getenv("RSX_PRESENT_LOG"); from = e ? atol(e) : -1; }
      if (from >= 0 && (long)g.frames >= from) {
          double mean = -1.0;
          const eng_surface* ps = &g.surfaces[target];
          if (ps->fmt == RSX_BE_FMT_R8G8B8A8 && ps->w && ps->h) {
              u8* buf = (u8*)malloc((size_t)ps->w * ps->h * 4);
              if (buf) { g.be->readback(g.be->user, ps->handle, 0, 0, ps->w, ps->h, buf, ps->w * 4);
                  unsigned long long sum = 0; u32 n = 0;
                  for (u32 p = 0; p < ps->w * ps->h; p += 61) { sum += buf[p * 4] + buf[p * 4 + 1] + buf[p * 4 + 2]; n += 3; }
                  mean = n ? (double)sum / n : 0.0; free(buf); } }
          fprintf(stderr, "[present] f=%u buffer=%u -> s%u(off=%08X) %s mean=%.0f skin=%u/%u draws:", g.frames, buffer_id, target, ps->offset, s_present_src, mean, s_skin_draws, s_skin_verts);
          for (u32 i = 0; i < g.n_surfaces; i++)
              if (s_surf_draws[i]) fprintf(stderr, " s%u=%u", i, s_surf_draws[i]);
          fputc('\n', stderr);
      }
      memset(s_surf_draws, 0, sizeof s_surf_draws); s_skin_draws = s_skin_verts = 0; }
    /* RSX_TRACE_ON_WHITE=1: read the presented frame back; when it comes out
     * (nearly) all white, trace the next four frames draw by draw. */
    if (s_ring_on > 0) { char line[120];
        snprintf(line, sizeof line, "f%u PRESENT buffer=%u -> s%u(off=%08X)", g.frames, buffer_id, target, g.surfaces[target].offset);
        eng_ring_push(line); }
    { static long on = -1; static int cr = -1, cg = 0, cb = 0, ctol = 12;
      if (on < 0) { const char* e = getenv("RSX_TRACE_ON_WHITE"); on = e ? atol(e) : 0;
          /* RSX_TRACE_ON_COLOR=r,g,b[,tol]: the same trigger for a frame whose
           * mean colour is within tol of r,g,b. The all-green Bink cutscene
           * is (0,135,0): the YUV->RGB of three planes that read as zero. */
          if ((e = getenv("RSX_TRACE_ON_COLOR"))) {
              if (sscanf(e, "%d,%d,%d,%d", &cr, &cg, &cb, &ctol) >= 3) { if (!on) on = 1; } else cr = -1; } }
      /* The value is the first frame to watch from (a title's own fade-from-
       * white at a level start is not the bug); each episode traces four
       * frames and the watch re-arms 600 frames later. */
      static int armed = 0; static u32 armed_at = 0;
      if (armed && g.frames > armed_at + 600u) armed = 0;
      if (on > 0 && (long)g.frames >= on && !armed && g.surfaces[target].fmt == RSX_BE_FMT_R8G8B8A8 && g.surfaces[target].w && g.surfaces[target].h) {
          const u32 w = g.surfaces[target].w, h = g.surfaces[target].h;
          u8* buf = (u8*)malloc((size_t)w * h * 4);
          if (buf) {
              g.be->readback(g.be->user, g.surfaces[target].handle, 0, 0, w, h, buf, w * 4);
              unsigned long long sr = 0, sg = 0, sb = 0; const u32 step = 61;
              u32 n = 0;
              for (u32 p = 0; p < w * h; p += step) { sr += buf[p * 4]; sg += buf[p * 4 + 1]; sb += buf[p * 4 + 2]; n++; }
              const double mr = n ? (double)sr / n : 0.0, mg = n ? (double)sg / n : 0.0, mb = n ? (double)sb / n : 0.0;
              const double mean = (mr + mg + mb) / 3.0;
              const int white = mean > 250.0;
              const int colour = cr >= 0 && mr >= cr - ctol && mr <= cr + ctol && mg >= cg - ctol && mg <= cg + ctol &&
                                 mb >= cb - ctol && mb <= cb + ctol;
              if (white || colour) {
                  eng_ring_dump(white ? "white present" : "colour present");
                  armed = 1; armed_at = g.frames; s_dtrace_frame = (long)g.frames + 1;
                  fprintf(stderr, "[draw-trace] frame %u presented %s (mean %.0f,%.0f,%.0f): tracing frames %ld..%ld\n",
                          g.frames, white ? "WHITE" : "the watched colour", mr, mg, mb, s_dtrace_frame, s_dtrace_frame + 3);
              }
              free(buf);
          }
      } }
    /* RSX_TRACE_BUSY=<draws>[,<from frame>]: trace the four frames after the
     * first one (from that frame on) that issued more than <draws> draws --
     * a gameplay frame, wherever the run's timing put it. Once per run. */
    { static long busy = -2, bfrom = 0; static int done = 0;
      if (busy == -2) { const char* e = getenv("RSX_TRACE_BUSY"); busy = -1;
          if (e) { char* d; busy = strtol(e, &d, 0); if (*d == ',') bfrom = strtol(d + 1, 0, 0); } }
      if (busy > 0 && (g.frames % 250) == 0)
          fprintf(stderr, "[draw-trace] busy watch: frame %u guest_draws=%u\n", g.frames, g.guest_draws);
      if (busy > 0 && !done && (long)g.frames >= bfrom && (long)g.guest_draws > busy) {
          done = 1; s_dtrace_frame = (long)g.frames + 1;
          fprintf(stderr, "[draw-trace] frame %u issued %u draws: tracing frames %ld..%ld\n",
                  g.frames, g.guest_draws, s_dtrace_frame, s_dtrace_frame + 3); } }
    /* Frame capture (rsx_capture.h): starts, counts and ends at presents. */
    { double cmean = -1.0;
      if (rsx_capture_wants_mean(g.frames)) {
          const eng_surface* ps = &g.surfaces[target];
          if (ps->fmt == RSX_BE_FMT_R8G8B8A8 && ps->w && ps->h) {
              u8* buf = (u8*)malloc((size_t)ps->w * ps->h * 4);
              if (buf) { g.be->readback(g.be->user, ps->handle, 0, 0, ps->w, ps->h, buf, ps->w * 4);
                  unsigned long long sum = 0; u32 n = 0;
                  for (u32 p = 0; p < ps->w * ps->h; p += 61) { sum += buf[p * 4] + buf[p * 4 + 1] + buf[p * 4 + 2]; n += 3; }
                  cmean = n ? (double)sum / n : 0.0; free(buf); } } }
      u32 db[8][6];
      for (u32 i = 0; i < 8; i++) {
          const eng_display_buffer* d = &g.display_buffers[i];
          db[i][0] = (u32)d->valid; db[i][1] = d->location; db[i][2] = d->offset;
          db[i][3] = d->pitch; db[i][4] = d->width; db[i][5] = d->height; }
      rsx_capture_present(g.frames, cmean, g.rsx.regs, RSX_DSP_NUM_REGS, g.rsx.vp, RSX_DSP_VP_WORDS,
                          &g.rsx.constants[0][0], RSX_DSP_NUM_CONSTANTS * 4u, &db[0][0], 8); }
    g.frames++;
    g_rsx_engine_frame = g.frames;
    eng_draw_stats_report();
    g.last_guest_draws = g.guest_draws;
    g.guest_draws = 0;
}

static void sink_flip(void* user, const rsx_dispatch* r, u32 arg)
{
    (void)user; (void)r;
    g.last_flip_buffer = arg & 7u;
    g.sink_flips++;
    s_present_src = "fifo";
    eng_present(g.last_flip_buffer);
    s_present_src = "host";
}

/* A title whose FIFO carries its flips -- the 0xE944 method, or the
 * 0xFEADxxxx word libgcm's flip and prepare-flip commands write -- presents
 * through sink_flip or rsx_draw_engine_fifo_flip, in order with its draws.
 * Once that has happened the host clock's presents are not harmless repeats:
 * by the time the clock runs, the drain may have gone through several more
 * frames of commands, and the buffer it names then holds a later frame's
 * clear. Drakengard 3 clears each display buffer to white for its
 * light-attenuation pass before drawing into it, so those late presents
 * flashed white, cyan (a shadow mask) and half-built frames between the
 * real ones, and its characters and distant buildings seemed to flicker. */
static int eng_host_present_allowed(void)
{
    if (!g.sink_flips) return 1;
    static int said = 0;
    if (!said++) fprintf(stderr, "[rsx engine] host presents ignored from now on: this title flips through its FIFO\n");
    return 0;
}

/* ---- public API ---------------------------------------------------------- */

void rsx_draw_engine_set_backend(const rsx_draw_backend* backend)
{
    g.be = backend;
}

void rsx_draw_engine_set_default(int on)
{
    g.default_on = on;
}

/* RSX_QUERY_NOSYNC (see the fence in rsx_draw_engine_method): -1 undecided.
 * A global so a host can switch it in a run. */
int g_rsx_query_nosync = -1;

int rsx_draw_engine_enabled(void)
{
    static int cached = -1;
    static int env_seen = 0;
    static int env_on = 0;
    if (!env_seen) {
        const char* e = getenv("PS3RECOMP_RSX_ENGINE");
        env_seen = 1;
        if (e && *e) env_on = (strcmp(e, "dispatch") == 0) ? 1 : -1;
    }
    /* The environment is authoritative in both directions; without it the
     * backend's own default decides. Cached after the first backend has
     * registered, so the FIFO walker pays one compare. */
    const int on = env_on > 0 ? 1 : (env_on < 0 ? 0 : g.default_on);
    if (!g.be) return 0;
    if (cached < 0) cached = on;
    return cached;
}

int rsx_draw_engine_init(u32 width, u32 height)
{
    if (!g.be) return -1;
    g.width  = width  ? width  : 1280;
    g.height = height ? height : 720;
    g.vp_cb = (u8*)calloc(1, ENG_VP_CB_BYTES);
    if (!g.vp_cb) return -1;
    g.vp_cb_valid = 0;
    g.vp_cb_bound = 0;
    if (g.be->init && g.be->init(g.be->user, g.width, g.height) != 0) {
        free(g.vp_cb); g.vp_cb = NULL;
        return -1;
    }
    rsx_dispatch_sink sink;
    memset(&sink, 0, sizeof sink);
    sink.clear            = sink_clear;
    sink.begin            = sink_begin;
    sink.end              = sink_end;
    sink.draw_arrays      = sink_draw_arrays;
    sink.draw_index_array = sink_draw_index;
    sink.inline_array     = sink_inline_array;
    sink.flip             = sink_flip;
    rsx_dispatch_init(&g.rsx, &sink);
    g.ready = 1;
    fprintf(stderr, "[rsx engine] register-file draw engine up (%ux%u)\n",
            g.width, g.height);
    return 0;
}

void rsx_draw_engine_shutdown(void)
{
    if (!g.be) return;
    eng_pipe_stop();
    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].pending) eng_pipe_poll(&g.pipelines[i]);
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_SHUTDOWN);
    for (u32 i = 0; i < g.n_textures; i++)
        if (g.textures[i].handle) g.be->texture_release(g.be->user, g.textures[i].handle);
    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].handle) g.be->pipeline_release(g.be->user, g.pipelines[i].handle);
    for (u32 i = 0; i < g.n_surfaces; i++)
        if (g.surfaces[i].handle) g.be->color_target_release(g.be->user, g.surfaces[i].handle);
    for (u32 i = 0; i < g.n_zdepths; i++)
        if (g.zdepths[i].handle) g.be->depth_target_release(g.be->user, g.zdepths[i].handle);
    if (s_vc) {
        for (u32 i = 0; i < ENG_VC_SLOTS; i++) if (s_vc[i].key) eng_vc_free(&s_vc[i]);
        free(s_vc); s_vc = NULL; s_vc_count = 0; s_vc_bytes = 0;
    }
    if (g.ready && g.be->shutdown) g.be->shutdown(g.be->user);

    rsx_draw_engine_set_guest_memory(NULL, NULL);
    free(dc.refs); free(dc.cuts); free(dc.verts);
    rsx_vertex_remap_destroy(&dc.ref_remap);
    memset(&dc, 0, sizeof dc);
    free(g.tex_staging); free(g.vp_cb); free(g.fp_cb); free(g.indices);
    const rsx_draw_backend* be = g.be;
    memset(&g, 0, sizeof g);
    g.be = be;
}

void rsx_draw_engine_method(u32 method, u32 arg)
{
    if (!g.ready) return;
    /* The subchannel is a binding slot, not an engine selector, and a title
     * whose SPU-built command lists bind NV4097 elsewhere would otherwise
     * store its state in the wrong register bank.  Standard NV4097 methods
     * (< 0x2000) strip the subchannel with & 0x1FFC.  Driver methods
     * (0xE9xx, 0xEBxx) encode the subchannel bits as part of the address;
     * preserve them with the wider mask. */
    u32 m = (method >= 0xE000u) ? (method & 0xFFFCu) : (method & 0x1FFCu);
    if (m == 0x1FD8u && s_snap_stats > 0) { s_inval_n[arg < 4 ? arg : 0]++; s_inval_total++; }
    /* Recorded after the dispatch, behind the pages it read (rsx_capture.c),
     * and only when the capture was already running when it began. */
    const int cap = g_rsx_capture_on;
    /* Occlusion-query methods in the draw trace: CLEAR_REPORT_VALUE (0x17C8),
     * GET_REPORT (0x1800), SET_ZPASS_PIXEL_COUNT_ENABLE (0x1D84). */
    /* Occlusion queries: CLEAR_REPORT_VALUE(ZPASS) opens a counter for the
     * draws that follow, GET_REPORT(ZPASS) closes it and asks the backend for
     * its total under that report index (rsx_draw_engine_query_result). The
     * title never sets ZPASS_PIXEL_COUNT_ENABLE and still expects counts, so
     * an open query counts regardless. A query whose draws never reached the
     * backend -- dropped, not drawn -- reports "visible": a bug of ours must
     * not hide the object. */
    if (m == 0x17C8u && arg == 1u && g.be->query_begin) {
        g.q_cur = g.be->query_begin(g.be->user);
        g.q_attempts = g.q_draws = 0;
    } else if (m == 0x1800u && (arg >> 24) == 1u && g.q_cur) {
        const u32 index = (arg & 0xFFFFFFu) / 16u;
        if (g.q_draws && g.q_draws >= g.q_attempts) { g.be->query_report(g.be->user, g.q_cur, index); g.q_unflushed++; }
        else rsx_draw_engine_query_result(index, 0xFFFFu);
        g.q_cur = 0;
        if (g.be->query_set) g.be->query_set(g.be->user, 0);
    } else if (m == 0x1D70u && g.q_unflushed && g.be->submit_and_wait) {
        /* BACK_END_WRITE_SEMAPHORE_RELEASE after occlusion queries: the title
         * reads a fence the GPU writes once it is past the query block, then
         * trusts the report values. The walker writes the fence as soon as it
         * reaches it, long before the counts come back, so Drakengard 3 read
         * last frame's values -- another object's, since report indices are a
         * per-frame pool -- and culled whatever drew a 0: characters and props
         * flickered out for single frames. Finish the GPU work up to here and
         * deliver the counts before the fence lands. RSX_QUERY_NOSYNC=1 off. */
        if (g_rsx_query_nosync < 0) g_rsx_query_nosync = getenv("RSX_QUERY_NOSYNC") ? 1 : 0;
        if (!g_rsx_query_nosync) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_QUERY_FENCE);
        g.q_unflushed = 0;
    }
    if ((m == 0x17C8u || m == 0x1800u || m == 0x1D84u) && s_dtrace_frame >= 0 &&
        (long)g.frames >= s_dtrace_frame && (long)g.frames < s_dtrace_frame + 4)
        fprintf(stderr, "[draw-trace] f%u QUERY %s 0x%08X\n", g.frames,
                m == 0x17C8u ? "CLEAR_REPORT_VALUE" : m == 0x1800u ? "GET_REPORT" : "ZPASS_ENABLE", arg);
    rsx_dispatch_method(&g.rsx, m, arg);
    if (cap) rsx_capture_method(method, arg);
}

void cellGcm_set_report_value(u32 index, u32 value);   /* cellGcmSys.c */
void rsx_draw_engine_query_result(u32 report_index, u64 count)
{
    static int log = -1; if (log < 0) log = getenv("RSX_QUERY_LOG") ? 1 : 0;
    if (log) fprintf(stderr, "[query] f%u report %u = %llu\n", g.frames, report_index, (unsigned long long)count);
    cellGcm_set_report_value(report_index, count > 0xFFFFFFFFull ? 0xFFFFFFFFu : (u32)count);
}

void rsx_draw_engine_set_display_buffer(u32 buffer_id, u32 location, u32 offset,
                                        u32 pitch, u32 width, u32 height)
{
    if (buffer_id >= 8) return;
    rsx_capture_display_buffer(buffer_id, location, offset, pitch, width, height);
    eng_display_buffer* d = &g.display_buffers[buffer_id];
    d->location = location;
    d->offset = offset;
    d->pitch = pitch;
    d->width = width;
    d->height = height;
    d->valid = width && height;
}

void rsx_draw_engine_sync_queries(void)
{
    if (!g.ready || !g.q_unflushed || !g.be->submit_and_wait) return;
    g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_QUERY_FENCE);
    g.q_unflushed = 0;
}

void rsx_draw_engine_flush(void)
{
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
}

void rsx_draw_engine_present(void)
{
    /* A host that drives the flip itself gets whichever buffer the guest's
     * own last flip named, and display buffer 0 before there has been one:
     * a runner that both consumes 0xE944 and calls here would otherwise
     * re-present a double-buffered title's OTHER scanout. Presenting twice is
     * harmless either way -- the engine never clears the surface, so a second
     * present just blits the same image again. */
    if (!eng_host_present_allowed()) return;
    eng_present(g.last_flip_buffer);
}

void rsx_draw_engine_present_buffer(u32 buffer_id)
{
    if (!eng_host_present_allowed()) return;
    g.last_flip_buffer = buffer_id & 7u;
    eng_present(g.last_flip_buffer);
}

int rsx_draw_engine_fifo_flip(u32 buffer_id)
{
    extern void (*g_gcm_trace_hook)(u32 type, u32 a, u32 b);
    if (g_gcm_trace_hook) g_gcm_trace_hook(8, buffer_id, g.frames);
    if (!g.ready) return 0;
    g.last_flip_buffer = buffer_id & 7u;
    g.sink_flips++;
    rsx_capture_flip(buffer_id);
    s_present_src = "fifo";
    eng_present(g.last_flip_buffer);
    s_present_src = "host";
    return 1;
}

u32 rsx_draw_engine_guest_draws(void)
{
    return g.last_guest_draws;
}

u32 rsx_draw_engine_readback_center(void)
{
    if (!g.ready || !g.be->readback) return 0;
    const u32 slot = g.last_present_surface;
    if (slot >= g.n_surfaces) return 0;
    const eng_surface* s = &g.surfaces[slot];
    if (!s->handle || s->fmt != RSX_BE_FMT_R8G8B8A8 || !s->w || !s->h) return 0;
    u8 px[4] = { 0, 0, 0, 0 };
    g.be->readback(g.be->user, s->handle, s->w / 2u, s->h / 2u, 1, 1, px, 4);
    /* Decoded rows are R,G,B,A; the guest's own clear colour is A8R8G8B8, and
     * a host comparing the two wants them in the same order. */
    return ((u32)px[3] << 24) | ((u32)px[0] << 16) |
           ((u32)px[1] << 8)  |  (u32)px[2];
}
