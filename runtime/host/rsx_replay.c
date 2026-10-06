/*
 * rsx_replay - play a frame capture (libs/video/rsx_capture.h) back through the
 * register-file draw engine and the headless Metal backend, with no game.
 *
 *   rsx_replay <capture.rsxcap> [--out <prefix>] [--every <n>] [--last <n>]
 *              [--from <present>] [--surf <dir>]
 *
 *   --out    write presented frames as <prefix>.<present>.ppm
 *   --every  ...every n-th present (default 1)
 *   --last   ...only the last n presents (overrides --every's start)
 *   --from   ...only from this present on
 *   --surf   also dump every render target (RSX_SURF_DUMP_DIR) at the
 *            presents --out writes
 *
 * Every switch the engine and backend read from the environment works here
 * too (RSX_SKIP_FP, RSX_DRAW_TRACE_FRAME, RSX_NO_SHADOW_CMP, ...), so an
 * experiment that took a 5-minute game run takes a few seconds.
 *
 * The capture's pages go into two sparse arenas, one per RSX location, as the
 * stream reaches them; the engine's guest reads are served from there, and a
 * read touching a page the capture never recorded fails as it did live.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <time.h>
#include <zlib.h>

#include "ps3emu/ps3types.h"
#include "../../libs/video/rsx_capture.h"
#include "../../libs/video/rsx_draw_engine.h"
#include "../../libs/video/rsx_dispatch.h"

int rsx_metal_backend_init(u32 width, u32 height, const char* title);

/* The runtime's flat guest VM. Replay never touches it: the engine reads
 * through the reader below. */
uint8_t* vm_base = NULL;
/* ppu_loader.cpp's dynamic store-watch address, referenced by the GCM layer's
 * park diagnostics; there is no PPU here. */
uint32_t g_ww_dyn = 0;
void ppu_vm_slow_any_update(void) {}   /* cellGcmSys.c's word watch, no PPU here */

#define ARENA_BYTES 0x100000000ull
#define NPAGES      (ARENA_BYTES / RSX_CAPTURE_PAGE)

static u8* s_arena[2];
static u8* s_have[2];   /* one bit per page: the capture recorded it */

static const u8* replay_reader(void* user, u32 location, u32 offset, u32 len)
{
    (void)user;
    if (location > 1 || !len) return NULL;
    const u64 end = (u64)offset + len;
    if (end > ARENA_BYTES) return NULL;
    for (u64 pg = offset / RSX_CAPTURE_PAGE; pg <= (end - 1) / RSX_CAPTURE_PAGE; pg++)
        if (!(s_have[location][pg >> 3] & (1u << (pg & 7)))) return NULL;
    return s_arena[location] + offset;
}

static gzFile s_gz;
static int rd(void* p, unsigned n)
{
    return gzread(s_gz, p, n) == (int)n;
}

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv)
{
    const char* cap = NULL; const char* out = NULL; const char* surf = NULL;
    long every = 1, last = 0, from = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atol(argv[++i]);
        else if (!strcmp(argv[i], "--last") && i + 1 < argc) last = atol(argv[++i]);
        else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = atol(argv[++i]);
        else if (!strcmp(argv[i], "--surf") && i + 1 < argc) surf = argv[++i];
        else if (argv[i][0] != '-' && !cap) cap = argv[i];
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!cap) {
        fprintf(stderr, "usage: rsx_replay <capture> [--out <prefix>] [--every n] [--last n] [--from n] [--surf dir]\n");
        return 2;
    }
    if (every <= 0) every = 1;

    /* First pass: how many presents, so --last can be honoured. */
    long total = 0;
    s_gz = gzopen(cap, "rb");
    if (!s_gz) { fprintf(stderr, "cannot open %s\n", cap); return 1; }
    char magic[8]; u32 hdr[4];
    if (!rd(magic, 8) || memcmp(magic, RSX_CAPTURE_MAGIC, 8) || !rd(hdr, sizeof hdr)) {
        fprintf(stderr, "%s is not an RSX capture\n", cap); return 1;
    }
    const u32 nregs = hdr[0], nvp = hdr[1], nconst = hdr[2], ndb = hdr[3];
    gzseek(s_gz, (z_off_t)(4u * (nregs + nvp + nconst + ndb * 6u)), SEEK_CUR);
    for (;;) {
        u8 kind; if (!rd(&kind, 1)) break;
        if (kind == RSX_CAP_END) break;
        if (kind == RSX_CAP_METHOD) gzseek(s_gz, 8, SEEK_CUR);
        else if (kind == RSX_CAP_PAGE) gzseek(s_gz, 5 + RSX_CAPTURE_PAGE, SEEK_CUR);
        else if (kind == RSX_CAP_FLIP) { gzseek(s_gz, 4, SEEK_CUR); total++; }
        else if (kind == RSX_CAP_DISPBUF) gzseek(s_gz, 24, SEEK_CUR);
        else { fprintf(stderr, "corrupt capture (record kind %u)\n", kind); return 1; }
    }
    gzclose(s_gz);
    if (last > 0 && last < total && total - last > from) from = total - last;

    /* The backend dumps every `every`-th present it is handed; presents
     * before `from` are skipped by turning the dump off until then. */
    setenv("PS3RECOMP_METAL_HEADLESS", "1", 1);
    if (surf) {   /* the engine counts presents from 0, as the loop below does */
        char v[32];
        setenv("RSX_SURF_DUMP_DIR", surf, 1);
        snprintf(v, sizeof v, "%ld", from);  setenv("RSX_SURF_DUMP_FROM", v, 1);
        snprintf(v, sizeof v, "%ld", every); setenv("RSX_SURF_DUMP_EVERY", v, 1);
        setenv("RSX_SURF_DUMP_COUNT", "100000", 1);
    }
    if (!getenv("PS3RECOMP_MSL_CACHE")) setenv("PS3RECOMP_MSL_CACHE", "cache/msl", 1);
    if (!getenv("PS3RECOMP_RSX_ENGINE")) setenv("PS3RECOMP_RSX_ENGINE", "dispatch", 1);
    /* Frames are compared byte for byte, so every draw waits for its pipeline. */
    if (!getenv("RSX_ASYNC_SHADERS")) setenv("RSX_ASYNC_SHADERS", "0", 1);

    for (int l = 0; l < 2; l++) {
        s_arena[l] = (u8*)mmap(NULL, ARENA_BYTES, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANON, -1, 0);
        s_have[l] = (u8*)calloc(NPAGES / 8, 1);
        if (s_arena[l] == MAP_FAILED || !s_have[l]) { fprintf(stderr, "arena allocation failed\n"); return 1; }
    }

    if (rsx_metal_backend_init(1280, 720, "rsx_replay") != 0) {
        fprintf(stderr, "Metal backend init failed\n"); return 1;
    }
    rsx_draw_engine_set_guest_memory(replay_reader, NULL);

    s_gz = gzopen(cap, "rb");
    gzbuffer(s_gz, 1u << 20);
    rd(magic, 8); rd(hdr, sizeof hdr);
    u32* regs = (u32*)malloc(nregs * 4u);
    u32* vp = (u32*)malloc(nvp * 4u);
    u32* cst = (u32*)malloc(nconst * 4u);
    if (!rd(regs, nregs * 4u) || !rd(vp, nvp * 4u) || !rd(cst, nconst * 4u)) {
        fprintf(stderr, "truncated header\n"); return 1;
    }
    rsx_draw_engine_seed_state(regs, nregs, vp, nvp, cst, nconst);
    for (u32 i = 0; i < ndb; i++) {
        u32 d[6]; if (!rd(d, sizeof d)) return 1;
        if (d[0]) rsx_draw_engine_set_display_buffer(i, d[1], d[2], d[3], d[4], d[5]);
    }

    fprintf(stderr, "[replay] %s: %ld presents; writing %s from present %ld every %ld\n",
            cap, total, out ? out : "(nothing)", from, every);
    const double t0 = now_s();
    long presents = 0; unsigned long long methods = 0, pages = 0;
    for (;;) {
        u8 kind; if (!rd(&kind, 1)) break;
        if (kind == RSX_CAP_END) break;
        if (kind == RSX_CAP_METHOD) {
            u32 w[2]; if (!rd(w, 8)) break;
            rsx_draw_engine_method(w[0], w[1]);
            methods++;
        } else if (kind == RSX_CAP_PAGE) {
            u8 loc; u32 off;
            if (!rd(&loc, 1) || !rd(&off, 4) || loc > 1) break;
            if (!rd(s_arena[loc] + off, RSX_CAPTURE_PAGE)) break;
            const u64 pg = off / RSX_CAPTURE_PAGE;
            s_have[loc][pg >> 3] |= (u8)(1u << (pg & 7));
            pages++;
        } else if (kind == RSX_CAP_FLIP) {
            u32 b; if (!rd(&b, 4)) break;
            /* Arm the dumps for this present only when it is one we want. */
            const int want = out && presents >= from && ((presents - from) % every) == 0;
            if (want) {
                char path[1024];
                snprintf(path, sizeof path, "%s.%06ld.ppm", out, presents);
                setenv("RSX_REPLAY_DUMP_PATH", path, 1);
            } else {
                unsetenv("RSX_REPLAY_DUMP_PATH");
            }
            rsx_draw_engine_fifo_flip(b);
            presents++;
        } else if (kind == RSX_CAP_DISPBUF) {
            u32 d[6]; if (!rd(d, sizeof d)) break;
            rsx_draw_engine_set_display_buffer(d[0], d[1], d[2], d[3], d[4], d[5]);
        } else {
            fprintf(stderr, "corrupt capture (record kind %u)\n", kind); break;
        }
    }
    gzclose(s_gz);
    const double dt = now_s() - t0;
    fprintf(stderr, "[replay] %ld presents, %llu methods, %llu pages in %.1f s (%.1f presents/s)\n",
            presents, methods, pages, dt, dt > 0 ? presents / dt : 0.0);
    return 0;
}
