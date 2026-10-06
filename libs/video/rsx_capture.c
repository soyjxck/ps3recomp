/*
 * ps3recomp - RSX frame capture for offline replay. See rsx_capture.h.
 *
 * File layout (gzip; all integers host-endian -- capture and replay run on the
 * same machine):
 *
 *   char magic[8] = "RSXCAP01"
 *   u32  nregs, nvp, nconst_words, ndispbuf
 *   u32  regs[nregs], vp[nvp], constants[nconst_words]
 *   u32  dispbuf[ndispbuf][6]          valid, location, offset, pitch, w, h
 *   records...                          u8 kind, then its payload
 *   u8   RSX_CAP_END
 *
 * Pages are written BEFORE the method whose processing read them: the engine
 * reads while it dispatches, so a method's record is written after its
 * dispatch returns, behind the pages that dispatch recorded.
 */
#include "rsx_capture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

extern u8* vm_base;
extern u32 ppu_vm_size;
u32 cellGcmResolveLocated(int local, u32 offset);

int g_rsx_capture_on;

static gzFile s_gz;
static u64*   s_page_hash[2];          /* per location, per 4 KB page; 0 = never */
static u32    s_frames_left;
static int    s_stop_pending;
static int    s_done;                  /* one capture per run */
static unsigned long long s_pages, s_methods, s_bytes_raw;

#define CAP_PAGES_PER_LOC (0x100000000ull / RSX_CAPTURE_PAGE)

static void cap_put(const void* p, unsigned n)
{
    if (s_gz && gzwrite(s_gz, p, n) != (int)n) {
        fprintf(stderr, "[rsx capture] write failed; capture abandoned\n");
        gzclose(s_gz); s_gz = NULL; g_rsx_capture_on = 0;
    }
    s_bytes_raw += n;
}

static u64 cap_hash_page(const u8* p)
{
    u64 h = 0x9E3779B97F4A7C15ull;
    for (u32 i = 0; i < RSX_CAPTURE_PAGE; i += 8) {
        u64 w; memcpy(&w, p + i, 8);
        h ^= w; h *= 0xFF51AFD7ED558CCDull; h ^= h >> 29;
    }
    return h | 1u;                     /* never 0: 0 means "not recorded" */
}

void rsx_capture_read(u32 location, u32 offset, u32 len)
{
    if (!g_rsx_capture_on || !len || location > 1 || !vm_base) return;
    const u64 first = offset / RSX_CAPTURE_PAGE;
    const u64 last  = ((u64)offset + len - 1) / RSX_CAPTURE_PAGE;
    for (u64 pg = first; pg <= last && pg < CAP_PAGES_PER_LOC; pg++) {
        const u32 page_off = (u32)(pg * RSX_CAPTURE_PAGE);
        /* RSX location 0 is local memory; cellGcmResolveLocated takes
         * "is local" as its first argument. */
        const u32 ea = cellGcmResolveLocated(location == 0, page_off);
        if (!ea) continue;
        if (ppu_vm_size && (u64)ea + RSX_CAPTURE_PAGE > ppu_vm_size) continue;
        const u8* src = vm_base + ea;
        const u64 h = cap_hash_page(src);
        if (s_page_hash[location][pg] == h) continue;
        s_page_hash[location][pg] = h;
        const u8 kind = RSX_CAP_PAGE, loc = (u8)location;
        cap_put(&kind, 1); cap_put(&loc, 1); cap_put(&page_off, 4);
        cap_put(src, RSX_CAPTURE_PAGE);
        s_pages++;
        if (!g_rsx_capture_on) return;
    }
}

void rsx_capture_method(u32 method, u32 arg)
{
    if (!g_rsx_capture_on) return;
    const u8 kind = RSX_CAP_METHOD;
    u32 w[2] = { method, arg };
    cap_put(&kind, 1); cap_put(w, 8);
    s_methods++;
}

void rsx_capture_flip(u32 buffer_id)
{
    if (!g_rsx_capture_on) return;
    const u8 kind = RSX_CAP_FLIP;
    cap_put(&kind, 1); cap_put(&buffer_id, 4);
}

void rsx_capture_display_buffer(u32 id, u32 location, u32 offset, u32 pitch,
                                u32 width, u32 height)
{
    if (!g_rsx_capture_on) return;
    const u8 kind = RSX_CAP_DISPBUF;
    u32 w[6] = { id, location, offset, pitch, width, height };
    cap_put(&kind, 1); cap_put(w, sizeof w);
}

static const char* s_path = (const char*)1;
static long s_from, s_bright, s_nframes;

static void cap_config(void)
{
    if (s_path != (const char*)1) return;
    s_path = getenv("RSX_CAPTURE");
    const char* e;
    s_from    = (e = getenv("RSX_CAPTURE_FROM"))   ? atol(e) : 0;
    s_bright  = (e = getenv("RSX_CAPTURE_BRIGHT")) ? atol(e) : -1;
    s_nframes = (e = getenv("RSX_CAPTURE_FRAMES")) ? atol(e) : 120;
    if (s_nframes <= 0) s_nframes = 120;
}

int rsx_capture_wants_mean(u32 frame)
{
    cap_config();
    /* Only while armed, and only every 15 frames: a readback per present is
     * a full GPU sync. */
    return s_path && *s_path && !s_done && !g_rsx_capture_on && s_bright >= 0 &&
           (long)frame >= s_from && (frame % 15u) == 0;
}

static void cap_close(void)
{
    const u8 kind = RSX_CAP_END;
    cap_put(&kind, 1);
    if (s_gz) gzclose(s_gz);
    s_gz = NULL;
    g_rsx_capture_on = 0;
    s_done = 1;
    for (int i = 0; i < 2; i++) { free(s_page_hash[i]); s_page_hash[i] = NULL; }
    fprintf(stderr, "[rsx capture] wrote %s: %ld frames, %llu methods, %llu pages (%.1f MB before compression)\n",
            s_path, s_nframes, s_methods, s_pages, s_bytes_raw / 1048576.0);
}

void rsx_capture_present(u32 frame, double mean,
                         const u32* regs, u32 nregs,
                         const u32* vp, u32 nvp,
                         const u32* constants, u32 nconst_words,
                         const u32* dispbuf, u32 ndispbuf)
{
    cap_config();
    if (g_rsx_capture_on) {
        if (s_frames_left && --s_frames_left == 0) cap_close();
        return;
    }
    if (!s_path || !*s_path || s_done || (long)frame < s_from) return;
    if (s_bright >= 0 && (mean < 0 || mean < (double)s_bright)) return;

    s_gz = gzopen(s_path, "wb1");
    if (!s_gz) {
        fprintf(stderr, "[rsx capture] cannot open %s\n", s_path);
        s_done = 1;
        return;
    }
    for (int i = 0; i < 2; i++) {
        s_page_hash[i] = (u64*)calloc(CAP_PAGES_PER_LOC, sizeof(u64));
        if (!s_page_hash[i]) {
            fprintf(stderr, "[rsx capture] out of memory\n");
            gzclose(s_gz); s_gz = NULL; s_done = 1;
            return;
        }
    }
    g_rsx_capture_on = 1;
    s_frames_left = (u32)s_nframes;
    cap_put(RSX_CAPTURE_MAGIC, 8);
    const u32 hdr[4] = { nregs, nvp, nconst_words, ndispbuf };
    cap_put(hdr, sizeof hdr);
    cap_put(regs, nregs * 4u);
    cap_put(vp, nvp * 4u);
    cap_put(constants, nconst_words * 4u);
    cap_put(dispbuf, ndispbuf * 6u * 4u);
    fprintf(stderr, "[rsx capture] recording %ld frames from engine frame %u%s into %s\n",
            s_nframes, frame, mean >= 0 ? " (brightness trigger)" : "", s_path);
    (void)s_stop_pending;
}
