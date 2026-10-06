/*
 * ps3recomp - RSX frame capture for offline replay
 *
 * Records what the register-file draw engine consumes over a run of frames --
 * its starting register file, every method it is handed, every display-buffer
 * change and flip, and the guest memory it reads, at the moment it reads it --
 * so tools/rsx_replay can feed the same frames through the engine and a
 * headless backend with no game running. A rendering fix is then checked in
 * seconds, on exactly the same frames, instead of by playing back to the scene.
 *
 * Memory is recorded in 4 KB pages keyed by (RSX location, offset), and a page
 * is written again only when its contents changed since it was last written,
 * so a texture read every frame costs its bytes once. Replay applies each page
 * at its place in the stream, so data the title rewrites inside a frame (vertex
 * rings, patched fragment-program constants) is seen as it was when used.
 *
 * Switches (game side):
 *   RSX_CAPTURE=<file.rsxcap>     where to write (gzip)
 *   RSX_CAPTURE_FROM=<frame>      engine frame to start at (default 0)
 *   RSX_CAPTURE_BRIGHT=<mean>     ...or the first frame from then whose presented
 *                                 image has at least this mean brightness (0-255)
 *   RSX_CAPTURE_FRAMES=<n>        presents to record (default 120)
 *
 * Not recorded: the contents of render targets at the starting frame (a pass
 * that reads last frame's luminance or history starts from black -- record a
 * second or two of frames and judge the later ones), and the GPU timing.
 */
#ifndef PS3RECOMP_RSX_CAPTURE_H
#define PS3RECOMP_RSX_CAPTURE_H

#include "ps3emu/ps3types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RSX_CAPTURE_MAGIC   "RSXCAP01"
#define RSX_CAPTURE_PAGE    4096u

enum {
    RSX_CAP_METHOD   = 1,   /* u32 method, u32 arg                          */
    RSX_CAP_PAGE     = 2,   /* u8 location, u32 offset, PAGE bytes          */
    RSX_CAP_FLIP     = 3,   /* u32 buffer id (the drain's FIFO flip)        */
    RSX_CAP_DISPBUF  = 4,   /* u32 id, location, offset, pitch, width, height */
    RSX_CAP_END      = 5,
};

/* Is a capture recording right now? (Cheap: one load.) */
extern int g_rsx_capture_on;

/* The engine read `len` bytes at (location, offset): record the pages they
 * touch that changed since they were last recorded. */
void rsx_capture_read(u32 location, u32 offset, u32 len);

void rsx_capture_method(u32 method, u32 arg);
void rsx_capture_flip(u32 buffer_id);
void rsx_capture_display_buffer(u32 id, u32 location, u32 offset, u32 pitch,
                                u32 width, u32 height);

/* Called at the end of every present. Starts the capture when its trigger
 * fires (wants_mean says whether the caller should pass the presented frame's
 * mean brightness; -1 when not computed), counts frames, and closes the file
 * after the last one. `regs`, `vp`, `constants` are the dispatcher's state and
 * `dispbuf` six u32 per display buffer (valid, location, offset, pitch, w, h). */
int  rsx_capture_wants_mean(u32 frame);
void rsx_capture_present(u32 frame, double mean,
                         const u32* regs, u32 nregs,
                         const u32* vp, u32 nvp,
                         const u32* constants, u32 nconst_words,
                         const u32* dispbuf, u32 ndispbuf);

#ifdef __cplusplus
}
#endif
#endif
