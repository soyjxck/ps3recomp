/* What the fork's Metal backend and draw engine reach in the rest of the
 * runtime, for a Metal test linked without it (upstream's backend tests are
 * built from the backend and the video sources alone): the GCM display
 * buffer, IO and report hooks, the frame capture (off), the store watch (not
 * available: the engine hashes), the job-chain walk timers and the guest
 * memory size. As test_rsx_draw_engine.c's. Weak, so a test that defines one
 * itself (test_metal_poly_offset.m: cellGcmResolveIO, ppu_vm_size) keeps its
 * own. The env cache is runtime/platform/env_cache.c, linked beside this. */
#include <stdint.h>

typedef uint32_t u32;
#define STANDIN __attribute__((weak))

STANDIN u32 cellGcmGetCurrentDisplayBufferId(void) { return 0; }
STANDIN u32 cellGcmResolveIO(u32 offset) { return offset; }
STANDIN void cellGcm_set_report_value(u32 index, u32 value) { (void)index; (void)value; }
STANDIN double   g_jc_walk_ms;
STANDIN unsigned g_jc_walks;
STANDIN u32 ppu_vm_size;
STANDIN int g_rsx_capture_on;
STANDIN void rsx_capture_read(u32 location, u32 offset, u32 len) { (void)location; (void)offset; (void)len; }
STANDIN void rsx_capture_method(u32 method, u32 arg) { (void)method; (void)arg; }
STANDIN int  rsx_capture_wants_mean(u32 frame) { (void)frame; return 0; }
STANDIN void rsx_capture_present(u32 frame, double mean, const u32* regs, u32 nregs, const u32* vp, u32 nvp,
                                 const u32* constants, u32 nconst_words, const u32* dispbuf, u32 ndispbuf)
{ (void)frame; (void)mean; (void)regs; (void)nregs; (void)vp; (void)nvp; (void)constants; (void)nconst_words; (void)dispbuf; (void)ndispbuf; }
STANDIN int      vm_watch_available(void) { return 0; }
STANDIN uint64_t vm_watch_arm(uint32_t ea, uint32_t len, uint32_t* unwatched) { (void)ea; if (unwatched) *unwatched = len; return 0; }
STANDIN void     vm_watch_stats(uint64_t* faults, uint64_t* pages_protected) { if (faults) *faults = 0; if (pages_protected) *pages_protected = 0; }
