/* The fork's runtime as cellGcmSys.c reaches it, for the fixtures that compile
 * cellGcmSys.c in. Include it after cellGcmSys.c and put
 * runtime/platform/env_cache.c (getenv's cache) on the link line.
 *
 * Guest time is host time (never paused). No store or DMA watch for
 * GCM_PARK_WATCH to arm, no page table to refresh. The draw engine presents no
 * flip, has no occlusion queries out, and takes a transform-constant or program
 * upload run as rsx_draw_engine.c does when its register file cannot take the
 * run whole: word by word through the fixture's rsx_draw_engine_method. */
#ifndef GCM_FORK_STANDINS_H
#define GCM_FORK_STANDINS_H
int64_t ps3_guest_ticks(int64_t host_ticks) { return host_ticks; }
uint32_t g_ww_dyn, g_spu_watchea_dyn;
void ppu_vm_slow_any_update(void) {}
int  rsx_draw_engine_fifo_flip(u32 buffer_id) { (void)buffer_id; return 0; }
void rsx_draw_engine_sync_queries(void) {}
int  rsx_draw_engine_upload_window(u32 method, u32 count)
{
    if (!count) return 0;
    const u32 m = method & 0x1FFCu, end = m + count * 4u;
    return (m >= 0x1F00u && end <= 0x1F80u) || (m >= 0x0B80u && end <= 0x0C00u);
}
void rsx_draw_engine_upload_run(u32 method, u32 count, const u8* be)
{
    for (u32 i = 0; i < count; i++) {
        const u8* p = be + i * 4u;
        rsx_draw_engine_method(method + i * 4u,
                               ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | (u32)p[3]);
    }
}
#endif
