/*
 * ps3recomp - the register-file draw engine's D3D12 backend
 *
 * rsx_d3d12_backend.c owns the public rsx_d3d12_backend_* entry points and
 * selects between its own rsx_state vtable path and this engine backend at
 * init. These are the hooks it calls; nothing else needs them.
 */
#ifndef PS3RECOMP_RSX_D3D12_ENGINE_H
#define PS3RECOMP_RSX_D3D12_ENGINE_H

#include "ps3emu/ps3types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Create the window (unless PS3RECOMP_D3D12_HEADLESS=1), the device and the
 * engine's resources, register with rsx_draw_engine and init it. Returns 0
 * when the engine is the live path, -1 when the caller should fall back to
 * the vtable path. */
int  rsx_d3d12_engine_init(u32 width, u32 height, const char* title);
void rsx_d3d12_engine_shutdown(void);
int  rsx_d3d12_engine_active(void);
/* Pump the window's messages; -1 once it has been closed. */
int  rsx_d3d12_engine_pump_messages(void);

/* Test hooks, as the Metal backend's. */
u32  rsx_d3d12_engine_debug_color(void);
u32  rsx_d3d12_engine_readback_center(void);

#ifdef __cplusplus
}
#endif
#endif
