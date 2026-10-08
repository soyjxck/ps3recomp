/*
 * ps3recomp - the register-file draw engine's Vulkan backend (Windows for now)
 *
 * The same contract as rsx_d3d12_engine.h, over Vulkan 1.3. rsx_d3d12_backend.c
 * picks it when RSX_BACKEND=vulkan (tools/rsx_replay too), and every other
 * entry point routes to it while it is active.
 */
#ifndef PS3RECOMP_RSX_VULKAN_ENGINE_H
#define PS3RECOMP_RSX_VULKAN_ENGINE_H

#include "ps3emu/ps3types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Window (unless headless: PS3RECOMP_VULKAN_HEADLESS or
 * PS3RECOMP_D3D12_HEADLESS), instance, device and resources; registers with
 * rsx_draw_engine and inits it. 0 when it is the live path, -1 otherwise
 * (no vulkan-1.dll, no Vulkan 1.3 device, ...). */
int  rsx_vulkan_engine_init(u32 width, u32 height, const char* title);
void rsx_vulkan_engine_shutdown(void);
int  rsx_vulkan_engine_active(void);
/* Pump the window's messages; -1 once it has been closed. */
int  rsx_vulkan_engine_pump_messages(void);

/* 1 when RSX_BACKEND asks for Vulkan. */
int  rsx_vulkan_engine_wanted(void);

#ifdef __cplusplus
}
#endif
#endif
