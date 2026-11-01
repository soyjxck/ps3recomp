/*
 * The display under a window, for HDR output (RSX_HDR).
 *
 * Windows: whether Windows has HDR on for the display the window is on
 * (DXGI's colour space for that output, Settings > Display > Use HDR), the
 * display's peak luminance, and Windows' SDR white level for it (Settings >
 * Display > HDR, SDR content brightness). The D3D12 and Vulkan engines both
 * ask it. Elsewhere it finds nothing, and a backend decides from what its
 * surface offers.
 */
#ifndef PS3RECOMP_RSX_HDR_DISPLAY_H
#define PS3RECOMP_RSX_HDR_DISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

/* found: the display was found; hdr: HDR is on for it. peak, white in nits,
 * 0 when not known. */
typedef struct { int found, hdr; float peak, white; } rsx_hdr_display;

/* The display the window (an HWND on Windows) is on. */
rsx_hdr_display rsx_hdr_display_query(void* window);

/* 1 when the answer of the last query may have changed: the displays did
 * (HDR switched in Windows, a display added or set up anew) or the window
 * moved to another. Cheap: for a poll as the title runs. */
int rsx_hdr_display_stale(void* window);

void rsx_hdr_display_release(void);

#ifdef __cplusplus
}
#endif

#endif
