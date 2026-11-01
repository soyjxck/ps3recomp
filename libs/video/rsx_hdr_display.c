/*
 * The display under a window, for HDR output (rsx_hdr_display.h).
 */
#include "rsx_hdr_display.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

/* COM through C: obj->lpVtbl->Method(obj, ...). */
#define CALL(o, m, ...) ((o)->lpVtbl->m((o), __VA_ARGS__))
#define CALL0(o, m)     ((o)->lpVtbl->m((o)))
#define RELEASE(p)      do { if (p) { (p)->lpVtbl->Release(p); (p) = NULL; } } while (0)

static IDXGIFactory1* s_factory;   /* stale (IsCurrent) once the displays change */
static HMONITOR s_monitor;

/* Windows' SDR white level for the display gdi names (\\.\DISPLAYn), in nits;
 * 0 when it cannot be read. */
static float sdr_white_nits(const wchar_t* gdi)
{
    UINT32 np = 0, nm = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &np, &nm) != ERROR_SUCCESS || !np) return 0.0f;
    DISPLAYCONFIG_PATH_INFO* paths = (DISPLAYCONFIG_PATH_INFO*)calloc(np, sizeof *paths);
    DISPLAYCONFIG_MODE_INFO* modes = (DISPLAYCONFIG_MODE_INFO*)calloc(nm ? nm : 1, sizeof *modes);
    float nits = 0.0f;
    if (paths && modes && QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &np, paths, &nm, modes, NULL) == ERROR_SUCCESS) {
        for (UINT32 i = 0; i < np && nits == 0.0f; i++) {
            DISPLAYCONFIG_SOURCE_DEVICE_NAME src;
            memset(&src, 0, sizeof src);
            src.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            src.header.size = sizeof src;
            src.header.adapterId = paths[i].sourceInfo.adapterId;
            src.header.id = paths[i].sourceInfo.id;
            if (DisplayConfigGetDeviceInfo(&src.header) != ERROR_SUCCESS || wcscmp(src.viewGdiDeviceName, gdi) != 0) continue;
            DISPLAYCONFIG_SDR_WHITE_LEVEL wl;
            memset(&wl, 0, sizeof wl);
            wl.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
            wl.header.size = sizeof wl;
            wl.header.adapterId = paths[i].targetInfo.adapterId;
            wl.header.id = paths[i].targetInfo.id;
            if (DisplayConfigGetDeviceInfo(&wl.header) == ERROR_SUCCESS && wl.SDRWhiteLevel)
                nits = (float)wl.SDRWhiteLevel / 1000.0f * 80.0f;   /* in thousandths of 80 nits */
        }
    }
    free(paths); free(modes);
    return nits;
}

/* Keeps the factory it asked, for rsx_hdr_display_stale. */
rsx_hdr_display rsx_hdr_display_query(void* window)
{
    rsx_hdr_display d = { 0, 0, 0.0f, 0.0f };
    if (!window) return d;
    HMONITOR mon = MonitorFromWindow((HWND)window, MONITOR_DEFAULTTONEAREST);
    RELEASE(s_factory);
    if (!mon || FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void**)&s_factory))) return d;
    s_monitor = mon;
    IDXGIAdapter1* ad = NULL;
    for (UINT a = 0; !d.found && CALL(s_factory, EnumAdapters1, a, &ad) != DXGI_ERROR_NOT_FOUND; a++) {
        IDXGIOutput* out = NULL;
        for (UINT o = 0; !d.found && CALL(ad, EnumOutputs, o, &out) != DXGI_ERROR_NOT_FOUND; o++) {
            DXGI_OUTPUT_DESC od;
            IDXGIOutput6* o6 = NULL;
            if (SUCCEEDED(CALL(out, GetDesc, &od)) && od.Monitor == mon &&
                SUCCEEDED(CALL(out, QueryInterface, &IID_IDXGIOutput6, (void**)&o6))) {
                DXGI_OUTPUT_DESC1 d1;
                if (SUCCEEDED(CALL(o6, GetDesc1, &d1))) {
                    d.found = 1;
                    d.hdr = d1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
                    d.peak = d1.MaxLuminance;
                    d.white = sdr_white_nits(od.DeviceName);
                }
                RELEASE(o6);
            }
            RELEASE(out);
        }
        RELEASE(ad);
    }
    return d;
}

int rsx_hdr_display_stale(void* window)
{
    return !s_factory || !CALL0(s_factory, IsCurrent) ||
           MonitorFromWindow((HWND)window, MONITOR_DEFAULTTONEAREST) != s_monitor;
}

void rsx_hdr_display_release(void) { RELEASE(s_factory); s_monitor = NULL; }

#else

rsx_hdr_display rsx_hdr_display_query(void* window)
{
    (void)window;
    const rsx_hdr_display d = { 0, 0, 0.0f, 0.0f };
    return d;
}
int rsx_hdr_display_stale(void* window) { (void)window; return 0; }
void rsx_hdr_display_release(void) {}

#endif
