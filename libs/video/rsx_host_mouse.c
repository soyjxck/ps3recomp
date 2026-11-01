/*
 * The presenting window's mouse: raw motion, the wheel, and the cursor's
 * capture. See rsx_host_mouse.h.
 */
#include "rsx_host_mouse.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#define MSG_APPLY (WM_APP + 0x2D3)   /* the capture request changed: apply it on the window's thread */

static HWND s_hwnd;
static volatile LONG s_want;       /* the port asked for the cursor */
static volatile LONG s_held;       /* and has it: the window is in front */
static volatile LONG s_dx, s_dy, s_wheel;
static volatile LONG s_escape_closes = 1;
static volatile LONG s_frame_click;   /* the window's frame (its title bar) is held */
static volatile LONG s_modal;         /* Windows' own loop: the window moved or sized, its menu open */

/* A mouse button down (the buttons as they are, swapped or not). */
static int button_down(void)
{
    return ((GetAsyncKeyState(VK_LBUTTON) | GetAsyncKeyState(VK_RBUTTON) | GetAsyncKeyState(VK_MBUTTON)) & 0x8000) != 0;
}

/* On the window's thread: take the cursor or give it back. Not while the
 * player has the window's frame in hand -- held inside the window, the
 * pointer would pull the window after it as it is dragged, down without end
 * -- nor while the click that brought the window in front is still down: it
 * is taken when that button is let go. (SDL waits the same way: its
 * in_title_click, in_modal_loop and focus_click_pending.) */
static void mouse_apply(HWND hwnd)
{
    const int may = s_want && GetForegroundWindow() == hwnd && !IsIconic(hwnd) && !s_frame_click && !s_modal;
    const int hold = may && (s_held || !button_down());
    if (hold) {
        RECT rc;
        POINT tl, br;
        GetClientRect(hwnd, &rc);
        tl.x = rc.left; tl.y = rc.top; br.x = rc.right; br.y = rc.bottom;
        ClientToScreen(hwnd, &tl);
        ClientToScreen(hwnd, &br);
        rc.left = tl.x; rc.top = tl.y; rc.right = br.x; rc.bottom = br.y;
        ClipCursor(&rc);
        SetCursor(NULL);
        if (!s_held) {   /* what the mouse did while it was someone else's is not ours */
            InterlockedExchange(&s_dx, 0);
            InterlockedExchange(&s_dy, 0);
            InterlockedExchange(&s_wheel, 0);
        }
        InterlockedExchange(&s_held, 1);
    } else {
        if (s_held) ClipCursor(NULL);
        InterlockedExchange(&s_held, 0);
    }
}

void rsx_host_mouse_attach(void* hwnd)
{
    RAWINPUTDEVICE rid;
    s_hwnd = (HWND)hwnd;
    rid.usUsagePage = 0x01;   /* generic desktop */
    rid.usUsage = 0x02;       /* mouse */
    rid.dwFlags = 0;
    rid.hwndTarget = s_hwnd;
    RegisterRawInputDevices(&rid, 1, sizeof rid);
    mouse_apply(s_hwnd);
}

int rsx_host_mouse_message(void* hwnd_, unsigned msg, uintptr_t wp, intptr_t lp, intptr_t* result)
{
    HWND hwnd = (HWND)hwnd_;
    switch (msg) {
    case WM_INPUT: {
        RAWINPUT ri;
        UINT size = sizeof ri;
        if ((s_held || s_want) &&
            GetRawInputData((HRAWINPUT)lp, RID_INPUT, &ri, &size, sizeof(RAWINPUTHEADER)) != (UINT)-1 &&
            ri.header.dwType == RIM_TYPEMOUSE) {
            if (!s_held) {   /* the click that brought the window in front let go, wherever the pointer is */
                if (ri.data.mouse.usButtonFlags &
                    (RI_MOUSE_LEFT_BUTTON_UP | RI_MOUSE_RIGHT_BUTTON_UP | RI_MOUSE_MIDDLE_BUTTON_UP))
                    mouse_apply(hwnd);
                if (!s_held) return 0;
            }
            if (!(ri.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) {
                InterlockedExchangeAdd(&s_dx, ri.data.mouse.lLastX);
                InterlockedExchangeAdd(&s_dy, ri.data.mouse.lLastY);
            }
            if (ri.data.mouse.usButtonFlags & RI_MOUSE_WHEEL)
                InterlockedExchangeAdd(&s_wheel, (SHORT)ri.data.mouse.usButtonData);
        }
        return 0;   /* DefWindowProc still has its cleanup to do */
    }
    case WM_SETCURSOR:
        if (s_held && LOWORD(lp) == HTCLIENT) {
            SetCursor(NULL);
            *result = TRUE;
            return 1;
        }
        return 0;
    case WM_NCLBUTTONDOWN:   /* the frame: Windows takes the mouse until it is let go */
        InterlockedExchange(&s_frame_click, 1);
        mouse_apply(hwnd);
        return 0;
    case WM_CAPTURECHANGED:
        InterlockedExchange(&s_frame_click, 0);
        mouse_apply(hwnd);
        return 0;
    case WM_ENTERSIZEMOVE:
    case WM_ENTERMENULOOP:
        InterlockedExchange(&s_modal, 1);
        mouse_apply(hwnd);
        return 0;
    case WM_EXITSIZEMOVE:
    case WM_EXITMENULOOP:
        InterlockedExchange(&s_modal, 0);
        InterlockedExchange(&s_frame_click, 0);
        mouse_apply(hwnd);
        return 0;
    case WM_LBUTTONUP:
    case WM_ACTIVATE:
    case WM_SETFOCUS:
    case WM_KILLFOCUS:
    case WM_MOVE:
    case WM_SIZE:
        mouse_apply(hwnd);
        return 0;
    case MSG_APPLY:
        mouse_apply(hwnd);
        *result = 0;
        return 1;
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE && !s_escape_closes) {
            *result = 0;
            return 1;
        }
        return 0;
    default:
        return 0;
    }
}

void rsx_host_mouse_capture(int on)
{
    if (InterlockedExchange(&s_want, on ? 1 : 0) == (on ? 1 : 0)) return;
    if (s_hwnd) PostMessageA(s_hwnd, MSG_APPLY, 0, 0);
}

int rsx_host_mouse_captured(void) { return (int)s_held; }

void rsx_host_mouse_take_motion(int* dx, int* dy)
{
    *dx = (int)InterlockedExchange(&s_dx, 0);
    *dy = (int)InterlockedExchange(&s_dy, 0);
}

int rsx_host_mouse_take_wheel(void)
{
    /* In WHEEL_DELTAs of a notch: a wheel that reports finer steps adds up to one. */
    const LONG units = InterlockedExchange(&s_wheel, 0);
    const LONG notches = units / WHEEL_DELTA;
    if (units - notches * WHEEL_DELTA) InterlockedExchangeAdd(&s_wheel, units - notches * WHEEL_DELTA);
    return (int)notches;
}

void rsx_host_window_escape_closes(int on) { InterlockedExchange(&s_escape_closes, on ? 1 : 0); }

#elif defined(__APPLE__)

/* The Metal backend's window: rsx_metal_backend.m has these. */

#elif defined(__linux__) && defined(PS3RECOMP_VULKAN_ENGINE_SDL)

/* The Vulkan engine's SDL window: rsx_vulkan_engine.c has these. */

#else

void rsx_host_mouse_capture(int on) { (void)on; }
int rsx_host_mouse_captured(void) { return 0; }
void rsx_host_mouse_take_motion(int* dx, int* dy) { *dx = 0; *dy = 0; }
int rsx_host_mouse_take_wheel(void) { return 0; }
void rsx_host_window_escape_closes(int on) { (void)on; }
void rsx_host_mouse_attach(void* hwnd) { (void)hwnd; }
int rsx_host_mouse_message(void* hwnd, unsigned msg, uintptr_t wp, intptr_t lp, intptr_t* result)
{
    (void)hwnd; (void)msg; (void)wp; (void)lp; (void)result;
    return 0;
}

#endif
