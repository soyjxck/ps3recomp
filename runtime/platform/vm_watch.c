/*
 * ps3recomp - write-watch on guest memory (see include/ps3emu/vm_watch.h)
 */
#include "ps3emu/vm_watch.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#define WP_SHIFT      12u                     /* 4 KiB pages                   */
#define WP_COUNT      (1u << 20)              /* the 32-bit guest space        */
#define WP_QUARANTINE 2000u                   /* ms a written page stays open  */

static uint8_t*       s_base;
static uint64_t       s_size;
static int            s_on;
static volatile LONG* s_gen;      /* writes seen, per page                     */
static uint8_t*       s_prot;     /* 1: read-only now                          */
static uint32_t*      s_quar;     /* GetTickCount64 of the last write, or 0    */
static volatile LONGLONG s_faults, s_protected;

void vm_watch_init(uint8_t* base, uint64_t size)
{
    if (!base || !size) return;
    s_gen  = (volatile LONG*)calloc(WP_COUNT, sizeof *s_gen);
    s_prot = (uint8_t*)calloc(WP_COUNT, 1);
    s_quar = (uint32_t*)calloc(WP_COUNT, sizeof *s_quar);
    if (!s_gen || !s_prot || !s_quar) return;
    s_base = base; s_size = size;
    { const char* e = getenv("RSX_TEX_WATCH"); s_on = !(e && *e == '0'); }
}

int vm_watch_available(void) { return s_on; }

static int wp_protect_run(uint32_t p0, uint32_t p1)   /* pages [p0, p1] */
{
    uint8_t* a = s_base + ((uint64_t)p0 << WP_SHIFT);
    const SIZE_T n = (SIZE_T)(p1 - p0 + 1) << WP_SHIFT;
    /* The VM is committed on first touch; a reserved page cannot be
     * protected. Committing the 64 KiB around the run first is idempotent
     * for the pages that already are. */
    uint8_t* c0 = (uint8_t*)((uintptr_t)a & ~(uintptr_t)0xFFFF);
    uint8_t* c1 = (uint8_t*)(((uintptr_t)a + n + 0xFFFF) & ~(uintptr_t)0xFFFF);
    if (!VirtualAlloc(c0, (SIZE_T)(c1 - c0), MEM_COMMIT, PAGE_READWRITE)) return 0;
    DWORD old;
    if (!VirtualProtect(a, n, PAGE_READONLY, &old)) return 0;
    for (uint32_t p = p0; p <= p1; p++) s_prot[p] = 1;
    InterlockedAdd64(&s_protected, (LONGLONG)(p1 - p0 + 1));
    return 1;
}

uint64_t vm_watch_arm(uint32_t ea, uint32_t len, uint32_t* unwatched)
{
    uint32_t un = 0;
    uint64_t stamp = 0;
    if (!s_on || !len) { if (unwatched) *unwatched = 1; return 0; }
    const uint32_t now = (uint32_t)GetTickCount64();
    const uint32_t p0 = ea >> WP_SHIFT;
    const uint32_t p1 = (uint32_t)(((uint64_t)ea + len - 1) >> WP_SHIFT);
    uint32_t run0 = 0; int in_run = 0;
    for (uint32_t p = p0; p <= p1; p++) {
        stamp += (uint64_t)(uint32_t)s_gen[p] * (uint64_t)(p - p0 + 1);
        int ok = s_prot[p];
        if (!ok) {
            const uint32_t q = s_quar[p];
            if (q && now - q < WP_QUARANTINE) { un++; }
            else if (!in_run) { in_run = 1; run0 = p; }
            if (in_run && (q && now - q < WP_QUARANTINE)) { if (!wp_protect_run(run0, p - 1)) un += p - run0; in_run = 0; }
            continue;
        }
        if (in_run) { if (!wp_protect_run(run0, p - 1)) un += p - run0; in_run = 0; }
    }
    if (in_run && !wp_protect_run(run0, p1)) un += p1 - run0 + 1;
    if (unwatched) *unwatched = un;
    return stamp;
}

void vm_watch_touch(uint32_t ea, uint32_t len)
{
    if (!s_on || !len) return;
    const uint32_t now = (uint32_t)GetTickCount64();
    const uint32_t p0 = ea >> WP_SHIFT;
    const uint32_t p1 = (uint32_t)(((uint64_t)ea + len - 1) >> WP_SHIFT);
    for (uint32_t p = p0; p <= p1; p++) {
        if (s_prot[p]) {
            DWORD old;
            VirtualProtect(s_base + ((uint64_t)p << WP_SHIFT), 1u << WP_SHIFT, PAGE_READWRITE, &old);
            s_prot[p] = 0;
        }
        InterlockedIncrement(&s_gen[p]);
        s_quar[p] = now ? now : 1u;
    }
}

int vm_watch_fault(uintptr_t addr, int is_write)
{
    if (!s_on || addr < (uintptr_t)s_base || addr >= (uintptr_t)s_base + s_size) return 0;
    const uint32_t p = (uint32_t)((addr - (uintptr_t)s_base) >> WP_SHIFT);
    if (p >= WP_COUNT || !s_prot[p]) return 0;
    (void)is_write;   /* a read cannot fault on a read-only page; whatever it was, open it */
    DWORD old;
    if (!VirtualProtect(s_base + ((uint64_t)p << WP_SHIFT), 1u << WP_SHIFT, PAGE_READWRITE, &old)) return 0;
    s_prot[p] = 0;
    InterlockedIncrement(&s_gen[p]);
    { const uint32_t now = (uint32_t)GetTickCount64(); s_quar[p] = now ? now : 1u; }
    InterlockedIncrement64(&s_faults);
    return 1;
}

void vm_watch_stats(uint64_t* faults, uint64_t* pages_protected)
{
    if (faults) *faults = (uint64_t)s_faults;
    if (pages_protected) *pages_protected = (uint64_t)s_protected;
}

#else   /* not Windows: the engine hashes as before */

void     vm_watch_init(uint8_t* base, uint64_t size) { (void)base; (void)size; }
int      vm_watch_available(void) { return 0; }
uint64_t vm_watch_arm(uint32_t ea, uint32_t len, uint32_t* unwatched) { (void)ea; (void)len; if (unwatched) *unwatched = 1; return 0; }
void     vm_watch_touch(uint32_t ea, uint32_t len) { (void)ea; (void)len; }
int      vm_watch_fault(uintptr_t addr, int is_write) { (void)addr; (void)is_write; return 0; }
void     vm_watch_stats(uint64_t* faults, uint64_t* pages_protected) { if (faults) *faults = 0; if (pages_protected) *pages_protected = 0; }

#endif
