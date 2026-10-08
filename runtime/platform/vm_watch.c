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
     * protected, so commit the run first. Only the run: MEM_COMMIT on pages
     * that are already committed RESETS their protection to the one given,
     * so committing the 64 KiB around the run (as this once did) silently
     * re-opened protected pages of neighbouring textures. Their writes then
     * never faulted, the engine never re-hashed them, and they kept stale
     * contents (black barrels in the town). The run's own pages are open
     * already, so committing them read-write changes nothing. */
    if (!vm_commit_reserved(a, n)) return 0;
    /* Marked before the protection takes effect: a write that faults the
     * instant it does must find the page marked, or the fault handler would
     * not know it as the watch's. */
    for (uint32_t p = p0; p <= p1; p++) s_prot[p] = 1;
    DWORD old;
    if (!VirtualProtect(a, n, PAGE_READONLY, &old)) {
        for (uint32_t p = p0; p <= p1; p++) s_prot[p] = 0;
        return 0;
    }
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

int vm_commit_reserved(void* p, uint64_t n)
{
    uint8_t* a = (uint8_t*)p;
    uint8_t* const end = a + n;
    while (a < end) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery(a, &mi, sizeof mi)) return 0;
        uint8_t* r_end = (uint8_t*)mi.BaseAddress + mi.RegionSize;
        if (r_end > end) r_end = end;
        if (mi.State == MEM_FREE) return 0;
        if (mi.State == MEM_RESERVE &&
            !VirtualAlloc(a, (SIZE_T)(r_end - a), MEM_COMMIT, PAGE_READWRITE)) return 0;
        a = r_end;
    }
    return 1;
}

void vm_watch_stats(uint64_t* faults, uint64_t* pages_protected)
{
    if (faults) *faults = (uint64_t)s_faults;
    if (pages_protected) *pages_protected = (uint64_t)s_protected;
}

#else   /* POSIX: mprotect, and faults through the vectored-handler shim */

#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* The same watch on the host's own pages: 4 KiB on x86-64, 16 KiB on Apple
 * silicon -- coarser there, so a texture sharing a page with data written
 * every frame is quarantined (and hashed) more often; nothing else changes.
 * A write to a watched page raises SIGBUS (macOS) or SIGSEGV (Linux); the
 * host's handler (main.cpp, through win32_compat's vectored-handler shim)
 * calls vm_watch_fault, which opens the page and lets the store re-run.
 * Everything a fault handler calls here -- mprotect, clock_gettime, atomic
 * increments -- is async-signal-safe. */
#define WP_QUARANTINE 2000u                   /* ms a written page stays open  */

static uint8_t*           s_base;
static uint64_t           s_size;
static int                s_on;
static unsigned           s_shift;           /* log2 of the host page size    */
static uint32_t           s_count;           /* host pages in the guest space */
static volatile uint32_t* s_gen;             /* writes seen, per page         */
static volatile uint8_t*  s_prot;            /* 1: read-only now              */
static volatile uint32_t* s_quar;            /* ms stamp of the last write, 0 */
static volatile uint64_t  s_faults, s_protected;

static uint32_t wp_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint32_t ms = (uint32_t)((uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u);
    return ms ? ms : 1u;
}

void vm_watch_init(uint8_t* base, uint64_t size)
{
    if (!base || !size) return;
    const long pg = sysconf(_SC_PAGESIZE);
    unsigned sh = 0;
    while (sh < 30 && (1L << sh) < pg) sh++;
    if ((1L << sh) != pg || ((uintptr_t)base & (uintptr_t)(pg - 1))) return;   /* base must be page aligned */
    s_shift = sh;
    s_count = (uint32_t)(0x100000000ull >> sh);
    s_gen  = (volatile uint32_t*)calloc(s_count, sizeof *s_gen);
    s_prot = (volatile uint8_t*)calloc(s_count, 1);
    s_quar = (volatile uint32_t*)calloc(s_count, sizeof *s_quar);
    if (!s_gen || !s_prot || !s_quar) return;
    s_base = base; s_size = size;
    { const char* e = getenv("RSX_TEX_WATCH"); s_on = !(e && *e == '0'); }
}

int vm_watch_available(void) { return s_on; }

static int wp_protect_run(uint32_t p0, uint32_t p1)   /* pages [p0, p1] */
{
    const uint64_t off = (uint64_t)p0 << s_shift, n = (uint64_t)(p1 - p0 + 1) << s_shift;
    if (off + n > s_size) return 0;                    /* past what is mapped */
    /* Marked before the protection takes effect: a write that faults the
     * instant it does must find the page marked. */
    for (uint32_t p = p0; p <= p1; p++) s_prot[p] = 1;
    if (mprotect(s_base + off, (size_t)n, PROT_READ) != 0) {
        for (uint32_t p = p0; p <= p1; p++) s_prot[p] = 0;
        return 0;
    }
    __atomic_add_fetch(&s_protected, (uint64_t)(p1 - p0 + 1), __ATOMIC_RELAXED);
    return 1;
}

uint64_t vm_watch_arm(uint32_t ea, uint32_t len, uint32_t* unwatched)
{
    uint32_t un = 0;
    uint64_t stamp = 0;
    if (!s_on || !len) { if (unwatched) *unwatched = 1; return 0; }
    const uint32_t now = wp_now_ms();
    const uint32_t p0 = ea >> s_shift;
    const uint32_t p1 = (uint32_t)(((uint64_t)ea + len - 1) >> s_shift);
    uint32_t run0 = 0; int in_run = 0;
    for (uint32_t p = p0; p <= p1; p++) {
        stamp += (uint64_t)__atomic_load_n(&s_gen[p], __ATOMIC_ACQUIRE) * (uint64_t)(p - p0 + 1);
        if (!s_prot[p]) {
            const uint32_t q = s_quar[p];
            const int quarantined = q && now - q < WP_QUARANTINE;
            if (quarantined) un++;
            else if (!in_run) { in_run = 1; run0 = p; }
            if (in_run && quarantined) { if (!wp_protect_run(run0, p - 1)) un += p - run0; in_run = 0; }
            continue;
        }
        if (in_run) { if (!wp_protect_run(run0, p - 1)) un += p - run0; in_run = 0; }
    }
    if (in_run && !wp_protect_run(run0, p1)) un += p1 - run0 + 1;
    if (unwatched) *unwatched = un;
    return stamp;
}

static void wp_open(uint32_t p, uint32_t now)
{
    mprotect(s_base + ((uint64_t)p << s_shift), (size_t)1 << s_shift, PROT_READ | PROT_WRITE);
    s_quar[p] = now;          /* before the flag drops: a racing fault sees a written page */
    s_prot[p] = 0;
    __atomic_add_fetch(&s_gen[p], 1u, __ATOMIC_RELEASE);
}

void vm_watch_touch(uint32_t ea, uint32_t len)
{
    if (!s_on || !len) return;
    const uint32_t now = wp_now_ms();
    const uint32_t p0 = ea >> s_shift;
    const uint32_t p1 = (uint32_t)(((uint64_t)ea + len - 1) >> s_shift);
    for (uint32_t p = p0; p <= p1; p++) {
        if (s_prot[p]) wp_open(p, now);
        else { __atomic_add_fetch(&s_gen[p], 1u, __ATOMIC_RELEASE); s_quar[p] = now; }
    }
}

int vm_watch_fault(uintptr_t addr, int is_write)
{
    (void)is_write;   /* a read cannot fault on a read-only page; whatever it was, open it */
    if (!s_on || addr < (uintptr_t)s_base || addr >= (uintptr_t)s_base + s_size) return 0;
    const uint32_t p = (uint32_t)((addr - (uintptr_t)s_base) >> s_shift);
    if (p >= s_count) return 0;
    if (s_prot[p]) {
        wp_open(p, wp_now_ms());
        __atomic_add_fetch(&s_faults, 1u, __ATOMIC_RELAXED);
        return 1;
    }
    /* Not protected now, but the watch opened it before: another thread's
     * fault on the same page got there first. Open it again (a no-op if it is
     * open) and let the store re-run. A page the watch never touched is not
     * ours -- a real fault, for whoever comes next. */
    if (s_quar[p]) {
        mprotect(s_base + ((uint64_t)p << s_shift), (size_t)1 << s_shift, PROT_READ | PROT_WRITE);
        return 1;
    }
    return 0;
}

void vm_watch_stats(uint64_t* faults, uint64_t* pages_protected)
{
    if (faults) *faults = s_faults;
    if (pages_protected) *pages_protected = s_protected;
}

/* POSIX guest memory is mapped read-write up front; nothing to commit. */
int vm_commit_reserved(void* p, uint64_t n) { (void)p; (void)n; return 1; }

#endif
