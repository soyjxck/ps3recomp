/*
 * ps3recomp - Timer and time syscalls (implementation)
 */

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE /* tm_gmtoff on glibc */
#endif
#include "sys_timer.h"
#include "sys_ppu_thread.h"   /* ppu_thread_note_block */
#include <time.h>
#include "sys_event.h"
#include "../memory/vm.h"
#include <stdlib.h>   /* getenv -- an implicit decl returns int, truncating the pointer */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* Environment switches are read where they are used, some of them per call:
 * answer from the pointer-keyed cache (ps3emu/env_cache.h). */
#include "ps3emu/env_cache.h"
#define getenv(name) ps3_env(name)
#include "ps3emu/log_first.h"
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif
#include "ps3emu/thread_name.h"

/* ---------------------------------------------------------------------------
 * Globals
 * -----------------------------------------------------------------------*/
sys_timer_info g_sys_timers[SYS_TIMER_MAX];

#ifdef _WIN32
static LARGE_INTEGER s_qpc_freq;
static int           s_qpc_init = 0;

static void ensure_qpc_init(void)
{
    if (!s_qpc_init) {
        QueryPerformanceFrequency(&s_qpc_freq);
        s_qpc_init = 1;
    }
}

int64_t lv2_usec_deadline(uint64_t usec)
{
    LARGE_INTEGER now;
    ensure_qpc_init();
    QueryPerformanceCounter(&now);
    return now.QuadPart +
        (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
}

int lv2_deadline_passed(int64_t deadline)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart >= deadline;
}
#endif

/* ---------------------------------------------------------------------------
 * PPU timebase (mftb/mftbu) -- THE guest clock.
 *
 * One global monotonic counter scaled to the PS3 timebase (79.8 MHz),
 * anchored at first use. The old lifter emission was a per-call-site
 * static that advanced 16667 ticks PER READ -- not a clock at all: every
 * guest timing loop (media pacers, throttles, profilers) computed garbage
 * from it, and each call site had a PRIVATE counter that only moved when
 * polled. Overflow-safe split multiply (rem < qpf, so
 * rem * 79.8e6 < ~8e14 << 2^63).
 * -----------------------------------------------------------------------*/
uint64_t ppu_timebase_now(void)
{
#ifdef _WIN32
    static LONGLONG t0 = 0;
    ensure_qpc_init();
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (!t0) t0 = now.QuadPart;   /* benign race: same anchor either way */
    const int64_t g = ps3_guest_ticks(now.QuadPart);
    uint64_t d = g > t0 ? (uint64_t)(g - t0) : 0;
    uint64_t q = (uint64_t)s_qpc_freq.QuadPart;
    return (d / q) * PS3_TIMEBASE_FREQ + (d % q) * PS3_TIMEBASE_FREQ / q;
#else
    const uint64_t ns = (uint64_t)ps3_guest_ticks((int64_t)host_monotonic_ns());
    return (ns / 1000000000ull) * PS3_TIMEBASE_FREQ +
           (ns % 1000000000ull) * PS3_TIMEBASE_FREQ / 1000000000ull;
#endif
}

#ifndef _WIN32
uint64_t host_monotonic_ns(void)
{
#ifdef __APPLE__
    static uint32_t numer, denom;
    if (!denom) {                       /* benign race: same values either way */
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        numer = tb.numer; denom = tb.denom;
    }
    const uint64_t t = mach_absolute_time();
    /* Split so t * numer cannot overflow (numer is 125 on Apple silicon). */
    return (t / denom) * numer + (t % denom) * numer / denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}
#endif

/* Guest time = host time - the time spent paused, held at s_guest_hold while
 * paused. Readers load the hold before the paused total and the resume
 * stores the total before clearing the hold, so a reader racing the resume
 * still clamps to the hold: the clocks never step back. */
static int64_t s_guest_paused;               /* host ticks spent paused */
static int64_t s_guest_hold = INT64_MAX;     /* while paused: the guest ticks the clocks hold at */

int64_t ps3_guest_ticks(int64_t host)
{
    const int64_t hold = __atomic_load_n(&s_guest_hold, __ATOMIC_ACQUIRE);
    const int64_t t = host - __atomic_load_n(&s_guest_paused, __ATOMIC_ACQUIRE);
    return t < hold ? t : hold;
}

void ps3_guest_clock_pause(int paused)
{
#ifdef _WIN32
    LARGE_INTEGER c;
    ensure_qpc_init();
    QueryPerformanceCounter(&c);
    const int64_t now = c.QuadPart, ms = s_qpc_freq.QuadPart / 1000;
#else
    const int64_t now = (int64_t)host_monotonic_ns(), ms = 1000000;
#endif
    const int64_t hold = __atomic_load_n(&s_guest_hold, __ATOMIC_ACQUIRE);
    if (paused && hold == INT64_MAX) {
        /* A millisecond on: a reader that took its host time just before this
         * and has not seen the hold yet cannot pass it, then step back. */
        __atomic_store_n(&s_guest_hold, now + ms - s_guest_paused, __ATOMIC_RELEASE);
    } else if (!paused && hold != INT64_MAX) {
        __atomic_store_n(&s_guest_paused, now - hold, __ATOMIC_RELEASE);   /* on from the hold */
        __atomic_store_n(&s_guest_hold, INT64_MAX, __ATOMIC_RELEASE);
    }
}

static void write_be32(uint32_t addr, uint32_t val)
{
    uint32_t* p = (uint32_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
          ((val <<  8) & 0xFF0000) | ((val << 24) & 0xFF000000u);
#endif
    *p = val;
}

static void write_be64(uint32_t addr, uint64_t val)
{
    uint64_t* p = (uint64_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 56) & 0xFFULL) |
          ((val >> 40) & 0xFF00ULL) |
          ((val >> 24) & 0xFF0000ULL) |
          ((val >>  8) & 0xFF000000ULL) |
          ((val <<  8) & 0xFF00000000ULL) |
          ((val << 24) & 0xFF0000000000ULL) |
          ((val << 40) & 0xFF000000000000ULL) |
          ((val << 56) & 0xFF00000000000000ULL);
#endif
    *p = val;
}

/* ---------------------------------------------------------------------------
 * sys_timer_usleep
 *
 * r3 = microseconds
 * -----------------------------------------------------------------------*/
/* Called on every guest usleep before it sleeps, when set, with the guest
 * return address and the requested time. A guest thread polling with usleep
 * is usually waiting on something another host thread must produce -- the
 * port's FIFO walker writing a fence label -- so the hook can wake that
 * thread, and for a poll it recognises it may do the waiting itself (until
 * whatever it waits on has had a chance to change, at most `usec`): it then
 * returns nonzero and the full sleep is skipped. */
int (*g_lv2_usleep_hook)(uint32_t lr, uint64_t usec) = 0;

/* Called first on every guest usleep, when set: the title may use the time
 * the guest asked to sleep (run deferred work on this thread, ctx as it is)
 * and shorten `*usec` by what that took. */
void (*g_lv2_usleep_pre)(ppu_context* ctx, uint64_t* usec) = 0;

/* Windows: sleeps from this many microseconds up to 1 ms wait on a timer
 * and spin only their end (DOD3_AB=usleepspin moves it past 1000). */
int g_usleep_spin_below = 500;

int64_t sys_timer_usleep(ppu_context* ctx)
{
    uint64_t usec = LV2_ARG_U64(ctx, 0);
    ppu_thread_note_block(ctx->thread_id);   /* PS3_PRIO_START */
    if (g_lv2_usleep_pre) g_lv2_usleep_pre(ctx, &usec);
    if (g_lv2_usleep_hook && g_lv2_usleep_hook((uint32_t)ctx->lr, usec)) return CELL_OK;
    /* A single sleep longer than a second is almost always a computed delay
     * gone wrong (a negative interval, a timebase in the wrong units) -- and a
     * thread asleep for an hour looks exactly like a hang. Say so. */
    if (usec > 1000000ull) { static int n = 0; if (n < 32 && ++n)
        fprintf(stderr, "[timer] LONG usleep(%llu us = %.1f s) lr=0x%08llX tid=%u\n",
                (unsigned long long)usec, usec / 1e6, (unsigned long long)ctx->lr, (unsigned)ctx->thread_id); }
    { extern unsigned long long ps3_qpc_us(void);
      static int n=0; if (PS3_LOG_FIRST(n, 60))
        fprintf(stderr, "[WAIT] t=%lluus timer_usleep(%llu us) lr=0x%08llX cia=0x%08llX\n", ps3_qpc_us(),
        (unsigned long long)usec, (unsigned long long)ctx->lr, (unsigned long long)ctx->cia); }
    /* PS3_POLLTOP=<seconds>: name the usleep poll sites.
     *
     * A title that is "hung" is usually not blocked on anything -- every thread
     * is spinning on sys_timer_usleep waiting for a flag, and [BLOCKSUM] only
     * says "syscall 141, 12s". This histograms the callers over a window and
     * prints the top ones, so the answer is a guest address rather than a
     * count. Keyed on ctx->lr: a host-stack backtrace through lifted code has
     * no unwind tables and invents chains (see the Guitar Hero III writeup),
     * and lr is the only guest chain that survives.
     *
     * Deliberately unlocked: the counts are a diagnostic, and a lost increment
     * under a race cannot change which site is at the top of a 30k-per-second
     * poll. */
    { static long s_pt = -1;
      if (s_pt < 0) { const char* e = getenv("PS3_POLLTOP");
                      s_pt = e ? (long)strtoul(e, 0, 10) : 0;
                      if (s_pt < 0) s_pt = 0; }
      if (s_pt > 0) {
        enum { PT_MAX = 128 };
        static uint32_t pt_lr[PT_MAX]; static uint32_t pt_n[PT_MAX];
        static uint64_t pt_us[PT_MAX]; static int pt_used = 0;
        static uint32_t pt_tid[PT_MAX];
        static unsigned long long pt_win = 0;
        uint32_t lr = (uint32_t)ctx->lr;
        int i = 0;
        for (; i < pt_used; i++) if (pt_lr[i] == lr) break;
        if (i == pt_used && pt_used < PT_MAX) {
            pt_lr[pt_used] = lr;
            /* The thread matters as much as the site: "who is waiting" is the
             * half that says whether a poll is the engine idling or the main
             * thread stuck. */
            pt_tid[pt_used] = (uint32_t)ctx->thread_id;
            pt_used++;
        }
        if (i < PT_MAX) { pt_n[i]++; pt_us[i] += usec; }
        { extern unsigned long long ps3_qpc_us(void);
          unsigned long long now = ps3_qpc_us();
          if (!pt_win) pt_win = now;
          else if (now - pt_win >= (unsigned long long)s_pt * 1000000ull) {
            fprintf(stderr, "--- usleep poll sites (last %lds) ---\n", s_pt);
            for (int k = 0; k < 8; k++) {
              int best = -1;
              for (int j = 0; j < pt_used; j++)
                if (pt_n[j] && (best < 0 || pt_n[j] > pt_n[best])) best = j;
              if (best < 0) break;
              fprintf(stderr, "  %8u calls  %6llums slept  tid=%-3u lr=0x%08X\n",
                      pt_n[best], (unsigned long long)(pt_us[best] / 1000),
                      pt_tid[best], pt_lr[best]);
              pt_n[best] = 0;
            }
            fflush(stderr);
            for (int j = 0; j < pt_used; j++) { pt_n[j] = 0; pt_us[j] = 0; }
            pt_win = now;
          } } } }

    /* PS3_WAIT_OBJ=<lr-hex>: when a usleep spin is reached from this return
     * address, dump the registers and the object they point at. A poll loop
     * tells you WHERE it is spinning; this tells you WHAT it is spinning on,
     * which is the part you actually need to find who never releases it. */
    { static long s_wo = -1;
      if (s_wo < 0) { const char* e = getenv("PS3_WAIT_OBJ");
                      s_wo = e ? (long)strtoul(e, 0, 16) : 0; }
      if (s_wo && (uint32_t)ctx->lr == (uint32_t)s_wo) {
        static int _n = 0;
        if (PS3_LOG_FIRST(_n, 8)) {
          /* Dump the words at EVERY plausible object register, not just r29.
           * Which register holds the object is per-title -- r29 was right for
           * the title this was written for and is a spin COUNTER in Guitar
           * Hero III, where the object is in r30/r31. Printing one guess makes
           * the tool silently report the wrong memory as "the thing it waits
           * on", which is worse than printing nothing. */
          fprintf(stderr, "[wait-obj] lr=0x%08X sleep=%lluus", (uint32_t)ctx->lr,
                  (unsigned long long)usec);
          static const int regs[] = { 28, 29, 30, 31, 3 };
          for (unsigned ri = 0; ri < sizeof regs / sizeof regs[0]; ri++) {
            const int r = regs[ri];
            const uint32_t o = (uint32_t)ctx->gpr[r];
            fprintf(stderr, "  r%d=0x%08X", r, o);
            /* Only deref something that looks like a guest pointer: a small
             * integer is a count, and dumping "memory at 4" is noise. */
            if (vm_base && o >= 0x10000u) {
              fprintf(stderr, "[");
              for (int w = 0; w < 4; w++) {
                const uint8_t* q = vm_base + o + (unsigned)w * 4u;
                uint32_t v = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) |
                             ((uint32_t)q[2] << 8)  | (uint32_t)q[3];
                fprintf(stderr, "%s%08X", w ? " " : "", v);
              }
              fprintf(stderr, "]");
            }
          }
          /* The label a GPU fence wait polls usually sits behind a pointer
           * in the object (DoD3's func_008B0C70: *(r30+0x50)). */
          { const uint32_t o = (uint32_t)ctx->gpr[30];
            if (vm_base && o >= 0x10000u) {
              const uint8_t* q = vm_base + o + 0x50u;
              const uint32_t pa = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) | ((uint32_t)q[2] << 8) | q[3];
              uint32_t lab = 0;
              if (pa >= 0x10000u) { const uint8_t* l = vm_base + pa;
                  lab = ((uint32_t)l[0] << 24) | ((uint32_t)l[1] << 16) | ((uint32_t)l[2] << 8) | l[3]; }
              fprintf(stderr, "  *(r30+0x50)=0x%08X -> 0x%08X (r29 target=0x%08X)", pa, lab, (uint32_t)ctx->gpr[29]);
            } }
          fprintf(stderr, "\n"); fflush(stderr);
        } } }

    /* POLLSITE: resolve the host chain of the 1ms poller (LBP bringup) to
     * guest functions -- names the stage that is starving. */
    { static int _ps = -1; if (_ps < 0) _ps = getenv("POLLSITE") ? 12 : 0;
      if (_ps > 0 && usec == 1000) { _ps--;
        extern void ppu_log_host_chain(const char*);
        ppu_log_host_chain("usleep1ms"); } }

#ifdef _WIN32
    /* A high-resolution waitable timer, one per thread and kept: a plain
     * waitable timer rounds to the scheduler's clock interval (1 ms with
     * timeBeginPeriod(1), 15.6 ms otherwise, and since Windows 10 2004 the
     * 1 ms is per process and not always honoured), so a guest audio or
     * render loop's usleep(1000) could take 2-16 ms. The HIGH_RESOLUTION
     * flag (Windows 10 1803+) fires within ~0.5 ms regardless; where the
     * flag is unknown the plain timer is the fallback. Creating and closing
     * a kernel object per call was also measurable at this title's rate. */
    if (usec >= 1000) {
        static __declspec(thread) HANDLE timer = NULL;
        if (!timer) {
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
            timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            if (!timer) timer = CreateWaitableTimerW(NULL, TRUE, NULL);
        }
        if (timer) {
            /* PS3_USLEEP_LOG=<ms>: a sleep that asked for, or took, more than
             * that, with its caller and the frame -- whether a long frame was
             * the guest asking to sleep or the sleep running over. */
            static long ul = -1;
            if (ul < 0) { const char* e = getenv("PS3_USLEEP_LOG"); ul = e ? atol(e) : 0; }
            LARGE_INTEGER uq0, uq1, uqf;
            if (ul > 0) { QueryPerformanceFrequency(&uqf); QueryPerformanceCounter(&uq0); }
            LARGE_INTEGER due;
            due.QuadPart = -((LONGLONG)usec * 10); /* 100ns units, negative = relative */
            SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(timer, INFINITE);
            if (ul > 0) {
                QueryPerformanceCounter(&uq1);
                const double took = (double)(uq1.QuadPart - uq0.QuadPart) * 1000.0 / (double)uqf.QuadPart;
                if (usec > (uint64_t)ul * 1000u || took > (double)ul) {
                    extern uint32_t g_rsx_engine_frame;
                    fprintf(stderr, "[usleep] frame %u tid %u lr=0x%08X asked %.2f ms, slept %.2f ms\n",
                            g_rsx_engine_frame, (unsigned)ctx->thread_id, (uint32_t)ctx->lr, usec / 1000.0, took);
                }
            }
        } else {
            Sleep((DWORD)(usec / 1000));
        }
    } else if (usec > 0 && usec >= (uint64_t)g_usleep_spin_below) {
        /* 0.5-1 ms: on the thread's high-resolution timer for all but the
         * last 250 us, spinning only those. The audio system loop's 800 us
         * polls spun two thirds of a core for their whole length, a core a
         * 6-core machine's game, render and RSX threads need. Below
         * g_usleep_spin_below (DOD3_AB=usleepspin sets it past 1000, the old
         * all-spin behaviour) the spin below as before. */
        ensure_qpc_init();
        LARGE_INTEGER q0, qn;
        QueryPerformanceCounter(&q0);
        const int64_t deadline = q0.QuadPart + (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
        static __declspec(thread) HANDLE hr = NULL;
        if (!hr) hr = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (hr) {
            LARGE_INTEGER due;
            due.QuadPart = -((LONGLONG)(usec - 250) * 10);
            SetWaitableTimer(hr, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(hr, INFINITE);
        }
        do {
            SwitchToThread();
            QueryPerformanceCounter(&qn);
        } while (qn.QuadPart < deadline);
    } else if (usec > 0) {
        /* Sub-millisecond sleep. Sleep()/waitable timers round up to the
         * ~0.5-1 ms scheduler quantum, far too coarse for a guest that relies
         * on accurate microsecond pacing -- a short usleep() spin waiting on
         * another thread to make progress otherwise collapses to a single yield
         * and busy-spins at full speed (so the wait is effectively a no-op).
         * Busy-wait to a precise QPC deadline, yielding inside the spin
         * (SwitchToThread) so a sibling host thread the guest may be waiting on
         * still gets the core. Mirrors RPCS3's "Usleep Only" TSC busy-tail. */
        ensure_qpc_init();
        LARGE_INTEGER qpc_start, qpc_now;
        QueryPerformanceCounter(&qpc_start);
        const int64_t qpc_deadline = qpc_start.QuadPart +
            (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
        do {
            SwitchToThread();
            QueryPerformanceCounter(&qpc_now);
        } while (qpc_now.QuadPart < qpc_deadline);
    }
#else
    if (usec > 0) {
        struct timespec ts;
        ts.tv_sec  = (time_t)(usec / 1000000);
        ts.tv_nsec = (long)((usec % 1000000) * 1000);
        nanosleep(&ts, NULL);
    }
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_sleep
 *
 * r3 = seconds
 * -----------------------------------------------------------------------*/
int64_t sys_timer_sleep(ppu_context* ctx)
{
    uint32_t sec = LV2_ARG_U32(ctx, 0);
    ppu_thread_note_block(ctx->thread_id);   /* PS3_PRIO_START */

#ifdef _WIN32
    Sleep(sec * 1000);
#else
    sleep(sec);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_time_get_current_time
 *
 * r3 = pointer to receive seconds (u64*)
 * r4 = pointer to receive nanoseconds (u64*)
 *
 * The wall clock: seconds since 1970, which a title's C library turns into
 * the date it shows and stamps its saves with.
 *
 * Windows: the host's wall clock at the first reading, carried on by the
 * guest's ticks -- so it stands still while the title is paused, like the
 * guest's other clocks (sys_timer.h). It used to be the performance counter
 * alone, divided into seconds: the time since the PC was started, taken for
 * the time since 1970. A machine up for four days told the title it was the
 * 5th of January 1970.
 *
 * PS3_TIME_LOG=1: the first answer of this call and of sys_time_get_timezone,
 * once each.
 * -----------------------------------------------------------------------*/
#ifdef _WIN32
static void wall_clock_now(uint64_t* sec, uint64_t* nsec)
{
    static volatile LONG s_lock = 0;
    static volatile LONG s_have = 0;
    static int64_t  s_base_ticks;
    static uint64_t s_base_100ns;      /* 100 ns units since 1970 */
    LARGE_INTEGER now;
    ensure_qpc_init();
    QueryPerformanceCounter(&now);
    const int64_t g = ps3_guest_ticks(now.QuadPart);
    if (!s_have) {
        FILETIME ft;
        ULARGE_INTEGER u;
        GetSystemTimePreciseAsFileTime(&ft);
        u.LowPart = ft.dwLowDateTime; u.HighPart = ft.dwHighDateTime;
        while (InterlockedCompareExchange(&s_lock, 1, 0) != 0) YieldProcessor();
        if (!s_have) {
            s_base_100ns = u.QuadPart - 116444736000000000ULL;   /* 1601 -> 1970 */
            s_base_ticks = g;
            InterlockedExchange(&s_have, 1);
        }
        InterlockedExchange(&s_lock, 0);
    }
    const int64_t d = g - s_base_ticks;
    const uint64_t el = d > 0 ? (uint64_t)d : 0, q = (uint64_t)s_qpc_freq.QuadPart;
    const uint64_t t = s_base_100ns + (el / q) * 10000000ULL + ((el % q) * 10000000ULL) / q;
    *sec = t / 10000000ULL;
    *nsec = (t % 10000000ULL) * 100ULL;
}
#endif

/* LV2 144: r3/r4 point to signed BE32 minute offsets. The first is the
 * standard UTC offset; the second is the current seasonal adjustment.
 * Derive the adjustment instead of assuming every DST shift is 60 minutes. */
int64_t sys_time_get_timezone(ppu_context* ctx)
{
    uint32_t zone_addr = LV2_ARG_PTR(ctx, 0);
    uint32_t dst_addr = LV2_ARG_PTR(ctx, 1);
    if (!zone_addr || !dst_addr || zone_addr > UINT32_MAX - 3 || dst_addr > UINT32_MAX - 3)
        return (int32_t)CELL_EFAULT;

    int32_t standard_minutes, daylight_minutes;
#ifdef _WIN32
    /* Ask the system, not the C runtime: _get_timezone and _get_dstbias do not exist in the msvcrt.dll that a
     * MinGW-w64 toolchain with the msvcrt runtime links against. Bias is minutes west of UTC. */
    TIME_ZONE_INFORMATION tz;
    DWORD zone_id = GetTimeZoneInformation(&tz);
    if (zone_id == TIME_ZONE_ID_INVALID)
        return (int32_t)CELL_EAGAIN;
    standard_minutes = -(int32_t)(tz.Bias + tz.StandardBias);
    daylight_minutes = zone_id == TIME_ZONE_ID_DAYLIGHT ? -(int32_t)(tz.DaylightBias - tz.StandardBias) : 0;
#else
    time_t now = time(NULL);
    struct tm local;
    if (!localtime_r(&now, &local)) return (int32_t)CELL_EAGAIN;
    daylight_minutes = 0;
    if (local.tm_isdst > 0) {
        struct tm standard = local;
        standard.tm_isdst = 0;
        time_t standard_epoch = mktime(&standard);
        if (standard_epoch == (time_t)-1) return (int32_t)CELL_EAGAIN;
        daylight_minutes = (int32_t)(difftime(standard_epoch, now) / 60);
    }
    standard_minutes = (int32_t)(local.tm_gmtoff / 60) - daylight_minutes;
#endif
    write_be32(zone_addr, (uint32_t)standard_minutes);
    write_be32(dst_addr, (uint32_t)daylight_minutes);
    { static int said = 0;
      if (!said && (said = 1) && getenv("PS3_TIME_LOG"))
          fprintf(stderr, "[SYS] sys_time_get_timezone: %d minutes east of UTC, summer time %d minutes\n",
                  standard_minutes, daylight_minutes); }
    return CELL_OK;
}

int64_t sys_time_get_current_time(ppu_context* ctx)
{
    uint32_t sec_addr  = LV2_ARG_PTR(ctx, 0);
    uint32_t nsec_addr = LV2_ARG_PTR(ctx, 1);

    uint64_t sec, nsec;

#ifdef _WIN32
    wall_clock_now(&sec, &nsec);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    sec  = (uint64_t)ts.tv_sec;
    nsec = (uint64_t)ts.tv_nsec;
#endif

    if (sec_addr != 0) {
        write_be64(sec_addr, sec);
    }
    if (nsec_addr != 0) {
        write_be64(nsec_addr, nsec);
    }
    { static int said = 0;
      if (!said && (said = 1) && getenv("PS3_TIME_LOG")) {
          const time_t t = (time_t)sec;
          struct tm u;
#ifdef _WIN32
          gmtime_s(&u, &t);
#else
          gmtime_r(&t, &u);
#endif
          fprintf(stderr, "[SYS] sys_time_get_current_time: %llu s since 1970 (%04d-%02d-%02d %02d:%02d:%02d UTC)\n",
                  (unsigned long long)sec, u.tm_year + 1900, u.tm_mon + 1, u.tm_mday, u.tm_hour, u.tm_min, u.tm_sec);
      } }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_time_get_timebase_frequency
 *
 * Returns the PS3 timebase frequency in r3.
 * -----------------------------------------------------------------------*/
int64_t sys_time_get_timebase_frequency(ppu_context* ctx)
{
    (void)ctx;
    return (int64_t)PS3_TIMEBASE_FREQ;
}

/* ---------------------------------------------------------------------------
 * Periodic timer thread
 * -----------------------------------------------------------------------*/

/* Forward declaration */
static void timer_send_event(sys_timer_info* t);

/* Microseconds on a monotonic host clock: the timer's expiration times. */
static uint64_t timer_now_us(void)
{
#ifdef _WIN32
    static LARGE_INTEGER f;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER n; QueryPerformanceCounter(&n);
    return (uint64_t)(n.QuadPart / f.QuadPart) * 1000000ull +
           (uint64_t)(n.QuadPart % f.QuadPart) * 1000000ull / (uint64_t)f.QuadPart;
#else
    return host_monotonic_ns() / 1000ull;
#endif
}

#ifdef _WIN32
static DWORD WINAPI timer_thread_proc(LPVOID param)
{
    sys_timer_info* t = (sys_timer_info*)param;
    ps3_thread_name("lv2 timer");

    while (t->running) {
        DWORD ms = (DWORD)(t->period_usec / 1000);
        if (ms == 0) ms = 1;

        DWORD result = WaitForSingleObject(t->stop_event, ms);
        if (result == WAIT_OBJECT_0) {
            break; /* stop signaled */
        }

        if (t->running) {
            timer_send_event(t);
        }
    }

    return 0;
}
#else
static void* timer_thread_proc(void* param)
{
    sys_timer_info* t = (sys_timer_info*)param;

    pthread_mutex_lock(&t->mtx);
    while (!t->stop_flag) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec  += (time_t)(t->period_usec / 1000000);
        ts.tv_nsec += (long)((t->period_usec % 1000000) * 1000);
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }

        int rc = pthread_cond_timedwait(&t->cv, &t->mtx, &ts);
        if (t->stop_flag) break;

        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&t->mtx);
            timer_send_event(t);
            pthread_mutex_lock(&t->mtx);
        }
    }
    pthread_mutex_unlock(&t->mtx);

    return NULL;
}
#endif

static void timer_send_event(sys_timer_info* t)
{
    if (t->event_queue_id <= 0 || t->event_queue_id > SYS_EVENT_QUEUE_MAX)
        return;

    sys_event_queue_info* q = &g_sys_event_queues[t->event_queue_id - 1];
    if (!q->active)
        return;

    sys_event_t evt;
    evt.source = t->source;
    /* lv2 (and RPCS3) deliver the connect's data1/data2 and, in data3, the
     * EXPIRATION TIME -- which advances every period. It was 0 here, so a
     * consumer that paces itself by the delta between ticks saw no time pass:
     * Twisted Metal's BoomRangBuss audio driver wakes on this timer, derives
     * how many blocks to service from data3, and serviced none, ever -- so
     * brb_StartSession timed out and every later BRB request hung. */
    evt.data1  = t->data1;
    evt.data2  = t->data2;
    evt.data3  = t->next_expire;
    t->next_expire += t->period_usec;

    /* Push event (ignore if queue full) */
#ifdef _WIN32
    EnterCriticalSection(&q->lock);
#else
    pthread_mutex_lock(&q->lock);
#endif

    if (q->count < q->capacity) {
        q->buffer[q->tail] = evt;
        q->tail = (q->tail + 1) % q->capacity;
        q->count++;
#ifdef _WIN32
        WakeConditionVariable(&q->not_empty);
#else
        pthread_cond_signal(&q->not_empty);
#endif
    }

#ifdef _WIN32
    LeaveCriticalSection(&q->lock);
#else
    pthread_mutex_unlock(&q->lock);
#endif
}

/* ---------------------------------------------------------------------------
 * sys_timer_create
 *
 * r3 = pointer to receive timer ID (u32*)
 * -----------------------------------------------------------------------*/
int64_t sys_timer_create(ppu_context* ctx)
{
    uint32_t id_out_addr = LV2_ARG_PTR(ctx, 0);

    int slot = -1;
    for (int i = 0; i < SYS_TIMER_MAX; i++) {
        if (!g_sys_timers[i].active) { slot = i; break; }
    }
    if (slot < 0)
        return (int64_t)(int32_t)CELL_EAGAIN;

    sys_timer_info* t = &g_sys_timers[slot];
    memset(t, 0, sizeof(*t));
    t->active  = 1;
    t->running = 0;
    t->event_queue_id = 0;

#ifdef _WIN32
    t->stop_event    = NULL;
    t->thread_handle = NULL;
    t->timer_handle  = NULL;
#else
    pthread_mutex_init(&t->mtx, NULL);
    pthread_cond_init(&t->cv, NULL);
    t->stop_flag = 0;
#endif

    uint32_t timer_id = (uint32_t)(slot + 1);
    if (id_out_addr != 0) {
        write_be32(id_out_addr, timer_id);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_destroy
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_destroy(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    /* Stop if running */
    if (t->running) {
        t->running = 0;
#ifdef _WIN32
        if (t->stop_event) SetEvent(t->stop_event);
        if (t->thread_handle) {
            WaitForSingleObject(t->thread_handle, 3000);
            CloseHandle(t->thread_handle);
        }
        if (t->stop_event) CloseHandle(t->stop_event);
#else
        pthread_mutex_lock(&t->mtx);
        t->stop_flag = 1;
        pthread_cond_signal(&t->cv);
        pthread_mutex_unlock(&t->mtx);
        pthread_join(t->thread, NULL);
        pthread_mutex_destroy(&t->mtx);
        pthread_cond_destroy(&t->cv);
#endif
    }

    t->active = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_connect_event_queue
 *
 * r3 = timer_id
 * r4 = queue_id
 * r5 = source
 * r6 = data1
 * -----------------------------------------------------------------------*/
int64_t sys_timer_connect_event_queue(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);
    uint32_t queue_id = LV2_ARG_U32(ctx, 1);
    uint64_t source   = LV2_ARG_U64(ctx, 2);
    uint64_t data1    = LV2_ARG_U64(ctx, 3);
    uint64_t data2    = LV2_ARG_U64(ctx, 4);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (queue_id == 0 || queue_id > SYS_EVENT_QUEUE_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    t->event_queue_id = (int32_t)queue_id;
    t->source         = source;
    t->data1          = data1;
    t->data2          = data2;

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_disconnect_event_queue
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_disconnect_event_queue(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    t->event_queue_id = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_start
 *
 * r3 = timer_id
 * r4 = base_time (absolute start time, 0 = now)
 * r5 = period_usec
 * -----------------------------------------------------------------------*/
int64_t sys_timer_start(ppu_context* ctx)
{
    uint32_t timer_id    = LV2_ARG_U32(ctx, 0);
    /* uint64_t base_time = LV2_ARG_U64(ctx, 1); -- ignored for now */
    uint64_t period      = LV2_ARG_U64(ctx, 2);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (t->running)
        return (int64_t)(int32_t)CELL_EBUSY;

    if (period == 0)
        return (int64_t)(int32_t)CELL_EINVAL;

    t->period_usec = period;
    t->running     = 1;
    t->next_expire = timer_now_us() + period;

#ifdef _WIN32
    t->stop_event    = CreateEventA(NULL, TRUE, FALSE, NULL);
    t->thread_handle = CreateThread(NULL, 0, timer_thread_proc, t, 0, NULL);
#else
    t->stop_flag = 0;
    pthread_create(&t->thread, NULL, timer_thread_proc, t);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_stop
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_stop(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (!t->running)
        return CELL_OK;

    t->running = 0;

#ifdef _WIN32
    if (t->stop_event) SetEvent(t->stop_event);
    if (t->thread_handle) {
        WaitForSingleObject(t->thread_handle, 3000);
        CloseHandle(t->thread_handle);
        t->thread_handle = NULL;
    }
    if (t->stop_event) {
        CloseHandle(t->stop_event);
        t->stop_event = NULL;
    }
#else
    pthread_mutex_lock(&t->mtx);
    t->stop_flag = 1;
    pthread_cond_signal(&t->cv);
    pthread_mutex_unlock(&t->mtx);
    pthread_join(t->thread, NULL);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Registration
 * -----------------------------------------------------------------------*/
void sys_timer_init(lv2_syscall_table* tbl)
{
    memset(g_sys_timers, 0, sizeof(g_sys_timers));

#ifdef _WIN32
    ensure_qpc_init();
#endif

    lv2_syscall_register(tbl, SYS_TIMER_CREATE,                   sys_timer_create);
    lv2_syscall_register(tbl, SYS_TIMER_DESTROY,                  sys_timer_destroy);
    lv2_syscall_register(tbl, SYS_TIMER_START,                    sys_timer_start);
    lv2_syscall_register(tbl, SYS_TIMER_STOP,                     sys_timer_stop);
    lv2_syscall_register(tbl, SYS_TIMER_CONNECT_EVENT_QUEUE,      sys_timer_connect_event_queue);
    lv2_syscall_register(tbl, SYS_TIMER_DISCONNECT_EVENT_QUEUE,   sys_timer_disconnect_event_queue);

    lv2_syscall_register(tbl, SYS_TIME_GET_TIMEBASE_FREQUENCY, sys_time_get_timebase_frequency);
    lv2_syscall_register(tbl, SYS_TIMER_USLEEP,            sys_timer_usleep);
    lv2_syscall_register(tbl, SYS_TIMER_SLEEP,             sys_timer_sleep);
    lv2_syscall_register(tbl, SYS_TIME_GET_TIMEZONE,       sys_time_get_timezone);
    lv2_syscall_register(tbl, SYS_TIME_GET_CURRENT_TIME,   sys_time_get_current_time);
}
