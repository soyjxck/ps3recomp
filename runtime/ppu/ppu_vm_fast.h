/*
 * ps3recomp - inline fast path for the lifted code's guest memory accessors
 *
 * Every lifted load and store calls vm_readN / vm_writeN in ppu_loader.cpp: an
 * out-of-line function that tests the bounds, the read/write watches, the
 * null page, the raw-SPU problem-state window, the SPU reservation set and
 * a handful of env-armed diagnostics before it touches memory. Drakengard 3's
 * render thread spent ~60% of its time inside those functions.
 *
 * With PPU_INLINE_VM defined for the lifted translation units, ppu_recomp.h
 * maps the accessor names onto these: an ordinary in-range access is one
 * bounds test and a byte-swapped load or store, and anything the slow path
 * exists for still takes it:
 *   - below 0x10000 (null page reports / null-store sweep)
 *   - out of the guest VM (vm_oob)
 *   - the raw SPU window at 0xE0000000 (register side effects)
 *   - a line an SPU holds a reservation on (store side)
 *   - any env-armed read or store diagnostic (g_ppu_vm_slow_*), or the
 *     run-time word watch g_ww_dyn
 * The slow functions keep their full behaviour, so a run with PPU_WWATCH etc.
 * sees exactly what it always did.
 *
 * The accesses are volatile: guest memory changes under other PPU threads
 * and the SPUs, and a lifted spin on a flag has no call between two loads
 * for the compiler to respect -- the out-of-line calls used to provide that
 * by accident. (Unaligned guest addresses are fine on x86-64 and arm64.)
 */
#ifndef PS3RECOMP_PPU_VM_FAST_H
#define PS3RECOMP_PPU_VM_FAST_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
extern uint8_t*      vm_base;
extern uint32_t      ppu_vm_size;          /* 0 = unchecked, as vm_oob treats it */
extern int           g_spu_coh_armed;      /* an SPU has reserved a line        */
extern unsigned char g_spu_coh_bitmap[];   /* one bit per 128-byte line         */
extern uint32_t      g_ww_dyn;             /* word watch armed at run time      */
extern uint32_t      g_null_sweep_hi;      /* null-store sweep in progress      */
extern int           g_ppu_vm_slow_reads;  /* a read diagnostic is armed        */
extern int           g_ppu_vm_slow_stores; /* a store diagnostic is armed       */

uint8_t  vm_read8_slow (uint64_t addr);
uint16_t vm_read16_slow(uint64_t addr);
uint32_t vm_read32_slow(uint64_t addr);
uint64_t vm_read64_slow(uint64_t addr);
void     vm_write8_slow (uint64_t addr, uint8_t  val);
void     vm_write16_slow(uint64_t addr, uint16_t val);
void     vm_write32_slow(uint64_t addr, uint32_t val);
void     vm_write64_slow(uint64_t addr, uint64_t val);
#ifdef __cplusplus
}
#endif

static inline int ppu_vm_fast_ok(uint32_t a, uint32_t n)
{
    if (a < 0x10000u || (a >> 24) == 0xE0u) return 0;
    return ppu_vm_size == 0 || (uint64_t)a + n <= ppu_vm_size;
}

static inline int ppu_vm_store_fast_ok(uint32_t a, uint32_t n)
{
    if (g_ppu_vm_slow_stores || g_ww_dyn || g_null_sweep_hi || !ppu_vm_fast_ok(a, n)) return 0;
    if (g_spu_coh_armed) {
        uint32_t line = a >> 7;
        if ((g_spu_coh_bitmap[line >> 3] >> (line & 7)) & 1u) return 0;
    }
    return 1;
}

static inline uint8_t vm_fast_read8(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(!g_ppu_vm_slow_reads && ppu_vm_fast_ok(a, 1), 1)) return *(volatile uint8_t*)(vm_base + a);
    return vm_read8_slow(ea);
}
static inline uint16_t vm_fast_read16(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(!g_ppu_vm_slow_reads && ppu_vm_fast_ok(a, 2), 1)) {
        uint16_t v = *(volatile uint16_t*)(vm_base + a); return __builtin_bswap16(v);
    }
    return vm_read16_slow(ea);
}
static inline uint32_t vm_fast_read32(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(!g_ppu_vm_slow_reads && ppu_vm_fast_ok(a, 4), 1)) {
        uint32_t v = *(volatile uint32_t*)(vm_base + a); return __builtin_bswap32(v);
    }
    return vm_read32_slow(ea);
}
static inline uint64_t vm_fast_read64(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(!g_ppu_vm_slow_reads && ppu_vm_fast_ok(a, 8), 1)) {
        uint64_t v = *(volatile uint64_t*)(vm_base + a); return __builtin_bswap64(v);
    }
    return vm_read64_slow(ea);
}
static inline void vm_fast_write8(uint64_t ea, uint8_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a, 1), 1)) { *(volatile uint8_t*)(vm_base + a) = v; return; }
    vm_write8_slow(ea, v);
}
static inline void vm_fast_write16(uint64_t ea, uint16_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a, 2), 1)) { *(volatile uint16_t*)(vm_base + a) = __builtin_bswap16(v); return; }
    vm_write16_slow(ea, v);
}
static inline void vm_fast_write32(uint64_t ea, uint32_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a, 4), 1)) { *(volatile uint32_t*)(vm_base + a) = __builtin_bswap32(v); return; }
    vm_write32_slow(ea, v);
}
static inline void vm_fast_write64(uint64_t ea, uint64_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a, 8), 1)) { *(volatile uint64_t*)(vm_base + a) = __builtin_bswap64(v); return; }
    vm_write64_slow(ea, v);
}

#endif /* PS3RECOMP_PPU_VM_FAST_H */
