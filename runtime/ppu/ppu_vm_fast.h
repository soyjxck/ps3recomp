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
 * maps the accessor names onto these: an ordinary access is one lookup in a
 * table of 64 KiB pages, one branch, and a byte-swapped load or store.
 * Anything the slow path exists for clears the page's bit, so it still takes
 * the function:
 *   - below 0x10000 (null page reports / null-store sweep)
 *   - out of the guest VM (vm_oob)
 *   - the raw SPU window at 0xE0000000 (register side effects)
 *   - the page of the GCM ref register (load side; see vm_fast_read32)
 *   - a page with a line an SPU holds a reservation on (store side; the
 *     function then tests the line itself)
 *   - any env-armed read or store diagnostic (g_ppu_vm_slow_*), or the
 *     run-time word watch g_ww_dyn: every page
 * The slow functions keep their full behaviour, so a run with PPU_WWATCH etc.
 * sees exactly what it always did.
 *
 * One branch, and not the four or five separate tests this used to be,
 * because the tests are repeated at every load and store in the lifted code
 * -- hundreds of thousands of sites -- and each conditional branch at each
 * site wants its own predictor entry. Measured on Drakengard 3 (x86-64, a
 * switch flipped every few seconds inside one run): taking the five-test
 * inline store path instead of calling the function cost the game and render
 * threads 9-11% MORE CPU per frame, although it executes fewer instructions.
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
/* One byte per 64 KiB guest page (defined in spu_coherency.c, which sets the
 * COH bit; ppu_vm_slow_any_update in ppu_loader.cpp keeps the other two and
 * has to run after anything they are computed from changes: ppu_vm_size,
 * ppu_hle_inject_base, g_ppu_vm_slow_reads/_stores, g_ww_dyn, the null-store
 * sweep). All zero, so every access is slow, until it has run once. */
extern unsigned char g_ppu_vm_page[65536];
#define PPU_VM_PAGE_RD   1u   /* loads take the inline path                    */
#define PPU_VM_PAGE_WR   2u   /* stores do too, unless ...                     */
#define PPU_VM_PAGE_COH  4u   /* ... an SPU has reserved a line in the page    */

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

static inline int ppu_vm_load_fast_ok(uint32_t a)
{
    return g_ppu_vm_page[a >> 16] & PPU_VM_PAGE_RD;
}

static inline int ppu_vm_store_fast_ok(uint32_t a)
{
    return (g_ppu_vm_page[a >> 16] & (PPU_VM_PAGE_WR | PPU_VM_PAGE_COH)) == PPU_VM_PAGE_WR;
}

static inline uint8_t vm_fast_read8(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_load_fast_ok(a), 1)) return *(volatile uint8_t*)(vm_base + a);
    return vm_read8_slow(ea);
}
static inline uint16_t vm_fast_read16(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_load_fast_ok(a), 1)) {
        uint16_t v = *(volatile uint16_t*)(vm_base + a); return __builtin_bswap16(v);
    }
    return vm_read16_slow(ea);
}
static inline uint32_t vm_fast_read32(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    /* A read of the GCM ref register is not a plain load: the slow path
     * publishes the next queued RSX fence on it (GCM_REFPOLL), and a title
     * spinning on it in cellGcmFinish waits forever without that. Its page
     * has no RD bit. */
    if (__builtin_expect(ppu_vm_load_fast_ok(a), 1)) {
        uint32_t v = *(volatile uint32_t*)(vm_base + a); return __builtin_bswap32(v);
    }
    return vm_read32_slow(ea);
}
static inline uint64_t vm_fast_read64(uint64_t ea)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_load_fast_ok(a), 1)) {
        uint64_t v = *(volatile uint64_t*)(vm_base + a); return __builtin_bswap64(v);
    }
    return vm_read64_slow(ea);
}
static inline void vm_fast_write8(uint64_t ea, uint8_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a), 1)) { *(volatile uint8_t*)(vm_base + a) = v; return; }
    vm_write8_slow(ea, v);
}
static inline void vm_fast_write16(uint64_t ea, uint16_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a), 1)) { *(volatile uint16_t*)(vm_base + a) = __builtin_bswap16(v); return; }
    vm_write16_slow(ea, v);
}
static inline void vm_fast_write32(uint64_t ea, uint32_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a), 1)) { *(volatile uint32_t*)(vm_base + a) = __builtin_bswap32(v); return; }
    vm_write32_slow(ea, v);
}
static inline void vm_fast_write64(uint64_t ea, uint64_t v)
{
    uint32_t a = (uint32_t)ea;
    if (__builtin_expect(ppu_vm_store_fast_ok(a), 1)) { *(volatile uint64_t*)(vm_base + a) = __builtin_bswap64(v); return; }
    vm_write64_slow(ea, v);
}

/* Store-conditionals with the caller's own context (see ppu_stwcx32_ctx). The
 * lifted code always has `ctx` in scope. */
struct ppu_context;
#ifdef __cplusplus
extern "C" {
#endif
int ppu_stwcx32_ctx(struct ppu_context* self, uint64_t ea, uint32_t expected, uint32_t val);
int ppu_stdcx64_ctx(struct ppu_context* self, uint64_t ea, uint64_t expected, uint64_t val);
#ifdef __cplusplus
}
#endif
#define ppu_stwcx32(ea, e, v) ppu_stwcx32_ctx(ctx, (ea), (e), (v))
#define ppu_stdcx64(ea, e, v) ppu_stdcx64_ctx(ctx, (ea), (e), (v))
#endif /* PS3RECOMP_PPU_VM_FAST_H */
