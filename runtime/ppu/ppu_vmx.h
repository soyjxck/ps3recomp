/* ppu_vmx.h -- exact VMX (AltiVec) semantics for lifted PPU code.
 *
 * Vector registers are held big-endian (byte 0 = the most significant byte,
 * element 0 = the leftmost element), exactly as lvx loads them. Every helper
 * computes into a temporary first, so the destination may alias any source.
 *
 * Semantics follow PowerISA 2.02 VMX as RPCS3's PPU interpreter implements it
 * with its accuracy options on (the oracle of tests/conformance/ppu):
 *   - saturating ops OR the SAT bit (VSCR 0x00000001) into *vscr;
 *   - VSCR[NJ] (0x00010000) flushes denormal float inputs and results to zero;
 *   - a NaN operand propagates (A, then B, then C) with its quiet bit set; a NaN
 *     produced from non-NaN operands is the default QNaN 0x7FC00000.
 * Each helper is the whole instruction, so the lifter emits one call per op.
 */
#ifndef PPU_VMX_H
#define PPU_VMX_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VMX_SAT 0x00000001u
#define VMX_NJ  0x00010000u
#define VMX_INL static inline

/* On x86-64 with AVX2 and FMA (the x86-64-v3 the lifted code is built for)
 * and on arm64 (NEON) the float arithmetic, the float compares and the word
 * shifts have vector forms at the end of this file, with the same results bit
 * for bit: the same IEEE operations, the same NaN, VSCR[NJ] and saturation
 * rules. The scalar forms stay, as name_ref, for tests/test_ppu_vmx_vec.c to
 * hold the vector ones to. PPU_VMX_SCALAR builds the scalar forms alone. */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) && defined(__FMA__) && \
    !defined(PPU_VMX_SCALAR)
#define VMX_X86 1
#else
#define VMX_X86 0
#endif
#if defined(__aarch64__) && !defined(PPU_VMX_SCALAR)
#define VMX_NEON 1
#else
#define VMX_NEON 0
#endif
#if VMX_X86 || VMX_NEON
#define VMX_REF(name) name##_ref
#else
#define VMX_REF(name) name
#endif

/* ---- element access (big-endian) ---- */
VMX_INL uint16_t vmx_h(const uint8_t* v, int i) { return (uint16_t)(v[2 * i] << 8 | v[2 * i + 1]); }
VMX_INL uint32_t vmx_w(const uint8_t* v, int i) {
    return (uint32_t)v[4 * i] << 24 | (uint32_t)v[4 * i + 1] << 16 | (uint32_t)v[4 * i + 2] << 8 | v[4 * i + 3];
}
VMX_INL void vmx_sh(uint8_t* v, int i, uint16_t x) { v[2 * i] = (uint8_t)(x >> 8); v[2 * i + 1] = (uint8_t)x; }
VMX_INL void vmx_sw(uint8_t* v, int i, uint32_t x) {
    v[4 * i] = (uint8_t)(x >> 24); v[4 * i + 1] = (uint8_t)(x >> 16); v[4 * i + 2] = (uint8_t)(x >> 8); v[4 * i + 3] = (uint8_t)x;
}
VMX_INL float    vmx_f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
VMX_INL uint32_t vmx_u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

#define VMX_D  void* d_
#define VMX_A  const void* a_
#define VMX_B  const void* b_
#define VMX_C  const void* c_
#define VMX_ARGS_PREP const uint8_t* a = (const uint8_t*)a_; (void)a; \
    const uint8_t* b = (const uint8_t*)b_; (void)b; uint8_t r[16]
#define VMX_OUT memcpy(d_, r, 16)

/* ---- saturation ---- */
VMX_INL int32_t vmx_sat_s(int64_t x, int64_t lo, int64_t hi, uint32_t* vscr) {
    if (x < lo) { *vscr |= VMX_SAT; return (int32_t)lo; }
    if (x > hi) { *vscr |= VMX_SAT; return (int32_t)hi; }
    return (int32_t)x;
}
VMX_INL uint32_t vmx_sat_u(int64_t x, int64_t hi, uint32_t* vscr) {
    if (x < 0) { *vscr |= VMX_SAT; return 0; }
    if (x > hi) { *vscr |= VMX_SAT; return (uint32_t)hi; }
    return (uint32_t)x;
}

/* ---- float helpers ---- */
VMX_INL uint32_t vmx_flush(uint32_t x, uint32_t vscr) {
    return ((vscr & VMX_NJ) && (x & 0x7F800000u) == 0) ? (x & 0x80000000u) : x;
}
VMX_INL int vmx_isnan(uint32_t x) { return (x & 0x7FFFFFFFu) > 0x7F800000u; }
VMX_INL uint32_t vmx_fix(uint32_t r) { return vmx_isnan(r) ? 0x7FC00000u : r; }
/* NaN precedence: A, B (quieted), else the fixed result */
VMX_INL uint32_t vmx_vnan2(uint32_t r, uint32_t a, uint32_t b) {
    if (vmx_isnan(a)) return a | 0x7FC00000u;
    if (vmx_isnan(b)) return b | 0x7FC00000u;
    return vmx_fix(r);
}
VMX_INL uint32_t vmx_vnan1(uint32_t r, uint32_t b) {
    if (vmx_isnan(b)) return b | 0x7FC00000u;
    return vmx_fix(r);
}

/* ======================================================================
 * integer element-wise
 * ==================================================================== */

#define VMX_BIN(name, cnt, get, set, expr) \
VMX_INL void name(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP; \
    for (int i = 0; i < cnt; i++) { uint64_t x = get(a, i), y = get(b, i); (void)x; (void)y; set(r, i, expr); } VMX_OUT; }
VMX_INL uint8_t vmx_b(const uint8_t* v, int i) { return v[i]; }
VMX_INL void vmx_sb(uint8_t* v, int i, uint8_t x) { v[i] = x; }

/* rotates / shifts: count from the low bits of the matching element of B */
VMX_BIN(vmx_vrlb, 16, vmx_b, vmx_sb, (uint8_t)((x << (y & 7)) | (x >> ((8 - (y & 7)) & 7))))
VMX_BIN(vmx_vrlh, 8, vmx_h, vmx_sh, (uint16_t)((x << (y & 15)) | (x >> ((16 - (y & 15)) & 15))))
VMX_BIN(vmx_vrlw, 4, vmx_w, vmx_sw, (uint32_t)((x << (y & 31)) | (x >> ((32 - (y & 31)) & 31))))
VMX_BIN(vmx_vslb, 16, vmx_b, vmx_sb, (uint8_t)(x << (y & 7)))
VMX_BIN(vmx_vslh, 8, vmx_h, vmx_sh, (uint16_t)(x << (y & 15)))
VMX_BIN(VMX_REF(vmx_vslw), 4, vmx_w, vmx_sw, (uint32_t)(x << (y & 31)))
VMX_BIN(vmx_vsrb, 16, vmx_b, vmx_sb, (uint8_t)(x >> (y & 7)))
VMX_BIN(vmx_vsrh, 8, vmx_h, vmx_sh, (uint16_t)(x >> (y & 15)))
VMX_BIN(VMX_REF(vmx_vsrw), 4, vmx_w, vmx_sw, (uint32_t)(x >> (y & 31)))
VMX_BIN(vmx_vsrab, 16, vmx_b, vmx_sb, (uint8_t)((int8_t)x >> (y & 7)))
VMX_BIN(vmx_vsrah, 8, vmx_h, vmx_sh, (uint16_t)((int16_t)x >> (y & 15)))
VMX_BIN(VMX_REF(vmx_vsraw), 4, vmx_w, vmx_sw, (uint32_t)((int32_t)x >> (y & 31)))

/* averages: (a + b + 1) >> 1 without overflow */
VMX_BIN(vmx_vavgub, 16, vmx_b, vmx_sb, (uint8_t)((x + y + 1) >> 1))
VMX_BIN(vmx_vavguh, 8, vmx_h, vmx_sh, (uint16_t)((x + y + 1) >> 1))
VMX_BIN(vmx_vavguw, 4, vmx_w, vmx_sw, (uint32_t)((x + y + 1) >> 1))
VMX_BIN(vmx_vavgsb, 16, vmx_b, vmx_sb, (uint8_t)(((int64_t)(int8_t)x + (int8_t)y + 1) >> 1))
VMX_BIN(vmx_vavgsh, 8, vmx_h, vmx_sh, (uint16_t)(((int64_t)(int16_t)x + (int16_t)y + 1) >> 1))
VMX_BIN(vmx_vavgsw, 4, vmx_w, vmx_sw, (uint32_t)(((int64_t)(int32_t)x + (int32_t)y + 1) >> 1))

/* carry / borrow out */
VMX_BIN(vmx_vaddcuw, 4, vmx_w, vmx_sw, (uint32_t)((x + y) >> 32))
VMX_BIN(vmx_vsubcuw, 4, vmx_w, vmx_sw, (uint32_t)(x >= y))

/* saturating add / sub */
#define VMX_SATOP(name, cnt, get, set, expr) \
VMX_INL void name(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP; \
    for (int i = 0; i < cnt; i++) { uint64_t x = get(a, i), y = get(b, i); set(r, i, expr); } VMX_OUT; }
VMX_SATOP(vmx_vaddubs, 16, vmx_b, vmx_sb, (uint8_t)vmx_sat_u((int64_t)x + (int64_t)y, 0xFF, vscr))
VMX_SATOP(vmx_vadduhs, 8, vmx_h, vmx_sh, (uint16_t)vmx_sat_u((int64_t)x + (int64_t)y, 0xFFFF, vscr))
VMX_SATOP(vmx_vadduws, 4, vmx_w, vmx_sw, (uint32_t)vmx_sat_u((int64_t)x + (int64_t)y, 0xFFFFFFFFll, vscr))
VMX_SATOP(vmx_vaddsbs, 16, vmx_b, vmx_sb, (uint8_t)vmx_sat_s((int64_t)(int8_t)x + (int8_t)y, -128, 127, vscr))
VMX_SATOP(vmx_vaddshs, 8, vmx_h, vmx_sh, (uint16_t)vmx_sat_s((int64_t)(int16_t)x + (int16_t)y, -32768, 32767, vscr))
VMX_SATOP(vmx_vaddsws, 4, vmx_w, vmx_sw, (uint32_t)vmx_sat_s((int64_t)(int32_t)x + (int32_t)y, INT32_MIN, INT32_MAX, vscr))
VMX_SATOP(vmx_vsububs, 16, vmx_b, vmx_sb, (uint8_t)vmx_sat_u((int64_t)x - (int64_t)y, 0xFF, vscr))
VMX_SATOP(vmx_vsubuhs, 8, vmx_h, vmx_sh, (uint16_t)vmx_sat_u((int64_t)x - (int64_t)y, 0xFFFF, vscr))
VMX_SATOP(vmx_vsubuws, 4, vmx_w, vmx_sw, (uint32_t)vmx_sat_u((int64_t)x - (int64_t)y, 0xFFFFFFFFll, vscr))
VMX_SATOP(vmx_vsubsbs, 16, vmx_b, vmx_sb, (uint8_t)vmx_sat_s((int64_t)(int8_t)x - (int8_t)y, -128, 127, vscr))
VMX_SATOP(vmx_vsubshs, 8, vmx_h, vmx_sh, (uint16_t)vmx_sat_s((int64_t)(int16_t)x - (int16_t)y, -32768, 32767, vscr))
VMX_SATOP(vmx_vsubsws, 4, vmx_w, vmx_sw, (uint32_t)vmx_sat_s((int64_t)(int32_t)x - (int32_t)y, INT32_MIN, INT32_MAX, vscr))

/* even / odd multiplies: element i of the result is the product of source
 * element 2i (even) or 2i+1 (odd), big-endian numbering */
#define VMX_MUL(name, n, get, set, cast, par) \
VMX_INL void name(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP; \
    for (int i = 0; i < n; i++) { int64_t x = (cast)get(a, 2 * i + par), y = (cast)get(b, 2 * i + par); set(r, i, x * y); } VMX_OUT; }
VMX_MUL(vmx_vmuleub, 8, vmx_b, vmx_sh, uint8_t, 0)
VMX_MUL(vmx_vmuloub, 8, vmx_b, vmx_sh, uint8_t, 1)
VMX_MUL(vmx_vmulesb, 8, vmx_b, vmx_sh, int8_t, 0)
VMX_MUL(vmx_vmulosb, 8, vmx_b, vmx_sh, int8_t, 1)
VMX_MUL(vmx_vmuleuh, 4, vmx_h, vmx_sw, uint16_t, 0)
VMX_MUL(vmx_vmulouh, 4, vmx_h, vmx_sw, uint16_t, 1)
VMX_MUL(vmx_vmulesh, 4, vmx_h, vmx_sw, int16_t, 0)
VMX_MUL(vmx_vmulosh, 4, vmx_h, vmx_sw, int16_t, 1)

/* ---- multiply-add / multiply-sum (VA-form) ---- */
VMX_INL void vmx_vmhaddshs(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t* vscr) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 8; i++) {
        int64_t p = ((int64_t)(int16_t)vmx_h(a, i) * (int16_t)vmx_h(b, i)) >> 15;
        vmx_sh(r, i, (uint16_t)vmx_sat_s(p + (int16_t)vmx_h(c, i), -32768, 32767, vscr));
    } VMX_OUT; }
VMX_INL void vmx_vmhraddshs(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t* vscr) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 8; i++) {
        int64_t p = ((int64_t)(int16_t)vmx_h(a, i) * (int16_t)vmx_h(b, i) + 0x4000) >> 15;
        vmx_sh(r, i, (uint16_t)vmx_sat_s(p + (int16_t)vmx_h(c, i), -32768, 32767, vscr));
    } VMX_OUT; }
VMX_INL void vmx_vmladduhm(VMX_D, VMX_A, VMX_B, VMX_C) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 8; i++) vmx_sh(r, i, (uint16_t)(vmx_h(a, i) * vmx_h(b, i) + vmx_h(c, i))); VMX_OUT; }
VMX_INL void vmx_vmsumubm(VMX_D, VMX_A, VMX_B, VMX_C) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) { uint32_t s = vmx_w(c, i);
        for (int j = 0; j < 4; j++) s += (uint32_t)a[4 * i + j] * b[4 * i + j];
        vmx_sw(r, i, s); } VMX_OUT; }
VMX_INL void vmx_vmsummbm(VMX_D, VMX_A, VMX_B, VMX_C) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) { uint32_t s = vmx_w(c, i);
        for (int j = 0; j < 4; j++) s += (uint32_t)((int32_t)(int8_t)a[4 * i + j] * (int32_t)b[4 * i + j]);
        vmx_sw(r, i, s); } VMX_OUT; }
VMX_INL void vmx_vmsumuhm(VMX_D, VMX_A, VMX_B, VMX_C) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) vmx_sw(r, i, vmx_w(c, i) + (uint32_t)vmx_h(a, 2 * i) * vmx_h(b, 2 * i)
                                                        + (uint32_t)vmx_h(a, 2 * i + 1) * vmx_h(b, 2 * i + 1));
    VMX_OUT; }
VMX_INL void vmx_vmsumuhs(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t* vscr) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) {
        int64_t s = (int64_t)vmx_w(c, i) + (int64_t)vmx_h(a, 2 * i) * vmx_h(b, 2 * i)
                  + (int64_t)vmx_h(a, 2 * i + 1) * vmx_h(b, 2 * i + 1);
        vmx_sw(r, i, vmx_sat_u(s, 0xFFFFFFFFll, vscr)); } VMX_OUT; }
VMX_INL void vmx_vmsumshm(VMX_D, VMX_A, VMX_B, VMX_C) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) vmx_sw(r, i, (uint32_t)((int32_t)vmx_w(c, i)
        + (int32_t)(int16_t)vmx_h(a, 2 * i) * (int16_t)vmx_h(b, 2 * i)
        + (int32_t)(int16_t)vmx_h(a, 2 * i + 1) * (int16_t)vmx_h(b, 2 * i + 1)));
    VMX_OUT; }
VMX_INL void vmx_vmsumshs(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t* vscr) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) {
        int64_t s = (int64_t)(int32_t)vmx_w(c, i) + (int64_t)(int16_t)vmx_h(a, 2 * i) * (int16_t)vmx_h(b, 2 * i)
                  + (int64_t)(int16_t)vmx_h(a, 2 * i + 1) * (int16_t)vmx_h(b, 2 * i + 1);
        vmx_sw(r, i, (uint32_t)vmx_sat_s(s, INT32_MIN, INT32_MAX, vscr)); } VMX_OUT; }

/* ---- sums across ---- */
VMX_INL void vmx_vsum4ubs(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP;
    for (int i = 0; i < 4; i++) { int64_t s = vmx_w(b, i);
        for (int j = 0; j < 4; j++) s += a[4 * i + j];
        vmx_sw(r, i, vmx_sat_u(s, 0xFFFFFFFFll, vscr)); } VMX_OUT; }
VMX_INL void vmx_vsum4sbs(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP;
    for (int i = 0; i < 4; i++) { int64_t s = (int32_t)vmx_w(b, i);
        for (int j = 0; j < 4; j++) s += (int8_t)a[4 * i + j];
        vmx_sw(r, i, (uint32_t)vmx_sat_s(s, INT32_MIN, INT32_MAX, vscr)); } VMX_OUT; }
VMX_INL void vmx_vsum4shs(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP;
    for (int i = 0; i < 4; i++) {
        int64_t s = (int64_t)(int32_t)vmx_w(b, i) + (int16_t)vmx_h(a, 2 * i) + (int16_t)vmx_h(a, 2 * i + 1);
        vmx_sw(r, i, (uint32_t)vmx_sat_s(s, INT32_MIN, INT32_MAX, vscr)); } VMX_OUT; }
VMX_INL void vmx_vsum2sws(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP;
    memset(r, 0, 16);
    for (int i = 0; i < 2; i++) {
        int64_t s = (int64_t)(int32_t)vmx_w(a, 2 * i) + (int32_t)vmx_w(a, 2 * i + 1) + (int32_t)vmx_w(b, 2 * i + 1);
        vmx_sw(r, 2 * i + 1, (uint32_t)vmx_sat_s(s, INT32_MIN, INT32_MAX, vscr)); } VMX_OUT; }
VMX_INL void vmx_vsumsws(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP;
    memset(r, 0, 16);
    int64_t s = (int32_t)vmx_w(b, 3);
    for (int i = 0; i < 4; i++) s += (int32_t)vmx_w(a, i);
    vmx_sw(r, 3, (uint32_t)vmx_sat_s(s, INT32_MIN, INT32_MAX, vscr)); VMX_OUT; }

/* ---- packs: A's elements fill the high half, B's the low half ---- */
#define VMX_PACK(name, n, get, set, conv) \
VMX_INL void name(VMX_D, VMX_A, VMX_B, uint32_t* vscr) { VMX_ARGS_PREP; (void)vscr; \
    for (int i = 0; i < n; i++) { int64_t x = get(a, i); set(r, i, conv); } \
    for (int i = 0; i < n; i++) { int64_t x = get(b, i); set(r, n + i, conv); } VMX_OUT; }
VMX_PACK(vmx_vpkuhum, 8, vmx_h, vmx_sb, (uint8_t)x)
VMX_PACK(vmx_vpkuwum, 4, vmx_w, vmx_sh, (uint16_t)x)
VMX_PACK(vmx_vpkuhus, 8, vmx_h, vmx_sb, (uint8_t)vmx_sat_u(x, 0xFF, vscr))
VMX_PACK(vmx_vpkuwus, 4, vmx_w, vmx_sh, (uint16_t)vmx_sat_u(x, 0xFFFF, vscr))
VMX_PACK(vmx_vpkshus, 8, vmx_h, vmx_sb, (uint8_t)vmx_sat_u((int16_t)x, 0xFF, vscr))
VMX_PACK(vmx_vpkswus, 4, vmx_w, vmx_sh, (uint16_t)vmx_sat_u((int32_t)x, 0xFFFF, vscr))
VMX_PACK(vmx_vpkshss, 8, vmx_h, vmx_sb, (uint8_t)vmx_sat_s((int16_t)x, -128, 127, vscr))
VMX_PACK(vmx_vpkswss, 4, vmx_w, vmx_sh, (uint16_t)vmx_sat_s((int32_t)x, -32768, 32767, vscr))
VMX_PACK(vmx_vpkpx, 4, vmx_w, vmx_sh, (uint16_t)(((x >> 24) & 1) << 15 | ((x >> 19) & 0x1F) << 10 |
                                                 ((x >> 11) & 0x1F) << 5 | ((x >> 3) & 0x1F)))

/* ---- unpacks (sign-extend; pixel 1:5:5:5 -> 8:8:8:8) ---- */
VMX_INL void vmx_vupkpx(VMX_D, VMX_B, int lo) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    for (int i = 0; i < 4; i++) { uint16_t h = vmx_h(b, i + 4 * lo);
        r[4 * i] = (h & 0x8000) ? 0xFF : 0; r[4 * i + 1] = (h >> 10) & 0x1F;
        r[4 * i + 2] = (h >> 5) & 0x1F; r[4 * i + 3] = h & 0x1F; } VMX_OUT; }
VMX_INL void vmx_vupksb(VMX_D, VMX_B, int lo) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    for (int i = 0; i < 8; i++) vmx_sh(r, i, (uint16_t)(int16_t)(int8_t)b[i + 8 * lo]); VMX_OUT; }
VMX_INL void vmx_vupksh(VMX_D, VMX_B, int lo) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    for (int i = 0; i < 4; i++) vmx_sw(r, i, (uint32_t)(int32_t)(int16_t)vmx_h(b, i + 4 * lo)); VMX_OUT; }

/* ---- whole-vector shifts ---- */
/* vsl / vsr: RPCS3 shifts each byte by that byte's own count (B byte & 7),
 * funnelling bits in from the neighbour; with all counts equal (the ISA's
 * defined case) this is the 128-bit shift. */
VMX_INL void vmx_vsl(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP;
    for (int i = 0; i < 16; i++) { int s = b[i] & 7; uint8_t nx = i < 15 ? a[i + 1] : 0;
        r[i] = (uint8_t)((a[i] << s) | (s ? nx >> (8 - s) : 0)); } VMX_OUT; }
VMX_INL void vmx_vsr(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP;
    for (int i = 0; i < 16; i++) { int s = b[i] & 7; uint8_t pv = i > 0 ? a[i - 1] : 0;
        r[i] = (uint8_t)((a[i] >> s) | (s ? pv << (8 - s) : 0)); } VMX_OUT; }
/* vslo / vsro: by octets, count from bits 1:4 of B's last byte */
VMX_INL void vmx_vslo(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP; int n = (b[15] >> 3) & 15;
    for (int i = 0; i < 16; i++) r[i] = i + n < 16 ? a[i + n] : 0; VMX_OUT; }
VMX_INL void vmx_vsro(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP; int n = (b[15] >> 3) & 15;
    for (int i = 0; i < 16; i++) r[i] = i - n >= 0 ? a[i - n] : 0; VMX_OUT; }

/* ======================================================================
 * floating point
 * ==================================================================== */

#define VMX_FBIN(name, op, flush_in) \
VMX_INL void name(VMX_D, VMX_A, VMX_B, uint32_t vscr) { VMX_ARGS_PREP; \
    for (int i = 0; i < 4; i++) { uint32_t x = vmx_w(a, i), y = vmx_w(b, i); \
        if (flush_in) { x = vmx_flush(x, vscr); y = vmx_flush(y, vscr); } \
        uint32_t z = vmx_u(op); vmx_sw(r, i, vmx_flush(vmx_vnan2(z, x, y), vscr)); } VMX_OUT; }

VMX_INL float vmx_maxf(float x, float y) {
    if (x != x || y != y) return x + y;
    if (x == 0 && y == 0) return signbit(x) ? y : x;      /* max(-0,+0) = +0 */
    return x > y ? x : y;
}
VMX_INL float vmx_minf(float x, float y) {
    if (x != x || y != y) return x + y;
    if (x == 0 && y == 0) return signbit(x) ? x : y;      /* min(-0,+0) = -0 */
    return x < y ? x : y;
}
VMX_FBIN(VMX_REF(vmx_vaddfp), vmx_f(x) + vmx_f(y), 1)
VMX_FBIN(VMX_REF(vmx_vsubfp), vmx_f(x) - vmx_f(y), 1)
VMX_FBIN(VMX_REF(vmx_vmaxfp), vmx_maxf(vmx_f(x), vmx_f(y)), 0)
VMX_FBIN(VMX_REF(vmx_vminfp), vmx_minf(vmx_f(x), vmx_f(y)), 0)

/* Fused multiply-add without a libm call per lane. Without -mfma, fmaf is a
 * software routine on MinGW, and vmaddfp/vnmsubfp called it once per lane. The
 * product of two floats is exact in double, so s = x*z + y has one double
 * rounding; rounding s to float then equals the correctly rounded fused result
 * unless s sits exactly on a float rounding midpoint (low 29 bits 0x10000000)
 * or outside the normal float range (zero, subnormal, overflow, inf, NaN).
 * Those rare cases still call fmaf, so every lane is bit-identical to fmaf.
 * ASSUMPTION: the host rounding mode is round-to-nearest (nothing in the
 * runtime changes MXCSR). */
VMX_INL float vmx_fmaf(float x, float z, float y) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
    return fmaf(x, z, y);   /* one instruction with -mfma / arm64 fmadd */
#else
    const double s = (double)x * (double)z + (double)y;
    uint64_t u; memcpy(&u, &s, sizeof u);
    const uint32_t e = (uint32_t)(u >> 52) & 0x7FFu;
    if ((u & 0x1FFFFFFFull) != 0x10000000ull && e - (1023u - 126u) < 254u)
        return (float)s;
    return fmaf(x, z, y);
#endif
}
VMX_INL void VMX_REF(vmx_vmaddfp)(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t vscr) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) {
        float x = vmx_f(vmx_flush(vmx_w(a, i), vscr)), y = vmx_f(vmx_flush(vmx_w(b, i), vscr)),
              z = vmx_f(vmx_flush(vmx_w(c, i), vscr));
        vmx_sw(r, i, vmx_flush(vmx_fix(vmx_u(vmx_fmaf(x, z, y))), vscr)); } VMX_OUT; }
VMX_INL void VMX_REF(vmx_vnmsubfp)(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t vscr) { VMX_ARGS_PREP; const uint8_t* c = (const uint8_t*)c_;
    for (int i = 0; i < 4; i++) {
        float x = vmx_f(vmx_flush(vmx_w(a, i), vscr)), y = vmx_f(vmx_flush(vmx_w(b, i), vscr)),
              z = vmx_f(vmx_flush(vmx_w(c, i), vscr));
        /* -(x*z - y), the difference rounded once and then negated: the sign
         * flipped as bits, since GCC folds -fmaf(x, z, -y) into a negated
         * multiply-add, which gives +0 where this is -0. */
        vmx_sw(r, i, vmx_flush(vmx_fix(vmx_u(vmx_fmaf(x, z, -y)) ^ 0x80000000u), vscr)); } VMX_OUT; }

#define VMX_FUN(name, expr, flush_in, nanmode) \
VMX_INL void name(VMX_D, VMX_B, uint32_t vscr) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16]; \
    for (int i = 0; i < 4; i++) { uint32_t y = vmx_w(b, i); if (flush_in) y = vmx_flush(y, vscr); \
        float f = vmx_f(y); uint32_t z = vmx_u(expr); \
        vmx_sw(r, i, nanmode == 2 ? vmx_flush(vmx_vnan1(z, y), vscr) : vmx_fix(z)); } VMX_OUT; }
VMX_FUN(VMX_REF(vmx_vrefp), 1.0f / f, 1, 2)
VMX_FUN(VMX_REF(vmx_vrsqrtefp), 1.0f / sqrtf(f), 1, 2)
VMX_FUN(vmx_vexptefp, exp2f(f), 0, 1)
VMX_FUN(vmx_vlogefp, log2f(f), 0, 1)
VMX_FUN(vmx_vrfin, nearbyintf(f), 0, 2)      /* host default rounding: to nearest even */
VMX_FUN(vmx_vrfiz, truncf(f), 0, 2)
VMX_FUN(vmx_vrfip, ceilf(f), 1, 2)
VMX_FUN(vmx_vrfim, floorf(f), 1, 2)

/* int <-> float with a 2^uimm scale. uimm is 5 bits, so 2^-uimm and 2^uimm are
 * normal floats and the scaling is exact: the same results as ldexpf, without
 * a libm call per lane (which also kept these helpers out of line). */
VMX_INL float vmx_pow2(int e) { return vmx_f((uint32_t)(127 + e) << 23); }
VMX_INL void vmx_vcfux(VMX_D, VMX_B, unsigned uimm) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    const float s = vmx_pow2(-(int)(uimm & 31u));
    for (int i = 0; i < 4; i++) vmx_sw(r, i, vmx_u((float)vmx_w(b, i) * s)); VMX_OUT; }
VMX_INL void VMX_REF(vmx_vcfsx)(VMX_D, VMX_B, unsigned uimm) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    const float s = vmx_pow2(-(int)(uimm & 31u));
    for (int i = 0; i < 4; i++) vmx_sw(r, i, vmx_u((float)(int32_t)vmx_w(b, i) * s)); VMX_OUT; }
VMX_INL void vmx_vctsxs(VMX_D, VMX_B, unsigned uimm, uint32_t* vscr) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    for (int i = 0; i < 4; i++) { float f = vmx_f(vmx_w(b, i)) * vmx_pow2((int)(uimm & 31u)); int32_t v;
        if (f != f) v = 0;
        else if (f < -2147483648.0f) { v = INT32_MIN; *vscr |= VMX_SAT; }
        else if (f >= 2147483648.0f) { v = INT32_MAX; *vscr |= VMX_SAT; }
        else v = (int32_t)f;
        vmx_sw(r, i, (uint32_t)v); } VMX_OUT; }
VMX_INL void vmx_vctuxs(VMX_D, VMX_B, unsigned uimm, uint32_t* vscr) { const uint8_t* b = (const uint8_t*)b_; uint8_t r[16];
    for (int i = 0; i < 4; i++) { float f = vmx_f(vmx_w(b, i)) * vmx_pow2((int)(uimm & 31u)); uint32_t v;
        if (f != f) v = 0;
        else if (f < 0.0f) { v = 0; *vscr |= VMX_SAT; }
        else if (f >= 4294967296.0f) { v = 0xFFFFFFFFu; *vscr |= VMX_SAT; }
        else v = (uint32_t)f;
        vmx_sw(r, i, v); } VMX_OUT; }

/* ---- compares; record forms return the CR6 nibble [all, 0, none, 0] ---- */
#define VMX_CMP(name, cnt, get, set, test) \
VMX_INL uint32_t name(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP; int all = 1, none = 1; \
    for (int i = 0; i < cnt; i++) { uint64_t x = get(a, i), y = get(b, i); (void)x; (void)y; int t = (test); \
        all &= t; none &= !t; set(r, i, t ? ~0ull : 0); } VMX_OUT; return (uint32_t)(all << 3 | none << 1); }
VMX_INL void vmx_sb64(uint8_t* v, int i, uint64_t x) { v[i] = (uint8_t)x; }
VMX_INL void vmx_sh64(uint8_t* v, int i, uint64_t x) { vmx_sh(v, i, (uint16_t)x); }
VMX_INL void vmx_sw64(uint8_t* v, int i, uint64_t x) { vmx_sw(v, i, (uint32_t)x); }
VMX_CMP(vmx_vcmpequb, 16, vmx_b, vmx_sb64, x == y)
VMX_CMP(vmx_vcmpequh, 8, vmx_h, vmx_sh64, x == y)
VMX_CMP(vmx_vcmpequw, 4, vmx_w, vmx_sw64, x == y)
VMX_CMP(vmx_vcmpgtub, 16, vmx_b, vmx_sb64, x > y)
VMX_CMP(vmx_vcmpgtuh, 8, vmx_h, vmx_sh64, x > y)
VMX_CMP(vmx_vcmpgtuw, 4, vmx_w, vmx_sw64, x > y)
VMX_CMP(vmx_vcmpgtsb, 16, vmx_b, vmx_sb64, (int8_t)x > (int8_t)y)
VMX_CMP(vmx_vcmpgtsh, 8, vmx_h, vmx_sh64, (int16_t)x > (int16_t)y)
VMX_CMP(vmx_vcmpgtsw, 4, vmx_w, vmx_sw64, (int32_t)x > (int32_t)y)
VMX_CMP(VMX_REF(vmx_vcmpeqfp), 4, vmx_w, vmx_sw64, vmx_f((uint32_t)x) == vmx_f((uint32_t)y))
VMX_CMP(VMX_REF(vmx_vcmpgefp), 4, vmx_w, vmx_sw64, vmx_f((uint32_t)x) >= vmx_f((uint32_t)y))
VMX_CMP(VMX_REF(vmx_vcmpgtfp), 4, vmx_w, vmx_sw64, vmx_f((uint32_t)x) > vmx_f((uint32_t)y))
/* vcmpbfp: bit 31 = !(a <= b), bit 30 = !(a >= -b); CR6 EQ = every lane in bounds */
VMX_INL uint32_t vmx_vcmpbfp(VMX_D, VMX_A, VMX_B) { VMX_ARGS_PREP; int inb = 1;
    for (int i = 0; i < 4; i++) { float x = vmx_f(vmx_w(a, i)), y = vmx_f(vmx_w(b, i));
        uint32_t v = (!(x <= y) ? 0x80000000u : 0) | (!(x >= -y) ? 0x40000000u : 0);
        inb &= v == 0; vmx_sw(r, i, v); } VMX_OUT; return (uint32_t)(inb << 1); }

#if VMX_X86
/* ======================================================================
 * x86-64-v3 forms. A register's words are big-endian in memory; one byte
 * shuffle puts them in host order in the lanes, element i in lane i, and
 * back. Each form is its scalar twin above step for step.
 * ==================================================================== */
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define VMX_X static inline __attribute__((always_inline))
#else
#define VMX_X static __forceinline
#endif
VMX_X __m128i vmx_x_ld(const void* p) {
    return _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)p),
                            _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
}
VMX_X void vmx_x_st(void* p, __m128i v) {
    _mm_storeu_si128((__m128i*)p, _mm_shuffle_epi8(v,
                     _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12)));
}
VMX_X __m128 vmx_x_f(__m128i v) { return _mm_castsi128_ps(v); }
VMX_X __m128i vmx_x_u(__m128 v) { return _mm_castps_si128(v); }
/* vmx_flush: with NJ, a zero exponent (zero or denormal) becomes the signed zero */
VMX_X __m128i vmx_x_flush(__m128i v, uint32_t vscr) {
    if (!(vscr & VMX_NJ)) return v;
    const __m128i z = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0x7F800000)), _mm_setzero_si128());
    return _mm_blendv_epi8(v, _mm_and_si128(v, _mm_set1_epi32((int)0x80000000u)), z);
}
/* vmx_isnan: (x & 0x7FFFFFFF) > 0x7F800000, signed compare (both sides positive) */
VMX_X __m128i vmx_x_isnan(__m128i v) {
    return _mm_cmpgt_epi32(_mm_and_si128(v, _mm_set1_epi32(0x7FFFFFFF)), _mm_set1_epi32(0x7F800000));
}
VMX_X __m128i vmx_x_fix(__m128i r) {
    return _mm_blendv_epi8(r, _mm_set1_epi32(0x7FC00000), vmx_x_isnan(r));
}
/* vmx_vnan2: A's NaN quieted, else B's, else the fixed result */
VMX_X __m128i vmx_x_nan2(__m128i r, __m128i a, __m128i b) {
    const __m128i q = _mm_set1_epi32(0x7FC00000);
    r = _mm_blendv_epi8(vmx_x_fix(r), _mm_or_si128(b, q), vmx_x_isnan(b));
    return _mm_blendv_epi8(r, _mm_or_si128(a, q), vmx_x_isnan(a));
}
VMX_X __m128i vmx_x_nan1(__m128i r, __m128i b) {
    return _mm_blendv_epi8(vmx_x_fix(r), _mm_or_si128(b, _mm_set1_epi32(0x7FC00000)), vmx_x_isnan(b));
}

VMX_INL void vmx_vslw(VMX_D, VMX_A, VMX_B) {
    vmx_x_st(d_, _mm_sllv_epi32(vmx_x_ld(a_), _mm_and_si128(vmx_x_ld(b_), _mm_set1_epi32(31))));
}
VMX_INL void vmx_vsrw(VMX_D, VMX_A, VMX_B) {
    vmx_x_st(d_, _mm_srlv_epi32(vmx_x_ld(a_), _mm_and_si128(vmx_x_ld(b_), _mm_set1_epi32(31))));
}
VMX_INL void vmx_vsraw(VMX_D, VMX_A, VMX_B) {
    vmx_x_st(d_, _mm_srav_epi32(vmx_x_ld(a_), _mm_and_si128(vmx_x_ld(b_), _mm_set1_epi32(31))));
}

VMX_INL void vmx_vaddfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const __m128i x = vmx_x_flush(vmx_x_ld(a_), vscr), y = vmx_x_flush(vmx_x_ld(b_), vscr);
    vmx_x_st(d_, vmx_x_flush(vmx_x_nan2(vmx_x_u(_mm_add_ps(vmx_x_f(x), vmx_x_f(y))), x, y), vscr));
}
VMX_INL void vmx_vsubfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const __m128i x = vmx_x_flush(vmx_x_ld(a_), vscr), y = vmx_x_flush(vmx_x_ld(b_), vscr);
    vmx_x_st(d_, vmx_x_flush(vmx_x_nan2(vmx_x_u(_mm_sub_ps(vmx_x_f(x), vmx_x_f(y))), x, y), vscr));
}
/* vmx_maxf / vmx_minf without the NaN case (vmx_x_nan2 decides those lanes):
 * the larger (smaller) one, and for equal operands their AND (OR) -- the same
 * word, or for two zeros +0 unless both are -0 (-0 unless both are +0). */
VMX_INL void vmx_vmaxfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const __m128i x = vmx_x_ld(a_), y = vmx_x_ld(b_);
    const __m128 fx = vmx_x_f(x), fy = vmx_x_f(y);
    __m128i z = _mm_and_si128(x, y);
    z = _mm_blendv_epi8(z, y, vmx_x_u(_mm_cmplt_ps(fx, fy)));
    z = _mm_blendv_epi8(z, x, vmx_x_u(_mm_cmpgt_ps(fx, fy)));
    vmx_x_st(d_, vmx_x_flush(vmx_x_nan2(z, x, y), vscr));
}
VMX_INL void vmx_vminfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const __m128i x = vmx_x_ld(a_), y = vmx_x_ld(b_);
    const __m128 fx = vmx_x_f(x), fy = vmx_x_f(y);
    __m128i z = _mm_or_si128(x, y);
    z = _mm_blendv_epi8(z, y, vmx_x_u(_mm_cmpgt_ps(fx, fy)));
    z = _mm_blendv_epi8(z, x, vmx_x_u(_mm_cmplt_ps(fx, fy)));
    vmx_x_st(d_, vmx_x_flush(vmx_x_nan2(z, x, y), vscr));
}
/* fmaf(x, z, y) and -fmaf(x, z, -y), one rounding each */
VMX_INL void vmx_vmaddfp(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t vscr) {
    const __m128i x = vmx_x_flush(vmx_x_ld(a_), vscr), y = vmx_x_flush(vmx_x_ld(b_), vscr),
                  z = vmx_x_flush(vmx_x_ld(c_), vscr);
    vmx_x_st(d_, vmx_x_flush(vmx_x_fix(vmx_x_u(_mm_fmadd_ps(vmx_x_f(x), vmx_x_f(z), vmx_x_f(y)))), vscr));
}
VMX_INL void vmx_vnmsubfp(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t vscr) {
    const __m128i x = vmx_x_flush(vmx_x_ld(a_), vscr), y = vmx_x_flush(vmx_x_ld(b_), vscr),
                  z = vmx_x_flush(vmx_x_ld(c_), vscr);
    const __m128i r = _mm_xor_si128(vmx_x_u(_mm_fmsub_ps(vmx_x_f(x), vmx_x_f(z), vmx_x_f(y))),
                                    _mm_set1_epi32((int)0x80000000u));
    vmx_x_st(d_, vmx_x_flush(vmx_x_fix(r), vscr));
}
VMX_INL void vmx_vrefp(VMX_D, VMX_B, uint32_t vscr) {
    const __m128i y = vmx_x_flush(vmx_x_ld(b_), vscr);
    vmx_x_st(d_, vmx_x_flush(vmx_x_nan1(vmx_x_u(_mm_div_ps(_mm_set1_ps(1.0f), vmx_x_f(y))), y), vscr));
}
VMX_INL void vmx_vrsqrtefp(VMX_D, VMX_B, uint32_t vscr) {
    const __m128i y = vmx_x_flush(vmx_x_ld(b_), vscr);
    const __m128 q = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(vmx_x_f(y)));
    vmx_x_st(d_, vmx_x_flush(vmx_x_nan1(vmx_x_u(q), y), vscr));
}
/* (float)(int32_t)w * 2^-uimm: the conversion rounds once, the scale is exact
 * (the smallest magnitude, 2^-31, is a normal single), as ldexpf */
VMX_INL void vmx_vcfsx(VMX_D, VMX_B, unsigned uimm) {
    const __m128 f = _mm_cvtepi32_ps(vmx_x_ld(b_));
    vmx_x_st(d_, vmx_x_u(_mm_mul_ps(f, _mm_set1_ps(vmx_pow2(-(int)(uimm & 31u))))));
}
/* ordered compares: a NaN lane is false, as in C */
VMX_X uint32_t vmx_x_cmp(void* d_, __m128 m) {
    _mm_storeu_si128((__m128i*)d_, vmx_x_u(m));
    const int bits = _mm_movemask_ps(m);
    return (uint32_t)((bits == 0xF) << 3 | (bits == 0) << 1);
}
VMX_INL uint32_t vmx_vcmpeqfp(VMX_D, VMX_A, VMX_B) {
    return vmx_x_cmp(d_, _mm_cmpeq_ps(vmx_x_f(vmx_x_ld(a_)), vmx_x_f(vmx_x_ld(b_))));
}
VMX_INL uint32_t vmx_vcmpgefp(VMX_D, VMX_A, VMX_B) {
    return vmx_x_cmp(d_, _mm_cmpge_ps(vmx_x_f(vmx_x_ld(a_)), vmx_x_f(vmx_x_ld(b_))));
}
VMX_INL uint32_t vmx_vcmpgtfp(VMX_D, VMX_A, VMX_B) {
    return vmx_x_cmp(d_, _mm_cmpgt_ps(vmx_x_f(vmx_x_ld(a_)), vmx_x_f(vmx_x_ld(b_))));
}
#endif /* VMX_X86 */

#if VMX_NEON
/* ======================================================================
 * arm64 (NEON) forms, as the x86-64-v3 ones: one rev32 puts a register's
 * big-endian words in host order in the lanes, element i in lane i, and back.
 * Each form is its scalar twin above step for step. Threads run with FPCR.FZ
 * and FPCR.DN clear (macOS's and Linux's default), so a denormal is computed
 * here as the scalar code computes it; VSCR[NJ] flushing and the NaN rules
 * are done on the bits, as there.
 * ==================================================================== */
#include <arm_neon.h>
#define VMX_N static inline __attribute__((always_inline))
VMX_N uint32x4_t vmx_n_ld(const void* p) {
    return vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8((const uint8_t*)p)));
}
VMX_N void vmx_n_st(void* p, uint32x4_t v) { vst1q_u8((uint8_t*)p, vrev32q_u8(vreinterpretq_u8_u32(v))); }
VMX_N float32x4_t vmx_n_f(uint32x4_t v) { return vreinterpretq_f32_u32(v); }
VMX_N uint32x4_t vmx_n_u(float32x4_t v) { return vreinterpretq_u32_f32(v); }
/* vmx_flush: with NJ, a zero exponent (zero or denormal) keeps only its sign */
VMX_N uint32x4_t vmx_n_flush(uint32x4_t v, uint32_t vscr) {
    if (!(vscr & VMX_NJ)) return v;
    const uint32x4_t z = vceqq_u32(vandq_u32(v, vdupq_n_u32(0x7F800000u)), vdupq_n_u32(0));
    return vbicq_u32(v, vandq_u32(z, vdupq_n_u32(0x7FFFFFFFu)));
}
VMX_N uint32x4_t vmx_n_isnan(uint32x4_t v) {
    return vcgtq_u32(vandq_u32(v, vdupq_n_u32(0x7FFFFFFFu)), vdupq_n_u32(0x7F800000u));
}
VMX_N uint32x4_t vmx_n_fix(uint32x4_t r) { return vbslq_u32(vmx_n_isnan(r), vdupq_n_u32(0x7FC00000u), r); }
/* vmx_vnan2 / vmx_vnan1: A's NaN quieted, else B's, else the fixed result */
VMX_N uint32x4_t vmx_n_nan1(uint32x4_t r, uint32x4_t b) {
    return vbslq_u32(vmx_n_isnan(b), vorrq_u32(b, vdupq_n_u32(0x7FC00000u)), vmx_n_fix(r));
}
VMX_N uint32x4_t vmx_n_nan2(uint32x4_t r, uint32x4_t a, uint32x4_t b) {
    return vbslq_u32(vmx_n_isnan(a), vorrq_u32(a, vdupq_n_u32(0x7FC00000u)), vmx_n_nan1(r, b));
}

/* the count is the low 5 bits of B's word; USHL/SSHL shift right by a negative one */
VMX_N int32x4_t vmx_n_count(const void* b_) {
    return vreinterpretq_s32_u32(vandq_u32(vmx_n_ld(b_), vdupq_n_u32(31)));
}
VMX_INL void vmx_vslw(VMX_D, VMX_A, VMX_B) { vmx_n_st(d_, vshlq_u32(vmx_n_ld(a_), vmx_n_count(b_))); }
VMX_INL void vmx_vsrw(VMX_D, VMX_A, VMX_B) { vmx_n_st(d_, vshlq_u32(vmx_n_ld(a_), vnegq_s32(vmx_n_count(b_)))); }
VMX_INL void vmx_vsraw(VMX_D, VMX_A, VMX_B) {
    vmx_n_st(d_, vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(vmx_n_ld(a_)), vnegq_s32(vmx_n_count(b_)))));
}

VMX_INL void vmx_vaddfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const uint32x4_t x = vmx_n_flush(vmx_n_ld(a_), vscr), y = vmx_n_flush(vmx_n_ld(b_), vscr);
    vmx_n_st(d_, vmx_n_flush(vmx_n_nan2(vmx_n_u(vaddq_f32(vmx_n_f(x), vmx_n_f(y))), x, y), vscr));
}
VMX_INL void vmx_vsubfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const uint32x4_t x = vmx_n_flush(vmx_n_ld(a_), vscr), y = vmx_n_flush(vmx_n_ld(b_), vscr);
    vmx_n_st(d_, vmx_n_flush(vmx_n_nan2(vmx_n_u(vsubq_f32(vmx_n_f(x), vmx_n_f(y))), x, y), vscr));
}
/* FMAX/FMIN order +0 above -0, as vmx_maxf/vmx_minf do; vmx_n_nan2 decides the NaN lanes */
VMX_INL void vmx_vmaxfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const uint32x4_t x = vmx_n_ld(a_), y = vmx_n_ld(b_);
    vmx_n_st(d_, vmx_n_flush(vmx_n_nan2(vmx_n_u(vmaxq_f32(vmx_n_f(x), vmx_n_f(y))), x, y), vscr));
}
VMX_INL void vmx_vminfp(VMX_D, VMX_A, VMX_B, uint32_t vscr) {
    const uint32x4_t x = vmx_n_ld(a_), y = vmx_n_ld(b_);
    vmx_n_st(d_, vmx_n_flush(vmx_n_nan2(vmx_n_u(vminq_f32(vmx_n_f(x), vmx_n_f(y))), x, y), vscr));
}
/* fmaf(x, z, y), and fmaf(x, z, -y) with the sign flipped as bits; one rounding each */
VMX_INL void vmx_vmaddfp(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t vscr) {
    const uint32x4_t x = vmx_n_flush(vmx_n_ld(a_), vscr), y = vmx_n_flush(vmx_n_ld(b_), vscr),
                     z = vmx_n_flush(vmx_n_ld(c_), vscr);
    vmx_n_st(d_, vmx_n_flush(vmx_n_fix(vmx_n_u(vfmaq_f32(vmx_n_f(y), vmx_n_f(x), vmx_n_f(z)))), vscr));
}
VMX_INL void vmx_vnmsubfp(VMX_D, VMX_A, VMX_B, VMX_C, uint32_t vscr) {
    const uint32x4_t x = vmx_n_flush(vmx_n_ld(a_), vscr), y = vmx_n_flush(vmx_n_ld(b_), vscr),
                     z = vmx_n_flush(vmx_n_ld(c_), vscr);
    const uint32x4_t r = veorq_u32(vmx_n_u(vfmaq_f32(vnegq_f32(vmx_n_f(y)), vmx_n_f(x), vmx_n_f(z))),
                                   vdupq_n_u32(0x80000000u));
    vmx_n_st(d_, vmx_n_flush(vmx_n_fix(r), vscr));
}
/* IEEE division and square root, as 1.0f / f and 1.0f / sqrtf(f) (not FRECPE/FRSQRTE) */
VMX_INL void vmx_vrefp(VMX_D, VMX_B, uint32_t vscr) {
    const uint32x4_t y = vmx_n_flush(vmx_n_ld(b_), vscr);
    vmx_n_st(d_, vmx_n_flush(vmx_n_nan1(vmx_n_u(vdivq_f32(vdupq_n_f32(1.0f), vmx_n_f(y))), y), vscr));
}
VMX_INL void vmx_vrsqrtefp(VMX_D, VMX_B, uint32_t vscr) {
    const uint32x4_t y = vmx_n_flush(vmx_n_ld(b_), vscr);
    const float32x4_t q = vdivq_f32(vdupq_n_f32(1.0f), vsqrtq_f32(vmx_n_f(y)));
    vmx_n_st(d_, vmx_n_flush(vmx_n_nan1(vmx_n_u(q), y), vscr));
}
/* (float)(int32_t)w * 2^-uimm: the conversion rounds once (to nearest), the scale is exact */
VMX_INL void vmx_vcfsx(VMX_D, VMX_B, unsigned uimm) {
    const float32x4_t f = vcvtq_f32_s32(vreinterpretq_s32_u32(vmx_n_ld(b_)));
    vmx_n_st(d_, vmx_n_u(vmulq_n_f32(f, vmx_pow2(-(int)(uimm & 31u)))));
}
/* ordered compares: a NaN lane is false, as in C */
VMX_N uint32_t vmx_n_cmp(void* d_, uint32x4_t m) {
    vmx_n_st(d_, m);
    return (uint32_t)((vminvq_u32(m) != 0) << 3 | (vmaxvq_u32(m) == 0) << 1);
}
VMX_INL uint32_t vmx_vcmpeqfp(VMX_D, VMX_A, VMX_B) {
    return vmx_n_cmp(d_, vceqq_f32(vmx_n_f(vmx_n_ld(a_)), vmx_n_f(vmx_n_ld(b_))));
}
VMX_INL uint32_t vmx_vcmpgefp(VMX_D, VMX_A, VMX_B) {
    return vmx_n_cmp(d_, vcgeq_f32(vmx_n_f(vmx_n_ld(a_)), vmx_n_f(vmx_n_ld(b_))));
}
VMX_INL uint32_t vmx_vcmpgtfp(VMX_D, VMX_A, VMX_B) {
    return vmx_n_cmp(d_, vcgtq_f32(vmx_n_f(vmx_n_ld(a_)), vmx_n_f(vmx_n_ld(b_))));
}
#endif /* VMX_NEON */

#ifdef __cplusplus
}
#endif
#endif /* PPU_VMX_H */
