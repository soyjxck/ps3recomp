/*
 * Standalone unit test for the RSX fragment-program decompiler.
 *
 * Build (no D3D12 dependency), from this directory:
 *   cc -std=c11 -Wall -Wextra -I../../../include \
 *       test_fp_decompiler.c ../rsx_fp_decompiler.c -o test_fp
 *
 * Exercises the decode path (word swap, source swizzle/type, opcode -> HLSL,
 * dest mask, inline constants) by hand-assembling tiny programs and checking
 * the generated HLSL contains the expected expressions, then the colour
 * export selection: which register file the fragment colour comes out of, and
 * how many SV_TARGETs a program that writes more than one export gets.
 */

#define _POSIX_C_SOURCE 200809L   /* setenv */
#include "../rsx_fp_decompiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

/* Store host word H into guest bytes so that rsx_fp_read_word(p) == H.
 * rsx_fp_read_word does be32-load then 16-bit half-swap, so we invert: the
 * guest stores big-endian of swap16(H). */
static void put_word(u8* p, u32 h)
{
    u32 be = (h << 16) | (h >> 16);
    p[0] = (u8)(be >> 24); p[1] = (u8)(be >> 16);
    p[2] = (u8)(be >> 8);  p[3] = (u8)(be);
}

static void put_float(u8* p, float f)
{
    u32 h; memcpy(&h, &f, 4);
    put_word(p, h);
}

static void check(const char* test, const char* hlsl, const char* needle)
{
    if (strstr(hlsl, needle)) {
        printf("[PASS] %s -- found \"%s\"\n", test, needle);
        g_pass++;
    } else {
        printf("[FAIL] %s -- missing \"%s\"\nHLSL:\n%s\n", test, needle, hlsl);
        g_fail++;
    }
}

static void check_absent(const char* test, const char* hlsl, const char* needle)
{
    if (!strstr(hlsl, needle)) {
        printf("[PASS] %s -- no \"%s\"\n", test, needle);
        g_pass++;
    } else {
        printf("[FAIL] %s -- unexpected \"%s\"\nHLSL:\n%s\n", test, needle, hlsl);
        g_fail++;
    }
}

static void check_true(const char* test, int cond)
{
    if (cond) { printf("[PASS] %s\n", test); g_pass++; }
    else      { printf("[FAIL] %s\n", test); g_fail++; }
}

/* OPDEST/SRC bit helpers (mirror the decoder's macros). */
#define OPC(x)      ((u32)(x) << 24)
#define OUTMASK_ALL (0xFu << 9)
#define END         (1u << 0)
#define OUTREG(i)   ((u32)(i) << 1)
#define OUTHALF     (1u << 7)
#define INSRC(x)    ((u32)(x) << 13)
#define TEXU(x)     ((u32)(x) << 17)
#define SWZ_IDENT   ((0u<<9)|(1u<<11)|(2u<<13)|(3u<<15))
#define T_TEMP      0u
#define T_INPUT     1u
#define T_CONST     2u
#define REG(i)      ((u32)(i) << 2)
/* The execution condition, exec_if lt/eq/gt at [18:20] with its swizzle at
 * [21:28]. All three bits set is unconditional; an all-zero source word is
 * "never", which suppresses the write -- so every instruction below carries
 * this, as a real program's does. */
#define EXEC_ALWAYS ((7u << 18) | (0u << 21) | (1u << 23) | (2u << 25) | (3u << 27))

/* SHADER_CONTROL: 0x40 exports fp32 (colour from r0/r2/r3/r4), clear exports
 * fp16 (colour from h0/h4/h6/h8). */
#define CTRL_32BIT  0x40u
#define CTRL_16BIT  0x00u

int main(void)
{
    char hlsl[16384];

    /* RSX_FP_DEBUG=<hash>:<n>:<reg>: the named register, captured after the
     * program's n-th instruction, becomes colour target 0 with alpha 1 --
     * in that program only. The variable is read once, on the first
     * decompile, so this runs first. */
    {
        u8 prog[32];
        u32 mask_xy = (1u<<9)|(1u<<10);
        put_word(prog + 0, OPC(0x02) | mask_xy | OUTREG(1));          /* MUL r1.xy, r0, r0 */
        put_word(prog + 4, T_TEMP | REG(0) | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, T_TEMP | REG(0) | SWZ_IDENT);
        put_word(prog + 12, 0);
        put_word(prog + 16, OPC(0x01) | OUTMASK_ALL | INSRC(1) | END); /* MOV r0, COL0 */
        put_word(prog + 20, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 24, 0);
        put_word(prog + 28, 0);
        char spec[64];
        snprintf(spec, sizeof spec, "%llx:1:r1.xyyy", (unsigned long long)
                 rsx_fp_structural_hash(prog, sizeof prog, 1469598103934665603ull));
        setenv("RSX_FP_DEBUG", spec, 1);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("fp_debug_capture", hlsl,
              "r[1].xy = _v.xy; }\n"
              "    _rsx_dbg = (float4)((r[1]).xyyy);\n"
              "    { float4 _v = (float4)((input.col0).xyzw);");
        check("fp_debug_export", hlsl, "_d.w = 1.0; r[0] = _d; h[0] = _d; }\n"
                                       "    float4 _o = r[0];");
        u8 other[16];
        memcpy(other, prog + 16, sizeof other);
        rsx_fp_decompile(other, sizeof(other), CTRL_32BIT, hlsl, sizeof(hlsl));
        check_absent("fp_debug_other_program", hlsl, "_rsx_dbg");
        unsetenv("RSX_FP_DEBUG");
    }

    /* Test 1: MOV r0, COL0  (write all, END). */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | END); /* MOV, COL0 */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);        /* src0 = INPUT */
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        int n = rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        printf("-- test1 MOV: %d instr\n", n);
        check("mov_dest",   hlsl, "r[0].xyzw =");
        check("mov_input",  hlsl, "input.col0");
        check("mov_entry",  hlsl, "float4 main(PSInput input) : SV_TARGET {");
        check("mov_return", hlsl, "    float4 _o = r[0];\n"
                                  "    return (_o == _o) ? _o : (float4)0;\n}\n");
    }

    /* Test 2: MUL r1.xy, r0, r0  (partial write mask). */
    {
        u8 prog[16];
        u32 mask_xy = (1u<<9)|(1u<<10);
        put_word(prog + 0, OPC(0x02) | mask_xy | OUTREG(1) | END); /* MUL, dest r1 */
        put_word(prog + 4, T_TEMP | REG(0) | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, T_TEMP | REG(0) | SWZ_IDENT);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("mul_mask", hlsl, "r[1].xy =");
        check("mul_expr", hlsl, "(r[0]).xyzw) * ((r[0]).xyzw)");
    }

    /* Test 3: MAD r0, r0, CONST, r0 with inline constant + saturate. */
    {
        u8 prog[32];
        put_word(prog + 0, OPC(0x04) | OUTMASK_ALL | (1u<<31) | END); /* MAD + SAT */
        put_word(prog + 4, T_TEMP  | REG(0) | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, T_CONST | SWZ_IDENT);                      /* src1 CONST */
        put_word(prog + 12, T_TEMP | REG(0) | SWZ_IDENT);
        put_float(prog + 16, 0.5f);  /* inline constant float4 */
        put_float(prog + 20, 0.25f);
        put_float(prog + 24, 0.0f);
        put_float(prog + 28, 1.0f);
        int n = rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        printf("-- test3 MAD: %d instr\n", n);
        check("mad_sat",   hlsl, "saturate(");
        check("mad_const", hlsl, "float4(0.5,0.25,0,1)");

        /* ...and the buffered variant, which hoists that constant into b1. */
        u32 nk = 0;
        rsx_fp_decompile_buffered_ex(prog, sizeof(prog), CTRL_32BIT, 0,
                                     hlsl, sizeof(hlsl), &nk);
        check_true("mad_buffered_count", nk == 1);
        check("mad_buffered", hlsl, "fp_constants[0]");
    }

    /* Test 4: TEX r0, TC0 from texture unit 3. */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x17) | OUTMASK_ALL | INSRC(4) | TEXU(3) | END); /* TEX */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("tex_sample", hlsl, "rsx_tex[3].Sample(rsx_samp[3]");
        check("tex_coord",  hlsl, "input.tc0");
    }

    /* Test 5: negate + abs on a source (MOV r0, -|r1|). */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | END);
        put_word(prog + 4, T_TEMP | REG(1) | SWZ_IDENT | (1u<<17) | (1u<<29)
                           | EXEC_ALWAYS); /* neg+abs */
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("neg_abs", hlsl, "-(abs((r[1]).xyzw))");
    }

    /* Test 6: the export register file follows SHADER_CONTROL, not whichever
     * temp was written. A program writing h0 exports h0 with 16-bit exports
     * and r0 with 32-bit ones. */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | OUTHALF | END);
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_16BIT, hlsl, sizeof(hlsl));
        check("h0_16bit", hlsl, "float4 _o = h[0];");
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("h0_32bit", hlsl, "float4 _o = r[0];");
    }

    /* Test 7: MRT. The second colour export is r2 with 32-bit exports and h4
     * with 16-bit ones, so the same instruction is a second render target in
     * one mode and an ordinary temp write in the other. A deferred pass
     * writing both planes has to come out as a struct of SV_TARGETs; a
     * material shader accumulating into r2 while its colour goes to h0 must
     * not. */
    {
        u8 prog[32];
        put_word(prog +  0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | OUTREG(0));
        put_word(prog +  4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog +  8, 0);
        put_word(prog + 12, 0);
        put_word(prog + 16, OPC(0x01) | OUTMASK_ALL | INSRC(2) | OUTREG(2) | END);
        put_word(prog + 20, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 24, 0);
        put_word(prog + 28, 0);

        int n = rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        printf("-- test7 MRT: %d instr\n", n);
        check("mrt_struct",  hlsl, "struct PSOutput {\n"
                                   "    float4 t0 : SV_TARGET0;\n"
                                   "    float4 t1 : SV_TARGET1;\n"
                                   "};\n");
        check("mrt_entry",   hlsl, "PSOutput main(PSInput input) {");
        check("mrt_t0",      hlsl, "float4 _o0 = r[0];\n"
                                   "    _out.t0 = (_o0 == _o0) ? _o0 : (float4)0;");
        check("mrt_t1",      hlsl, "float4 _o1 = r[2];\n"
                                   "    _out.t1 = (_o1 == _o1) ? _o1 : (float4)0;");
        check("mrt_return",  hlsl, "    return _out;\n}\n");
        check_absent("mrt_no_target2", hlsl, "SV_TARGET2");

        /* ...and the alpha test still patches it, testing target 0's alpha. */
        check_true("mrt_alpha_patched",
                   rsx_fp_apply_alpha_test(hlsl, sizeof(hlsl), 0x204u, 0.5f) == 1);
        check("mrt_alpha_type", hlsl, "PSOutput _rsx_out = _out;");
        check("mrt_alpha_cmp",  hlsl, "if (!(_rsx_out.t0.a > _rsx_alpha_ref)) discard;");

        /* The same program under 16-bit exports: r2 is a plain temp there, so
         * the colour is h0 alone and nothing about the output changes. */
        rsx_fp_decompile(prog, sizeof(prog), CTRL_16BIT, hlsl, sizeof(hlsl));
        check_absent("mrt_16bit_single", hlsl, "PSOutput");
        check("mrt_16bit_h0", hlsl, "float4 _o = h[0];");
    }

    /* Test 8: all four exports, and the highest written one decides how many
     * targets come out -- r3 alone still declares SV_TARGET0 through 2. */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | OUTREG(3) | END);
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("r3_target0", hlsl, "float4 t0 : SV_TARGET0;");
        check("r3_target2", hlsl, "float4 t2 : SV_TARGET2;");
        check("r3_reads",   hlsl, "float4 _o2 = r[3];");
        check_absent("r3_no_target3", hlsl, "SV_TARGET3");
    }

    /* Test 9: the structural hash separates programs that differ only in
     * which export register they write -- so a pipeline cache keyed on it
     * cannot hand an MRT program the single-target program's shader. It is
     * the instruction bytes that are hashed, and the destination register is
     * one of them; nothing had to be added for the MRT case. */
    {
        u8 single[16], mrt[16];
        put_word(single + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | OUTREG(0) | END);
        put_word(single + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(single + 8, 0);
        put_word(single + 12, 0);
        memcpy(mrt, single, sizeof mrt);
        put_word(mrt + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | OUTREG(2) | END);

        const u64 seed = 1469598103934665603ull;
        const u64 hs = rsx_fp_structural_hash(single, sizeof single, seed);
        const u64 hm = rsx_fp_structural_hash(mrt, sizeof mrt, seed);
        check_true("hash_nonzero", hs != 0 && hm != 0);
        check_true("hash_distinguishes_mrt", hs != hm);
        check_true("hash_stable",
                   hs == rsx_fp_structural_hash(single, sizeof single, seed));
    }

    /* Test 10: IFE with an else branch. The IFE's SRC1/SRC2 hold the else and
     * end addresses in words; each side must land in its own HLSL block, not
     * run one after the other (Crazy Taxi's uber shader drew the city black
     * that way). @0 IFE(cc0.x > 0) else=@32 end=@48; @16 MOV r0,COL0;
     * @32 MOV r0,COL1; @48 LIF r1,r0 END. */
    {
        u8 prog[64];
        const u32 exec_gt_x = (1u << 20);   /* exec_if gr, cond swizzle x */
        put_word(prog +  0, OPC(0x02));
        put_word(prog +  4, exec_gt_x);
        put_word(prog +  8, (1u << 31) | (32u >> 2));   /* branch, else */
        put_word(prog + 12, 48u >> 2);                  /* end */
        put_word(prog + 16, OPC(0x01) | OUTMASK_ALL | INSRC(1));
        put_word(prog + 20, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 24, 0);
        put_word(prog + 28, 0);
        put_word(prog + 32, OPC(0x01) | OUTMASK_ALL | INSRC(2));
        put_word(prog + 36, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 40, 0);
        put_word(prog + 44, 0);
        put_word(prog + 48, OPC(0x3C) | OUTMASK_ALL | OUTREG(1) | END);
        put_word(prog + 52, T_TEMP | REG(0) | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 56, 0);
        put_word(prog + 60, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        const char* ife  = strstr(hlsl, "    if (cc0.x > 0.0) {\n");
        const char* then = strstr(hlsl, "input.col0");
        const char* els  = strstr(hlsl, "    } else {\n");
        const char* alt  = strstr(hlsl, "input.col1");
        const char* lif  = strstr(hlsl, "float4(1.0, ");
        check_true("ife_order", ife && then && els && alt && lif &&
                                ife < then && then < els && els < alt && alt < lif);
        check_absent("ife_not_skipped", hlsl, "flow-control op skipped");
        check_absent("lif_handled", hlsl, "unhandled FP opcode");
    }

    /* A unit bound to a 3D texture (SET_TEXTURE_FORMAT dimension 3) is
     * declared Texture3D and sampled with three coordinates; TXP divides
     * them by w. inFAMOUS 2's occlusion volumes look up an 8x8x8 cone LUT. */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x17) | OUTMASK_ALL | INSRC(4) | TEXU(0) | END); /* TEX TC0 */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_set_volume_units(1u);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("vol_decl",   hlsl, "Texture3D   rsx_tex0 : register(t0);");
        check("vol_sample", hlsl, "rsx_tex0.Sample(rsx_samp[0], ((input.tc0).xyzw).xyz)");
        put_word(prog + 0, OPC(0x18) | OUTMASK_ALL | INSRC(4) | TEXU(0) | END); /* TXP TC0 */
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("vol_txp", hlsl, "rsx_tex0.Sample(rsx_samp[0], ((input.tc0).xyzw).xyz / ((input.tc0).xyzw).w)");
        /* No volume units: the array declaration and 2D sample as before. */
        rsx_fp_set_volume_units(0);
        put_word(prog + 0, OPC(0x17) | OUTMASK_ALL | INSRC(4) | TEXU(0) | END);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("vol_none", hlsl, "rsx_tex[0].Sample(rsx_samp[0], ((input.tc0).xyzw).xy)");
        check_absent("vol_none_decl", hlsl, "Texture3D");
    }

    /* A predicated write is a select, not a blend: MOV r0, COL0 executed
     * only where CC0 > 0 must keep r0 bit for bit in the other components.
     * lerp(r0, v, mask) is arithmetic, so an Inf/NaN in either operand
     * leaks into the components the condition leaves alone. */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | END); /* MOV r0, COL0 */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | (1u << 20) |          /* exec_if gt */
                           (0u << 21) | (1u << 23) | (2u << 25) | (3u << 27));
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("pred_select", hlsl,
              "r[0].xyzw = (_p.xyzw != 0.0) ? _v.xyzw : r[0].xyzw;");
        check_absent("pred_no_lerp", hlsl, "lerp(");
    }

    /* EXP_TEX (OPDEST bit 21, the _bx2 modifier) returns the fetch as x*2-1
     * on every lane, on any unit, and the unit's own texel conversions step
     * aside for it (RPCS3 _process_texel). inFAMOUS 2 reads its tangent-space
     * normal maps that way; read unexpanded, every normal leaned towards
     * (1,1,1). */
    {
        u8 prog[16];
        u32 ops[16] = {0};
        put_word(prog + 0, OPC(0x17) | OUTMASK_ALL | INSRC(4) | TEXU(0) | (1u << 21) | END); /* TEX TC0, EXP_TEX */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("bx2_expanded", hlsl, "((rsx_tex[0].Sample(rsx_samp[0], ((input.tc0).xyzw).xy)) * 2.0 - 1.0)");

        /* Gamma and the biased remap on every lane of the unit: a _bx2 fetch
         * still comes back as x*2-1, not through the unit's conversions. */
        ops[0] = 0xFFu;
        rsx_fp_set_texel_ops(ops);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        rsx_fp_set_texel_ops(NULL);
        check("bx2_over_texop", hlsl, "((rsx_tex[0].Sample(rsx_samp[0], ((input.tc0).xyzw).xy)) * 2.0 - 1.0)");
        check_absent("bx2_no_texop_call", hlsl, "rsx_texop0(rsx_tex");

        /* Without the bit the unit's conversions apply, as before. */
        put_word(prog + 0, OPC(0x17) | OUTMASK_ALL | INSRC(4) | TEXU(0) | END);
        rsx_fp_set_texel_ops(ops);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        rsx_fp_set_texel_ops(NULL);
        check("texop_without_bx2", hlsl, "rsx_texop0(rsx_tex[0].Sample(rsx_samp[0], ((input.tc0).xyzw).xy))");
        check_absent("texop_without_bx2_math", hlsl, "* 2.0 - 1.0");
    }

    /* SRC2 bit 31 (perspective_corr) on a texcoord read multiplies the input
     * by 1/w (RPCS3 FragmentProgramDecompiler: "Perspective correct flag
     * multiplies the result by 1/w"). inFAMOUS 2's water writes uv*w from the
     * vertex program and samples TC2 with the flag set; read raw, uv*w lands
     * far outside the screen copy and the river samples black. The helper
     * takes the whole input, so the program does not name input.position
     * (which would give it a WPOS-scaled build of its own). */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x17) | OUTMASK_ALL | INSRC(6) | TEXU(0) | END); /* TEX TC2 */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 1u << 31);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("persp_tc_scaled", hlsl, "(input.tc2 * _rsx_inv_w(input))");
        check("persp_helper",    hlsl, "#define _rsx_inv_w(i)");
        check_absent("persp_no_wpos", hlsl, "input.position");

        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("persp_off_raw",    hlsl, "((input.tc2).xyzw).xy");
        check_absent("persp_off_no_helper", hlsl, "_rsx_inv_w");
    }

    /* FACING (input 0xE) is +1 on a front face and -1 on a back face; the
     * face flag is declared only in a program that reads it. */
    {
        u8 prog[16];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(0xE) | END); /* MOV r0, FACING */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check("facing_decl", hlsl, "    bool _rsx_front : SV_IsFrontFace;\n");
        check("facing_read", hlsl, "(input._rsx_front ? float4(1,1,1,1) : float4(-1,-1,-1,-1))");
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | END);   /* MOV r0, COL0 */
        rsx_fp_decompile(prog, sizeof(prog), CTRL_32BIT, hlsl, sizeof(hlsl));
        check_absent("facing_not_declared", hlsl, "SV_IsFrontFace");
    }

    /* WPOS on an antialiased target held at its sample grid: every sample
     * reads the centre of its pixel. A perspective-corrected texcoord keeps
     * reading w from the same SV_POSITION, which the rewrite leaves alone. */
    {
        u8 prog[32];
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(0));            /* MOV r0, WPOS */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        put_word(prog + 16, OPC(0x17) | OUTMASK_ALL | OUTREG(1) | INSRC(6) | TEXU(0) | END);
        put_word(prog + 20, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);              /* TEX r1, TC2 */
        put_word(prog + 24, 0);
        put_word(prog + 28, 1u << 31);
        u32 nk = 0;
        rsx_fp_decompile_buffered_ex(prog, sizeof prog, CTRL_32BIT, 0, hlsl, sizeof hlsl, &nk);
        char before[sizeof hlsl];
        memcpy(before, hlsl, sizeof hlsl);

        check_true("wpos_grid_1x1_noop", rsx_fp_apply_wpos_grid(hlsl, sizeof hlsl, 1, 1) == 0 &&
                                         !strcmp(before, hlsl));
        check_true("wpos_grid_short_buffer",
                   rsx_fp_apply_wpos_grid(hlsl, (u32)strlen(hlsl) + 8, 2, 1) == -1 &&
                   !strcmp(before, hlsl));
        check_true("wpos_grid_2x1", rsx_fp_apply_wpos_grid(hlsl, sizeof hlsl, 2, 1) == 1);
        check("wpos_grid_first_line", hlsl,
              "float4 main(PSInput input) : SV_TARGET {\n"
              "    input.position.xy = floor(input.position.xy / float2(2, 1)) + 0.5;\n");
        check("wpos_grid_inv_w_kept", hlsl, "_rsx_inv_w(input)");
        check_true("wpos_grid_once", rsx_fp_apply_wpos_grid(hlsl, sizeof hlsl, 2, 1) == -1);

        /* Two colour exports: the PSOutput entry point. */
        put_word(prog + 16, OPC(0x01) | OUTMASK_ALL | OUTREG(2) | INSRC(0) | END); /* MOV r2, WPOS */
        put_word(prog + 28, 0);
        rsx_fp_decompile_buffered_ex(prog, sizeof prog, CTRL_32BIT, 0, hlsl, sizeof hlsl, &nk);
        check_true("wpos_grid_mrt_2x2", rsx_fp_apply_wpos_grid(hlsl, sizeof hlsl, 2, 2) == 1);
        check("wpos_grid_mrt_line", hlsl,
              "PSOutput main(PSInput input) {\n"
              "    input.position.xy = floor(input.position.xy / float2(2, 2)) + 0.5;\n");

        /* No WPOS read: byte-identical. */
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | END);      /* MOV r0, COL0 */
        rsx_fp_decompile_buffered_ex(prog, 16, CTRL_32BIT, 0, hlsl, sizeof hlsl, &nk);
        memcpy(before, hlsl, sizeof hlsl);
        check_true("wpos_grid_no_wpos_noop", rsx_fp_apply_wpos_grid(hlsl, sizeof hlsl, 2, 2) == 0 &&
                                             !strcmp(before, hlsl));
    }

    /* WPOS follows the SET_SHADER_WINDOW convention through fp_alpha.y/.z in a
     * buffered program; an unbuffered one has no fp_alpha and reads the host
     * position as before. A program that does not read WPOS gets no helper. */
    {
        u8 prog[16];
        u32 nk = 0;
        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(0) | END); /* MOV r0, WPOS */
        put_word(prog + 4, T_INPUT | SWZ_IDENT | EXEC_ALWAYS);
        put_word(prog + 8, 0);
        put_word(prog + 12, 0);
        rsx_fp_decompile_buffered_ex(prog, sizeof prog, CTRL_32BIT, 0, hlsl, sizeof hlsl, &nk);
        check("wpos_buffered_read",   hlsl, "_rsx_wpos(input.position)");
        check("wpos_buffered_helper", hlsl,
              "#define _rsx_wpos(p) float4((p).x + fp_alpha.z, "
              "(fp_alpha.y != 0.0 ? fp_alpha.y - (p).y : (p).y) + fp_alpha.z, (p).z, (p).w)");
        rsx_fp_decompile(prog, sizeof prog, CTRL_32BIT, hlsl, sizeof hlsl);
        check("wpos_unbuffered_identity", hlsl, "#define _rsx_wpos(p) (p)\n");

        put_word(prog + 0, OPC(0x01) | OUTMASK_ALL | INSRC(1) | END); /* MOV r0, COL0 */
        rsx_fp_decompile_buffered_ex(prog, sizeof prog, CTRL_32BIT, 0, hlsl, sizeof hlsl, &nk);
        check_absent("wpos_no_read_no_helper", hlsl, "_rsx_wpos");

    }

    printf("\n===========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
