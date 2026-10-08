/*
 * ps3recomp - HLSL -> SPIR-V for the Vulkan backend
 *
 * The decompilers emit HLSL; glslang's HLSL front end lowers it to SPIR-V
 * (the same front end rsx_shader_msl.cpp uses on the way to Metal). Unlike
 * that path, the bindings here are explicit, one descriptor set (set 0) for
 * both stages, register + a per-stage shift, so a pipeline layout can be
 * written down once:
 *
 *   binding  0      VPConst      (VS b0)         uniform buffer (dynamic)
 *   binding  1      PSConstants  (FS b1)         uniform buffer (dynamic)
 *   binding  2      rsx_tex[16]  (FS t0, array)  sampled images x16
 *   bindings 3..17  rsx_tex<u>   (FS t1..t15, one per unit, when the program
 *                                 declares its units singly: any cube map)
 *   bindings 18..21 rsx_vtex<N>  (VS t16..t19)   sampled images
 *   binding  32     rsx_samp[16] (FS s0, array)  samplers x16
 *   bindings 48..51 rsx_vsamp<N> (VS s0..s3)     samplers
 *
 * The vertex stage's samplers reuse s0..s3, so the stages get different
 * sampler shifts (a draw can filter vertex unit N and fragment unit N
 * differently). Vertex inputs and varyings keep glslang's declaration-order
 * locations, which both decompilers keep in step (as the Metal path).
 */
#ifndef PS3RECOMP_RSX_SHADER_SPIRV_H
#define PS3RECOMP_RSX_SHADER_SPIRV_H

#include "ps3emu/ps3types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RSX_SPV_STAGE_VERTEX   0
#define RSX_SPV_STAGE_FRAGMENT 1

#define RSX_SPV_BIND_VS_CB       0u
#define RSX_SPV_BIND_FS_CB       1u
#define RSX_SPV_BIND_FS_TEX      2u    /* the array, and t0 of the single form */
#define RSX_SPV_BIND_VS_TEX      18u   /* + N */
#define RSX_SPV_BIND_FS_SAMP     32u
#define RSX_SPV_BIND_VS_SAMP     48u   /* + N */

/* 1 when built with glslang (PS3RECOMP_HAVE_GLSLANG_SPIRV), 0 for the stub. */
int rsx_hlsl_to_spirv_available(void);

/* Translate one HLSL shader (entry point `main`) to SPIR-V. On success returns
 * the word count and sets *words to a malloc'd array the caller frees; on
 * failure returns 0 with the reason in `log` (may be NULL). Thread-safe. */
u32 rsx_hlsl_to_spirv(const char* hlsl, int stage, u32** words, char* log, u32 log_size);

#ifdef __cplusplus
}
#endif
#endif
