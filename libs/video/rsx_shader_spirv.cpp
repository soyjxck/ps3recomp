/* ps3recomp - HLSL -> SPIR-V with the Vulkan backend's binding plan: see
 * rsx_shader_spirv.h. */
#include "rsx_shader_spirv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PS3RECOMP_HAVE_GLSLANG_SPIRV
#include <mutex>
#include <vector>
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

namespace {

void put_log(char* log, u32 log_size, const char* what, const char* detail)
{
    if (!log || !log_size) return;
    snprintf(log, log_size, "%s%s%s", what ? what : "", (what && detail && *detail) ? ": " : "", detail ? detail : "");
}

std::once_flag s_init_once;
bool s_init_ok;

}  // namespace

extern "C" int rsx_hlsl_to_spirv_available(void) { return 1; }

extern "C" u32 rsx_hlsl_to_spirv(const char* hlsl, int stage, u32** words, char* log, u32 log_size)
{
    *words = nullptr;
    std::call_once(s_init_once, [] { s_init_ok = glslang::InitializeProcess(); });
    if (!s_init_ok) { put_log(log, log_size, "glslang", "InitializeProcess failed"); return 0; }

    const EShLanguage lang = stage == RSX_SPV_STAGE_VERTEX ? EShLangVertex : EShLangFragment;
    const EShMessages msgs = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules | EShMsgReadHlsl);
    glslang::TShader shader(lang);
    const char* strings[1] = { hlsl };
    shader.setStrings(strings, 1);
    shader.setEnvInput(glslang::EShSourceHlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
    shader.setEntryPoint("main");
    /* Explicit registers -> binding = register + the stage's shift, set 0. */
    shader.setAutoMapBindings(false);
    shader.setAutoMapLocations(true);
    shader.setShiftBinding(glslang::EResUbo, 0);
    shader.setShiftBinding(glslang::EResTexture, RSX_SPV_BIND_FS_TEX);   /* t0 -> 2, t16 -> 18 */
    shader.setShiftBinding(glslang::EResSampler,
                           stage == RSX_SPV_STAGE_VERTEX ? RSX_SPV_BIND_VS_SAMP : RSX_SPV_BIND_FS_SAMP);
    if (!shader.parse(GetDefaultResources(), 100, false, msgs)) {
        put_log(log, log_size, "HLSL parse", shader.getInfoLog());
        return 0;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(msgs)) { put_log(log, log_size, "HLSL link", program.getInfoLog()); return 0; }
    if (!program.mapIO()) { put_log(log, log_size, "HLSL io map", program.getInfoLog()); return 0; }

    std::vector<unsigned int> spirv;
    spv::SpvBuildLogger logger;
    glslang::SpvOptions opts;
    opts.disableOptimizer = false;      /* the HLSL legalization passes */
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &logger, &opts);
    if (spirv.empty()) { put_log(log, log_size, "SPIR-V generation", logger.getAllMessages().c_str()); return 0; }
    u32* out = (u32*)malloc(spirv.size() * sizeof(u32));
    if (!out) return 0;
    memcpy(out, spirv.data(), spirv.size() * sizeof(u32));
    *words = out;
    return (u32)spirv.size();
}

#else   /* no glslang */

extern "C" int rsx_hlsl_to_spirv_available(void) { return 0; }
extern "C" u32 rsx_hlsl_to_spirv(const char* hlsl, int stage, u32** words, char* log, u32 log_size)
{
    (void)hlsl; (void)stage;
    *words = nullptr;
    if (log && log_size) snprintf(log, log_size, "built without glslang");
    return 0;
}

#endif
