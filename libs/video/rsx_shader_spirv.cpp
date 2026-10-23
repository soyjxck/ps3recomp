/* ps3recomp - HLSL -> SPIR-V with the Vulkan backend's binding plan: see
 * rsx_shader_spirv.h.
 *
 * Translations are cached. glslang takes 20-150 ms per shader, on the thread
 * that asks for it (Twisted Metal: 236 translations in its first minute, 74
 * distinct). The cache key is the stage plus the whole HLSL text, so a hit
 * returns exactly the words that text translated to. With
 * PS3RECOMP_SPIRV_CACHE=<existing directory> the cache is also kept on disk,
 * one file per shader, and the next run starts warm. A file whose key,
 * glslang version or translator build differs from the request is a miss,
 * and is rewritten. Failed translations are not cached. */
#include "rsx_shader_spirv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PS3RECOMP_HAVE_GLSLANG_SPIRV
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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

/* One translation through glslang. False with the reason in `log`. */
bool translate(const char* hlsl, int stage, std::vector<unsigned int>& spirv, char* log, u32 log_size)
{
    std::call_once(s_init_once, [] { s_init_ok = glslang::InitializeProcess(); });
    if (!s_init_ok) { put_log(log, log_size, "glslang", "InitializeProcess failed"); return false; }

    const EShLanguage lang = stage == RSX_SPV_STAGE_VERTEX ? EShLangVertex : EShLangFragment;
    const EShMessages msgs = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules | EShMsgReadHlsl);
    glslang::TShader shader(lang);
    const char* strings[1] = { hlsl };
    shader.setStrings(strings, 1);
    shader.setEnvInput(glslang::EShSourceHlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
    /* SPIR-V FragCoord.w is 1/w (FXC's SV_POSITION.w is w): tells the FP
     * decompiler's _rsx_inv_w helper which one it gets. */
    shader.setPreamble("#define RSX_FRAGCOORD_W_IS_RCP 1\n");
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
        return false;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(msgs)) { put_log(log, log_size, "HLSL link", program.getInfoLog()); return false; }
    if (!program.mapIO()) { put_log(log, log_size, "HLSL io map", program.getInfoLog()); return false; }

    spv::SpvBuildLogger logger;
    glslang::SpvOptions opts;
    opts.disableOptimizer = false;      /* the HLSL legalization passes */
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &logger, &opts);
    if (spirv.empty()) { put_log(log, log_size, "SPIR-V generation", logger.getAllMessages().c_str()); return false; }
    return true;
}

/* ---- translation cache ------------------------------------------------- */

const u32 kCacheFormat = 1;
const u32 kCacheMagic = 0x56505352u;   /* "RSPV" */
/* This file's compile time stands for translate()'s options and the binding
 * plan in rsx_shader_spirv.h: when either changes, this file is rebuilt, and
 * the files an older build wrote become misses. */
const char kTranslatorBuild[24] = __DATE__ " " __TIME__;

struct Cache {
    std::mutex lock;
    std::unordered_map<std::string, std::vector<unsigned int>> map;
    std::string dir;
    bool dir_read = false;
    RsxSpirvCacheStats stats = {};
};

Cache& cache()
{
    static Cache c;
    return c;
}

const std::string& cache_dir(Cache& c)
{
    if (!c.dir_read) {
        const char* e = getenv("PS3RECOMP_SPIRV_CACHE");
        c.dir = (e && *e) ? e : "";
        c.dir_read = true;
    }
    return c.dir;
}

/* The key is the stage, then the HLSL text. */
std::string cache_key(const char* hlsl, int stage)
{
    std::string key(1, (char)('0' + stage));
    key += hlsl;
    return key;
}

std::string cache_path(const std::string& dir, const std::string& key)
{
    unsigned long long h = 0xcbf29ce484222325ull;       /* FNV-1a 64 */
    for (unsigned char ch : key) { h ^= ch; h *= 0x100000001b3ull; }
    char name[40];
    snprintf(name, sizeof name, "/%016llx.spv", h);
    return dir + name;
}

/* Header: magic, format, glslang major/minor/patch, translator build (six
 * words), key bytes, word count; then the key and the words. */
const int kHeaderWords = 13;

void cache_header(u32 hdr[kHeaderWords], const std::string& key, size_t nwords)
{
    const glslang::Version v = glslang::GetVersion();
    hdr[0] = kCacheMagic; hdr[1] = kCacheFormat;
    hdr[2] = (u32)v.major; hdr[3] = (u32)v.minor; hdr[4] = (u32)v.patch;
    memcpy(&hdr[5], kTranslatorBuild, sizeof kTranslatorBuild);
    hdr[11] = (u32)key.size(); hdr[12] = (u32)nwords;
}

bool cache_load(const std::string& path, const std::string& key,
                std::vector<unsigned int>& spirv)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    u32 want[kHeaderWords], hdr[kHeaderWords];
    cache_header(want, key, 0);
    bool ok = fread(hdr, sizeof hdr, 1, f) == 1 &&
              memcmp(hdr, want, (kHeaderWords - 1) * sizeof(u32)) == 0 &&
              hdr[kHeaderWords - 1] != 0 && hdr[kHeaderWords - 1] <= (1u << 24);
    if (ok) {
        std::string stored(key.size(), '\0');
        ok = fread(&stored[0], 1, stored.size(), f) == stored.size() && stored == key;
    }
    if (ok) {
        spirv.resize(hdr[kHeaderWords - 1]);
        ok = fread(spirv.data(), sizeof(u32), spirv.size(), f) == spirv.size();
    }
    fclose(f);
    if (!ok) spirv.clear();
    return ok;
}

/* Written under a temporary name and renamed, so a reader never sees half a file. */
void cache_store(const std::string& path, const std::string& key,
                 const std::vector<unsigned int>& spirv)
{
    char suffix[48];
    snprintf(suffix, sizeof suffix, ".%zx.tmp", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    const std::string tmp = path + suffix;
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    u32 hdr[kHeaderWords];
    cache_header(hdr, key, spirv.size());
    bool ok = fwrite(hdr, sizeof hdr, 1, f) == 1 &&
              fwrite(key.data(), 1, key.size(), f) == key.size() &&
              fwrite(spirv.data(), sizeof(u32), spirv.size(), f) == spirv.size();
    ok = (fclose(f) == 0) && ok;
#ifdef _WIN32
    if (ok) remove(path.c_str());      /* rename does not replace on Windows */
#endif
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) remove(tmp.c_str());
}

/* The caller's copy: a malloc'd array it frees. */
u32 copy_out(const std::vector<unsigned int>& spirv, u32** words)
{
    u32* out = (u32*)malloc(spirv.size() * sizeof(u32));
    if (!out) return 0;
    memcpy(out, spirv.data(), spirv.size() * sizeof(u32));
    *words = out;
    return (u32)spirv.size();
}

}  // namespace

extern "C" int rsx_hlsl_to_spirv_available(void) { return 1; }

extern "C" u32 rsx_hlsl_to_spirv(const char* hlsl, int stage, u32** words, char* log, u32 log_size)
{
    *words = nullptr;
    if (!hlsl) { put_log(log, log_size, "bad arguments", nullptr); return 0; }

    Cache& c = cache();
    const std::string key = cache_key(hlsl, stage);
    std::string dir;
    {
        std::lock_guard<std::mutex> guard(c.lock);
        auto it = c.map.find(key);
        if (it != c.map.end()) {
            c.stats.memory_hits++;
            return copy_out(it->second, words);
        }
        dir = cache_dir(c);
    }
    /* A miss is read or translated outside the lock, so the engines' build
     * threads still translate side by side. Two threads that miss on the same
     * text both translate it; the first entry stays. */
    std::vector<unsigned int> spirv;
    const std::string path = dir.empty() ? std::string() : cache_path(dir, key);
    const bool from_disk = !dir.empty() && cache_load(path, key, spirv);
    if (!from_disk) {
        if (!translate(hlsl, stage, spirv, log, log_size)) return 0;
        if (!dir.empty()) cache_store(path, key, spirv);
    }
    {
        std::lock_guard<std::mutex> guard(c.lock);
        if (from_disk) c.stats.disk_hits++;
        else c.stats.translations++;
        c.map.emplace(key, spirv);
    }
    return copy_out(spirv, words);
}

extern "C" void rsx_hlsl_to_spirv_cache_stats(RsxSpirvCacheStats* out)
{
    Cache& c = cache();
    std::lock_guard<std::mutex> guard(c.lock);
    if (out) *out = c.stats;
}

extern "C" void rsx_hlsl_to_spirv_cache_reset(void)
{
    Cache& c = cache();
    std::lock_guard<std::mutex> guard(c.lock);
    c.map.clear();
    c.dir_read = false;
    c.stats = {};
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

extern "C" void rsx_hlsl_to_spirv_cache_stats(RsxSpirvCacheStats* out)
{
    if (out) memset(out, 0, sizeof *out);
}

extern "C" void rsx_hlsl_to_spirv_cache_reset(void) {}

#endif
