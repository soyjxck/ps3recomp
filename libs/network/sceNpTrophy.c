/*
 * ps3recomp - sceNpTrophy HLE implementation
 *
 * The title's own trophy pack (PS3_GAME/TROPDIR/<NP comm id>/TROPHY.TRP,
 * trophy_pack.h) gives the trophies' names, grades, hidden flags and icons,
 * read when the game registers its context, in the console's language. The
 * unlocks are kept in gamedata/trophies/<comm id>.json (or under
 * $PS3_GAMEDATA_ROOT), written at each one. A title without a pack still
 * registers: 128 nameless bronze slots, as this file always had.
 *
 * The behaviour follows the firmware as RPCS3 documents it (its sceNpTrophy.cpp):
 * the firmware's error codes, the guest structs' layouts, a hidden trophy's
 * details withheld until it is earned, the platinum earned with the last
 * trophy linked to it (pid) and handed back through platinumId, times as
 * CellRtcTick.
 *
 * The port hears of each unlock (trophy_host.h: its pop-up).
 *
 * Tests: PS3_TROPHY_PREVIEW=<id>[,<id>...] calls the listener for those
 * trophies when the context registers, without recording them (a port's
 * pop-up, seen at boot).
 */

#include "sceNpTrophy.h"
#include "trophy_host.h"
#include "trophy_pack.h"
#include "../system/cellSysutil.h"
#include "ps3emu/guest_call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../../runtime/ppu/ppu_memory.h"   /* vm_base (guest mem) */
#include "ps3emu/endian.h"                   /* ps3_bswap64 -- guest is big-endian */
#include "../guest_struct.h"   /* GUEST_EA, guest_struct_load/store */
/* HLE args arrive as guest effective addresses; translate before deref. */
#define GUEST_PTR(p, T) ((T)((p) ? (void*)(vm_base + (uint32_t)(uintptr_t)(p)) : (void*)0))

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <io.h>
#define mkdir_p(path) _mkdir(path)
#define PATH_SEP '\\'
static SRWLOCK s_lock = SRWLOCK_INIT;
#define TROPHY_LOCK()   AcquireSRWLockExclusive(&s_lock)
#define TROPHY_UNLOCK() ReleaseSRWLockExclusive(&s_lock)
#else
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#define mkdir_p(path) mkdir(path, 0755)
#define PATH_SEP '/'
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
#define TROPHY_LOCK()   pthread_mutex_lock(&s_lock)
#define TROPHY_UNLOCK() pthread_mutex_unlock(&s_lock)
#endif

extern const char* ppu_vfs_root;   /* runtime/ppu/ppu_fs.cpp: the disc's folder */

/* CellRtcTick of the Unix epoch: microseconds from 0001-01-01 to 1970-01-01 */
#define RTC_UNIX_EPOCH 62135596800000000ULL

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

static int s_trophy_initialized = 0;
static char s_storage_path[512] = "gamedata/trophies";
static int  s_storage_path_set = 0;
static char s_pack_dir[1024];
static trophy_unlock_listener s_listener;
static void* s_listener_user;

/* $PS3_GAMEDATA_ROOT (see cellSaveData.c) moves the default next to the saves;
 * an explicit sceNpTrophySetStoragePath still wins. */
static void trophy_adopt_gamedata_root(void)
{
    static int s_done = 0;
    if (s_done || s_storage_path_set) return;
    s_done = 1;
    const char* env = getenv("PS3_GAMEDATA_ROOT");
    if (env && *env)
        snprintf(s_storage_path, sizeof(s_storage_path), "%s/trophies", env);
}

/* Per-context data */
typedef struct {
    int                      in_use;
    int                      registered;
    SceNpCommunicationId     commId;
    char                     name[16];       /* "NPWR04682_00": the pack's folder */
    TrophyPack*              pack;           /* NULL: the title has none */
    TrophyUnlocks            u;
    u32                      total_trophies;
} TrophyContext;

typedef struct {
    int in_use;
    int aborted;
} TrophyHandle;

/* ID zero is the firmware invalid sentinel; retain all four usable slots. */
static TrophyContext s_contexts[SCE_NP_TROPHY_MAX_CONTEXTS + 1];
static TrophyHandle  s_handles[SCE_NP_TROPHY_MAX_HANDLES + 1];

static int ctx_ok(SceNpTrophyContext c) { return c > 0 && c <= SCE_NP_TROPHY_MAX_CONTEXTS && s_contexts[c].in_use; }
static int handle_ok(SceNpTrophyHandle h) { return h > 0 && h <= SCE_NP_TROPHY_MAX_HANDLES && s_handles[h].in_use; }

static void guest_bytes(u32 ea, const void* src, size_t n)
{
    if (ea) memcpy(vm_base + ea, src, n);
}

static const TrophyPackTrophy* ctx_trophy(const TrophyContext* c, s32 id)
{
    return c->pack ? trophy_pack_trophy(c->pack, id) : NULL;
}

/* ---------------------------------------------------------------------------
 * Persistent storage helpers
 * -----------------------------------------------------------------------*/

static void trophy_ensure_dir(void)
{
    char tmp[512];
    char* p;

    trophy_adopt_gamedata_root();
    strncpy(tmp, s_storage_path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    /* Create directories recursively */
    for (p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
            mkdir_p(tmp);
            *p = PATH_SEP;
        }
    }
    mkdir_p(tmp);
}

static void trophy_get_filepath(const TrophyContext* ctx, char* buf, size_t bufsize)
{
    trophy_adopt_gamedata_root();
    trophy_unlocks_path(s_storage_path, ctx->name, buf, bufsize);
}

static void trophy_save(TrophyContext* ctx)
{
    char filepath[1024];
    trophy_ensure_dir();
    trophy_get_filepath(ctx, filepath, sizeof(filepath));
    if (trophy_unlocks_save(filepath, &ctx->u, ctx->name) != 0)
        printf("[sceNpTrophy] WARNING: Could not save trophies to %s\n", filepath);
}

static u32 trophy_count_unlocked(TrophyContext* ctx)
{
    u32 count = 0;
    for (u32 i = 0; i < ctx->total_trophies; i++)
        if (ctx->u.unlocked[i]) count++;
    return count;
}

static void trophy_load(TrophyContext* ctx)
{
    char filepath[1024];
    trophy_get_filepath(ctx, filepath, sizeof(filepath));
    trophy_unlocks_load(filepath, &ctx->u);
    for (u32 i = ctx->total_trophies; i < SCE_NP_TROPHY_MAX_NUM_TROPHIES; i++) ctx->u.unlocked[i] = 0;
    printf("[sceNpTrophy] %u unlocked, from %s\n", (unsigned)trophy_count_unlocked(ctx), filepath);
}

/* The title's pack: the port's folder, the disc's, the working directory's. */
static TrophyPack* trophy_find_pack(const char* name)
{
    const char* dirs[4];
    int n = 0;
    if (s_pack_dir[0]) dirs[n++] = s_pack_dir;
    const char* env = getenv("PS3_VFS_ROOT");
    if (env && *env) dirs[n++] = env;
    if (ppu_vfs_root && *ppu_vfs_root) dirs[n++] = ppu_vfs_root;
    dirs[n++] = ".";
    for (int i = 0; i < n; i++) {
        char path[1024];
        if (trophy_pack_find(dirs[i], name, path, sizeof path) != 0) continue;
        TrophyPack* pk = trophy_pack_load(path, ps3_sysutil_language());
        if (pk) {
            printf("[sceNpTrophy] pack %s: '%s', %d trophies (%s)\n", path, pk->title, pk->count, pk->language_file);
            return pk;
        }
        printf("[sceNpTrophy] WARNING: %s is not a trophy pack\n", path);
    }
    printf("[sceNpTrophy] no trophy pack for %s: nameless trophies\n", name);
    return NULL;
}

/* An unlock for the port, with the icon. */
static void trophy_tell(const TrophyContext* ctx, s32 id, int preview)
{
    trophy_unlock_listener fn = s_listener;
    if (!fn) return;
    TrophyUnlockEvent e;
    memset(&e, 0, sizeof e);
    snprintf(e.np_comm_id, sizeof e.np_comm_id, "%s", ctx->name);
    e.id = id;
    e.preview = preview;
    e.at = id >= 0 && id < SCE_NP_TROPHY_MAX_NUM_TROPHIES && ctx->u.at[id] ? ctx->u.at[id] : (uint64_t)time(NULL);
    const TrophyPackTrophy* t = ctx_trophy(ctx, id);
    uint8_t* png = NULL;
    size_t n = 0;
    if (t) {
        e.grade = t->grade;
        e.hidden = t->hidden;
        snprintf(e.name, sizeof e.name, "%s", t->name);
        snprintf(e.detail, sizeof e.detail, "%s", t->detail);
        if (trophy_pack_icon(ctx->pack, id, &png, &n) == 0) { e.icon_png = png; e.icon_size = n; }
    } else {
        e.grade = 'B';
        snprintf(e.name, sizeof e.name, "Trophy %d", id);
    }
    fn(&e, s_listener_user);
    free(png);
}

/* ---------------------------------------------------------------------------
 * The port's side (trophy_host.h)
 * -----------------------------------------------------------------------*/

void trophy_host_set_listener(trophy_unlock_listener fn, void* user)
{
    TROPHY_LOCK();
    s_listener = fn;
    s_listener_user = user;
    TROPHY_UNLOCK();
}

void trophy_host_set_pack_dir(const char* dir)
{
    snprintf(s_pack_dir, sizeof s_pack_dir, "%s", dir ? dir : "");
}

const char* trophy_host_storage_dir(void)
{
    trophy_adopt_gamedata_root();
    return s_storage_path;
}

int trophy_host_snapshot(TrophyPack** pack, TrophyUnlocks* unlocks)
{
    int rc = -1;
    if (pack) *pack = NULL;
    TROPHY_LOCK();
    for (int i = 1; i <= SCE_NP_TROPHY_MAX_CONTEXTS; i++) {
        TrophyContext* c = &s_contexts[i];
        if (!c->in_use || !c->registered || !c->pack) continue;
        if (pack) {
            *pack = (TrophyPack*)malloc(sizeof(TrophyPack));
            if (*pack) memcpy(*pack, c->pack, sizeof(TrophyPack));
        }
        if (unlocks) *unlocks = c->u;
        rc = 0;
        break;
    }
    TROPHY_UNLOCK();
    return rc;
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

void sceNpTrophySetStoragePath(const char* path)
{
    path = GUEST_PTR(path, const char*);
    if (path) {
        strncpy(s_storage_path, path, sizeof(s_storage_path) - 1);
        s_storage_path[sizeof(s_storage_path) - 1] = '\0';
        s_storage_path_set = 1;
    }
}

s32 sceNpTrophyInit(void* poolPtr, u32 poolSize, u32 containerId, u64 options)
{
    (void)poolPtr; (void)poolSize; (void)containerId; (void)options;

    printf("[sceNpTrophy] Init()\n");

    if (s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_ALREADY_INITIALIZED;

    TROPHY_LOCK();
    for (int i = 0; i <= SCE_NP_TROPHY_MAX_CONTEXTS; i++) trophy_pack_free(s_contexts[i].pack);
    memset(s_contexts, 0, sizeof(s_contexts));
    memset(s_handles, 0, sizeof(s_handles));
    s_trophy_initialized = 1;
    TROPHY_UNLOCK();
    return CELL_OK;
}

s32 sceNpTrophyTerm(void)
{
    printf("[sceNpTrophy] Term()\n");

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    TROPHY_LOCK();
    /* Save all registered contexts */
    for (int i = 1; i <= SCE_NP_TROPHY_MAX_CONTEXTS; i++) {
        if (s_contexts[i].in_use && s_contexts[i].registered)
            trophy_save(&s_contexts[i]);
        trophy_pack_free(s_contexts[i].pack);
    }

    memset(s_contexts, 0, sizeof(s_contexts));
    memset(s_handles, 0, sizeof(s_handles));
    s_trophy_initialized = 0;
    TROPHY_UNLOCK();
    return CELL_OK;
}

s32 sceNpTrophyCreateContext(SceNpTrophyContext* context,
                             const SceNpCommunicationId* commId,
                             const SceNpCommunicationSignature* commSign,
                             u64 options)
{
    (void)commSign; (void)options;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!context || !commId)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    const SceNpCommunicationId* commId_h = GUEST_PTR(commId, const SceNpCommunicationId*);
    if (commId_h->num > 99)
        return SCE_NP_TROPHY_ERROR_INVALID_NP_COMM_ID;

    TROPHY_LOCK();
    for (s32 i = 1; i <= SCE_NP_TROPHY_MAX_CONTEXTS; i++) {
        if (!s_contexts[i].in_use) {
            TrophyContext* c = &s_contexts[i];
            trophy_pack_free(c->pack);
            memset(c, 0, sizeof(TrophyContext));
            c->in_use = 1;
            c->commId = *commId_h;
            char data[SCE_NP_COMMUNICATION_ID_MAX_LENGTH + 1] = {0};
            memcpy(data, commId_h->data, SCE_NP_COMMUNICATION_ID_MAX_LENGTH);
            snprintf(c->name, sizeof c->name, "%s_%02u", data, (unsigned)commId_h->num);
            c->total_trophies = SCE_NP_TROPHY_MAX_NUM_TROPHIES;
            TROPHY_UNLOCK();
            vm_write32(GUEST_EA(context), (u32)i);
            printf("[sceNpTrophy] CreateContext(commId=\"%s\") -> ctx=%d\n", c->name, i);
            return CELL_OK;
        }
    }
    TROPHY_UNLOCK();

    return SCE_NP_TROPHY_ERROR_OUT_OF_MEMORY;
}

s32 sceNpTrophyDestroyContext(SceNpTrophyContext context)
{
    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    TROPHY_LOCK();
    if (!ctx_ok(context)) {
        TROPHY_UNLOCK();
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    }

    if (s_contexts[context].registered)
        trophy_save(&s_contexts[context]);

    trophy_pack_free(s_contexts[context].pack);
    s_contexts[context].pack = NULL;
    s_contexts[context].in_use = 0;
    s_contexts[context].registered = 0;
    TROPHY_UNLOCK();
    printf("[sceNpTrophy] DestroyContext(ctx=%d)\n", context);
    return CELL_OK;
}

s32 sceNpTrophyCreateHandle(SceNpTrophyHandle* handle)
{
    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!handle)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;

    for (s32 i = 1; i <= SCE_NP_TROPHY_MAX_HANDLES; i++) {
        if (!s_handles[i].in_use) {
            s_handles[i].in_use = 1;
            s_handles[i].aborted = 0;
            vm_write32(GUEST_EA(handle), (u32)i);
            printf("[sceNpTrophy] CreateHandle() -> handle=%d\n", i);
            return CELL_OK;
        }
    }

    return SCE_NP_TROPHY_ERROR_OUT_OF_MEMORY;
}

s32 sceNpTrophyDestroyHandle(SceNpTrophyHandle handle)
{
    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!handle_ok(handle))
        return SCE_NP_TROPHY_ERROR_INVALID_HANDLE;

    s_handles[handle].in_use = 0;
    printf("[sceNpTrophy] DestroyHandle(handle=%d)\n", handle);
    return CELL_OK;
}

s32 sceNpTrophyAbortHandle(SceNpTrophyHandle handle)
{
    if (handle <= 0 || handle > SCE_NP_TROPHY_MAX_HANDLES)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    if (!s_handles[handle].in_use)
        return SCE_NP_TROPHY_ERROR_UNKNOWN_HANDLE;
    s_handles[handle].aborted = 1;   /* it cannot be used again */
    return CELL_OK;
}

s32 sceNpTrophyRegisterContext(SceNpTrophyContext context,
                               SceNpTrophyHandle handle,
                               u32 statusCb,
                               u32 arg,
                               u64 options)
{
    (void)options;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!handle_ok(handle))
        return SCE_NP_TROPHY_ERROR_INVALID_HANDLE;
    if (s_handles[handle].aborted)
        return SCE_NP_TROPHY_ERROR_ABORT;

    if (s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_ALREADY_REG;

    if (!statusCb) return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    if (!g_ps3_guest_caller) return SCE_NP_TROPHY_ERROR_UNKNOWN;

    /* The pack, then any previously saved trophy data */
    TrophyPack* pk = s_contexts[context].pack ? NULL : trophy_find_pack(s_contexts[context].name);
    TROPHY_LOCK();
    TrophyContext* c = &s_contexts[context];
    if (pk) {
        c->pack = pk;
        c->total_trophies = (u32)pk->end;
    }
    trophy_load(c);
    c->registered = 1;
    TROPHY_UNLOCK();

    printf("[sceNpTrophy] RegisterContext(ctx=%d, handle=%d, statusCb=0x%08X, "
           "arg=0x%08X) -> loaded saved data\n", context, handle, statusCb, arg);

    /* Use the runner's OPD bridge, not the optional code/TOC scaffold entry
     * point. Registration status is synchronous; its terminal notification is
     * delivered through sysutil polling, after the HLE storage load completes.
     * Intermediate firmware installation progress is not simulated here. */
    g_ps3_guest_caller(statusCb, (u32)context, SCE_NP_TROPHY_STATUS_INSTALLED,
                       0, 0, arg, 0, 0, 0);
    if (!s_trophy_initialized) return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;
    if (!s_contexts[context].in_use) return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    if (!s_handles[handle].in_use) return SCE_NP_TROPHY_ERROR_INVALID_HANDLE;
    const u64 completion[8] = {(u32)context, SCE_NP_TROPHY_STATUS_PROCESSING_COMPLETE,
                               0, 0, arg, 0, 0, 0};
    if (cellSysutilQueueGuestCallbackArgs(statusCb, completion) != CELL_OK) {
        s_contexts[context].registered = 0;
        return SCE_NP_TROPHY_ERROR_OUT_OF_MEMORY;
    }

    const char* pv = getenv("PS3_TROPHY_PREVIEW");
    if (pv && *pv && s_contexts[context].pack) {
        for (const char* p = pv; *p;) {   /* "5" or "0,5,27" */
            trophy_tell(&s_contexts[context], atoi(p), 1);
            const char* comma = strchr(p, ',');
            if (!comma) break;
            p = comma + 1;
        }
    }
    return CELL_OK;
}

s32 sceNpTrophyGetRequiredDiskSpace(SceNpTrophyContext context,
                                    SceNpTrophyHandle handle,
                                    u64* reqSpace, u64 options)
{
    (void)handle; (void)options;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!reqSpace)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    reqSpace = GUEST_PTR(reqSpace, u64*);

    /* Typical trophy pack size. The guest is big-endian and reads this u64
     * back in BE order, so the value MUST be byte-swapped on write -- otherwise
     * a host-LE 0x100000 is read as 0x0010000000000000 (2^44). LBP feeds this
     * "required disk space" straight into its save-data free-space accounting
     * (sub_379EC0: errB = budget - usedKB - reqSpaceKB); a garbage 2^44 makes
     * errB underflow negative -> cellGame GameData check returns ERROR-B ->
     * boot bails. (Same LE-write class as the cellUserInfo fix.) */
    *reqSpace = ps3_bswap64((u64)(1024 * 1024)); /* 1 MB, big-endian */
    printf("[sceNpTrophy] GetRequiredDiskSpace(ctx=%d) -> 1MB\n", context);
    return CELL_OK;
}

s32 sceNpTrophySetSoundLevel(SceNpTrophyContext context, SceNpTrophyHandle handle,
                             u32 level, u64 options)
{
    if (level > 100 || level < 20)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    if (options > 0)
        return SCE_NP_TROPHY_ERROR_NOT_SUPPORTED;
    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;
    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    if (!handle_ok(handle))
        return SCE_NP_TROPHY_ERROR_INVALID_HANDLE;
    return CELL_OK;
}

s32 sceNpTrophyGetGameInfo(SceNpTrophyContext context,
                           SceNpTrophyHandle handle,
                           SceNpTrophyGameDetails* details,
                           SceNpTrophyGameData* data)
{
    (void)handle;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;
    if (!details && !data)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    const u32 details_ea = GUEST_EA(details), data_ea = GUEST_EA(data);
    TrophyContext* c = &s_contexts[context];

    /* These structs live in guest memory, so every integer field is stored
     * big-endian. Native stores left numTrophies=32 reading back as 0x20000000,
     * and Drakengard 3 sized a buffer from it: malloc(0xA0000380). */
    u32 num[5] = {0}, got[5] = {0};   /* all, P, G, S, B */
    char title[SCE_NP_TROPHY_GAME_TITLE_MAX_SIZE] = {0};
    char desc[SCE_NP_TROPHY_GAME_DESC_MAX_SIZE] = {0};
    TROPHY_LOCK();
    if (c->pack) {
        for (int id = 0; id < c->pack->end; id++) {
            const TrophyPackTrophy* t = trophy_pack_trophy(c->pack, id);
            if (!t) continue;
            const int g = trophy_grade_number(t->grade);
            num[0]++;
            if (g >= 1 && g <= 4) num[g]++;
            if (c->u.unlocked[id]) {
                got[0]++;
                if (g >= 1 && g <= 4) got[g]++;
            }
        }
        snprintf(title, sizeof title, "%s", c->pack->title);
        snprintf(desc, sizeof desc, "%s", c->pack->detail);
    } else {
        /* no pack: the default set this file always answered */
        num[0] = 32; num[1] = 1; num[2] = 2; num[3] = 8; num[4] = 21;
        got[0] = got[4] = trophy_count_unlocked(c);
        snprintf(title, sizeof title, "PS3 Game");
        snprintf(desc, sizeof desc, "Trophy set");
    }
    TROPHY_UNLOCK();

    if (details_ea) {
        memset(vm_base + details_ea, 0, sizeof(SceNpTrophyGameDetails));
        for (int i = 0; i < 5; i++) vm_write32(details_ea + 4 * i, num[i]);
        guest_bytes(details_ea + 20, title, sizeof title - 1);
        guest_bytes(details_ea + 20 + SCE_NP_TROPHY_GAME_TITLE_MAX_SIZE, desc, sizeof desc - 1);
    }
    if (data_ea)
        for (int i = 0; i < 5; i++) vm_write32(data_ea + 4 * i, got[i]);

    printf("[sceNpTrophy] GetGameInfo(ctx=%d): %u of %u\n", context, got[0], num[0]);
    return CELL_OK;
}

s32 sceNpTrophyGetTrophyInfo(SceNpTrophyContext context,
                             SceNpTrophyHandle handle,
                             SceNpTrophyId trophyId,
                             SceNpTrophyDetails* details,
                             SceNpTrophyData* data)
{
    (void)handle;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;
    if (!details && !data)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;

    TrophyContext* c = &s_contexts[context];
    if (trophyId < 0 || (u32)trophyId >= c->total_trophies)
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    const TrophyPackTrophy* t = ctx_trophy(c, trophyId);
    if (c->pack && !t)
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    const int unlocked = c->u.unlocked[trophyId];
    if (t && t->hidden && !unlocked)
        return SCE_NP_TROPHY_ERROR_HIDDEN;   /* as the firmware: earned first */

    const u32 details_ea = GUEST_EA(details), data_ea = GUEST_EA(data);
    if (details_ea) {
        memset(vm_base + details_ea, 0, sizeof(SceNpTrophyDetails));
        vm_write32(details_ea + 0, (u32)trophyId);
        vm_write32(details_ea + 4, t ? (u32)trophy_grade_number(t->grade) : SCE_NP_TROPHY_GRADE_BRONZE);
        char name[SCE_NP_TROPHY_NAME_MAX_SIZE] = {0};
        char desc[SCE_NP_TROPHY_DESC_MAX_SIZE] = {0};
        if (t) {
            snprintf(name, sizeof name, "%s", t->name);
            snprintf(desc, sizeof desc, "%s", t->detail);
        } else {
            snprintf(name, sizeof name, "Trophy %d", trophyId);
            snprintf(desc, sizeof desc, "Trophy #%d description", trophyId);
        }
        guest_bytes(details_ea + 8, name, sizeof name - 1);
        guest_bytes(details_ea + 8 + SCE_NP_TROPHY_NAME_MAX_SIZE, desc, sizeof desc - 1);
        vm_write8(details_ea + 8 + SCE_NP_TROPHY_NAME_MAX_SIZE + SCE_NP_TROPHY_DESC_MAX_SIZE, t ? t->hidden : 0);
    }

    if (data_ea) {
        memset(vm_base + data_ea, 0, sizeof(SceNpTrophyData));
        vm_write64(data_ea + 0, unlocked ? c->u.at[trophyId] * 1000000ULL + RTC_UNIX_EPOCH : 0);
        vm_write32(data_ea + 8, (u32)trophyId);
        vm_write8(data_ea + 12, (u8)(unlocked ? 1 : 0));
    }

    return CELL_OK;
}

/* The platinum `id` completes, or -1: every trophy linked to its platinum is in. */
static s32 trophy_platinum_earned(const TrophyContext* c, s32 id)
{
    const TrophyPackTrophy* t = ctx_trophy(c, id);
    if (!t || t->pid < 0) return -1;
    const s32 plat = t->pid;
    const TrophyPackTrophy* p = ctx_trophy(c, plat);
    if (!p || p->grade != 'P' || c->u.unlocked[plat]) return -1;
    for (int i = 0; i < c->pack->end; i++) {
        const TrophyPackTrophy* o = trophy_pack_trophy(c->pack, i);
        if (o && i != plat && o->pid == plat && !c->u.unlocked[i]) return -1;
    }
    return plat;
}

s32 sceNpTrophyUnlockTrophy(SceNpTrophyContext context,
                            SceNpTrophyHandle handle,
                            SceNpTrophyId trophyId,
                            SceNpTrophyId* platinumId)
{
    (void)handle;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;

    TrophyContext* c = &s_contexts[context];
    if (trophyId < 0 || (u32)trophyId >= c->total_trophies)
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    const TrophyPackTrophy* t = ctx_trophy(c, trophyId);
    if (c->pack && !t)
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    if (t && t->grade == 'P')
        return SCE_NP_TROPHY_ERROR_CANNOT_UNLOCK_PLATINUM;

    TROPHY_LOCK();
    if (c->u.unlocked[trophyId]) {
        TROPHY_UNLOCK();
        return SCE_NP_TROPHY_ERROR_ALREADY_UNLOCKED;
    }

    /* Unlock the trophy, and the platinum it completes */
    const uint64_t now = (uint64_t)time(NULL);
    c->u.unlocked[trophyId] = 1;
    c->u.at[trophyId] = now;
    const s32 plat = trophy_platinum_earned(c, trophyId);
    if (plat >= 0) {
        c->u.unlocked[plat] = 1;
        c->u.at[plat] = now;
    }
    trophy_save(c);
    TROPHY_UNLOCK();

    printf("[sceNpTrophy] TROPHY UNLOCKED: #%d %s '%s' (%s)%s\n", trophyId, t ? trophy_grade_name(t->grade) : "",
           t ? t->name : "", c->name, plat >= 0 ? " -- and the platinum" : "");

    if (platinumId)
        vm_write32(GUEST_EA(platinumId), (u32)(plat >= 0 ? plat : SCE_NP_TROPHY_INVALID_TROPHY_ID));

    trophy_tell(c, trophyId, 0);
    if (plat >= 0) trophy_tell(c, plat, 0);
    return CELL_OK;
}

s32 sceNpTrophyGetTrophyUnlockState(SceNpTrophyContext context,
                                    SceNpTrophyHandle handle,
                                    SceNpTrophyFlagArray* flags,
                                    u32* count)
{
    (void)handle;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;

    if (!flags || !count)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;

    u32 flags_ea = (u32)(uintptr_t)flags;
    for (u32 o = 0; o < sizeof(SceNpTrophyFlagArray); o += 4)
        vm_write32(flags_ea + o, 0);

    for (u32 i = 0; i < s_contexts[context].total_trophies; i++) {
        if (s_contexts[context].u.unlocked[i]) {
            u32 word_ea = flags_ea + (i / 32) * 4;
            vm_write32(word_ea, vm_read32(word_ea) | (1u << (i % 32)));
        }
    }

    vm_write32((u32)(uintptr_t)count, (u32)s_contexts[context].total_trophies);
    return CELL_OK;
}

s32 sceNpTrophyGetGameProgress(SceNpTrophyContext context,
                               SceNpTrophyHandle handle,
                               s32* percentage)
{
    (void)handle;

    if (!percentage)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;

    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;

    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;

    if (!s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;

    u32 total = s_contexts[context].total_trophies;
    u32 unlocked = trophy_count_unlocked(&s_contexts[context]);

    /* rounded to the nearest, as the firmware */
    s32 pct = total > 0 ? (s32)((unlocked * 100 + total / 2) / total) : 0;
    vm_write32(GUEST_EA(percentage), (u32)pct);

    printf("[sceNpTrophy] GetGameProgress(ctx=%d) -> %d%%\n",
           context, pct);
    return CELL_OK;
}

/* An icon into the guest's buffer: its size always; the bytes if they fit. */
static s32 trophy_icon_out(const TrophyContext* c, int id, void* buffer, u32* size)
{
    uint8_t* png = NULL;
    size_t n = 0;
    if (!c->pack || trophy_pack_icon(c->pack, id, &png, &n) != 0)
        return SCE_NP_TROPHY_ERROR_UNKNOWN_FILE;
    const u32 size_ea = GUEST_EA(size), buf_ea = GUEST_EA(buffer);
    if (buf_ea && vm_read32(size_ea) >= (u32)n)
        guest_bytes(buf_ea, png, n);
    vm_write32(size_ea, (u32)n);
    free(png);
    return CELL_OK;
}

s32 sceNpTrophyGetGameIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
                           void* buffer, u32* size)
{
    (void)handle;
    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;
    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    if (!size)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    if (!s_contexts[context].registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;
    return trophy_icon_out(&s_contexts[context], -1, buffer, size);
}

s32 sceNpTrophyGetTrophyIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
                             SceNpTrophyId trophyId, void* buffer, u32* size)
{
    (void)handle;
    if (!s_trophy_initialized)
        return SCE_NP_TROPHY_ERROR_NOT_INITIALIZED;
    if (!ctx_ok(context))
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    if (!size)
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    TrophyContext* c = &s_contexts[context];
    if (!c->registered)
        return SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED;
    if (trophyId < 0 || (u32)trophyId >= c->total_trophies)
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    if (!c->u.unlocked[trophyId]) {
        const TrophyPackTrophy* t = ctx_trophy(c, trophyId);
        return t && t->hidden ? SCE_NP_TROPHY_ERROR_HIDDEN : SCE_NP_TROPHY_ERROR_LOCKED;
    }
    return trophy_icon_out(c, trophyId, buffer, size);
}
