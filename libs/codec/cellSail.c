/*
 * ps3recomp - cellSail HLE implementation
 *
 * Stub. Player lifecycle and state management work.
 * No actual media playback occurs — games typically show
 * a black screen for cutscenes, which is acceptable for
 * initial recompilation testing.
 */

#include "ps3emu/guest_call.h"
#include "cellSail.h"
#include "../../runtime/ppu/ppu_memory.h"   /* vm_read32 / vm_write32 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>   /* getenv -- Clang errors on the implicit declaration MSVC allows */
#include <time.h>     /* nanosleep: cellSailFutureGet off Windows */
#include "../guest_struct.h"   /* GUEST_EA, guest_struct_load/store */

/* Internal state */

static int s_initialized = 0;

typedef struct {
    int in_use;
    s32 state;
    CellSailPlayerCallback callback;
    void* callbackArg;
} SailPlayer;

static SailPlayer s_players[CELL_SAIL_PLAYER_MAX];

/* Players registered by cellSailPlayerInitialize2, keyed by the guest struct EA
 * the title passed as `pSelf`. Declared here because the adapter setters below
 * need it -- they take that same pointer, not an index into s_players[]. */
static struct { u32 self_ea, allocator_ea, callback_ea, arg_ea; int in_use; int state; }
    s_sail_players[CELL_SAIL_PLAYER_MAX];
/* SAIL_EOS_ON_POLL: when each player's deferred end of stream is due (ms, 0 = none). */
static unsigned long long s_sail_eos_at[CELL_SAIL_PLAYER_MAX];

/* A CellSailPlayer is a GUEST POINTER, not a small integer handle.
 *
 * Every player entry point below used to validate its first argument as an
 * index (`handle >= CELL_SAIL_PLAYER_MAX`) into the handle-keyed s_players[]
 * table, which rejects every real call: a guest pointer is always larger than
 * the table. cellSailPlayerInitialize2 has always registered the player by its
 * `pSelf` EA, so the two halves of this file disagreed about what a handle is.
 *
 * That is not a harmless error return. Tokyo Jungle sets its sound adapter
 * during audio init; on failure it abandons the path, closes a resource it
 * never opened ("sgxResClose unknown ID (0)"), terminates its SGX sound system
 * and then blocks forever joining an audio thread that will not exit.
 *
 * One lookup, used by all of them, so the disagreement cannot come back. */
static int sail_player_slot(u32 self_ea)
{
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++)
        if (s_sail_players[i].in_use && s_sail_players[i].self_ea == self_ea)
            return i;
    return -1;
}

/* ---------------------------------------------------------------------------
 * Host playback (SAIL_MOVIES, default on; SAIL_MOVIES=0 turns it off).
 *
 * The file is decoded on the host (sail_movie.h: Media Foundation on Windows)
 * and handed to the title the way the firmware's player does it, through the
 * adapters the title set up:
 *  - graphics: the title's adapter callbacks allocate the frame buffers
 *    (pAlloc(pArg, size, num, ppFrame) -> number allocated); the title polls
 *    cellSailGraphicsAdapterGetFrame each frame, which converts the frame due
 *    on the wall clock into the next buffer in the coding the title asked for
 *    in SetPreferredFormat (RGBA, ARGB or YUV420 planar), and reads the size
 *    back with GetFormat;
 *  - sound: the title polls cellSailSoundAdapterGetFrame once per audio period
 *    for 256 samples of float stereo at 48 kHz and mixes them itself
 *    (cellAudioAdd2chData), so the audio runs on the cellAudio clock.
 * The stream ends (STOP_TRANSITION, OPENED), from the title's next poll, once
 * the movie's length has passed. If the host cannot open the file the player
 * keeps the old behaviour: a stream that has already ended.
 *
 * Tested with a publisher logo movie (AVI: DivX 5 at 1280x720, 29.97 fps,
 * MP3 48 kHz stereo) played this way.
 * -----------------------------------------------------------------------*/
#include "sail_movie.h"
extern int cellfs_translate_path(const char* ps3_path, char* host_buf, size_t buf_size);
extern unsigned long long ps3_ms_now(void);
static const char* sail_desc_uri(u32 player);   /* below, with the descriptors */

enum { SAIL_CODING_ARGB = 0, SAIL_CODING_RGBA = 1, SAIL_CODING_YUV420 = 3 };
#define SAIL_SESSION_ID   0x5A110001u   /* any value the title has not cached */
#define SAIL_SND_SAMPLES  256u
#define SAIL_SND_TAIL     (SAIL_SND_SAMPLES * 8u * 4u)   /* up to 8 ch of float */

/* fn[]: the title's CellSailGraphicsAdapterFuncs (makeup, cleanup, formatChanged,
 * alloc, free), copied at Initialize as libsail does. The title passes the table
 * on its stack; read back at the end of the movie, the
 * free slot had been overwritten with 0, the frames (3 x 3.6 MB of the title's
 * heap) were never returned, and the character models no longer fitted: missing
 * or garbled for the rest of the game. */
typedef struct { u32 self, funcs, arg; int coding, w, h; u32 alpha;
                 u32 frames[3]; int nframes; u32 frame_bytes; u32 fn[5]; } SailGfxAdapter;
static SailGfxAdapter s_gfx[CELL_SAIL_PLAYER_MAX];

static struct {
    sail_movie* m;
    int slot;             /* player slot playing, -1 none */
    int running;
    unsigned long long t0_ms;
    SailGfxAdapter* gfx;
    int cur;              /* buffer holding the frame last converted, -1 none */
    u32 tag;
    unsigned long long pts90;
    u32 snd_buf;          /* guest: one block of BE float stereo */
    unsigned snd_pos;     /* frames of audio handed out */
} s_mv = { 0, -1, 0, 0, 0, -1, 0, 0, 0, 0 };

static int sail_movies_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("SAIL_MOVIES"); on = (e && *e == '0') ? 0 : 1; }
    return on;
}

static SailGfxAdapter* sail_gfx(u32 self)
{
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) if (s_gfx[i].self == self) return &s_gfx[i];
    return 0;
}
static SailGfxAdapter* sail_gfx_new(u32 self)
{
    SailGfxAdapter* g = sail_gfx(self);
    for (int i = 0; !g && i < CELL_SAIL_PLAYER_MAX; i++) if (!s_gfx[i].self) g = &s_gfx[i];
    if (g) { memset(g, 0, sizeof(*g)); g->self = self; g->coding = SAIL_CODING_RGBA; g->alpha = 255; }
    return g;
}

static void sail_movie_stop(void)
{
    s_mv.running = 0;
    if (s_mv.m) { sail_movie_close(s_mv.m); s_mv.m = 0; }
    s_mv.slot = -1; s_mv.cur = -1; s_mv.snd_buf = 0; s_mv.snd_pos = 0;
}

/* Frame buffers from the title's own allocator, sized for the coding it asked
 * for plus room for one block of sound behind the first frame. */
static int sail_alloc_frames(SailGfxAdapter* g, int w, int h)
{
    if (!g || !g->funcs || !g_ps3_guest_caller) return 0;
    const u32 pAlloc = g->fn[3];
    if (!pAlloc) return 0;
    g->w = w; g->h = h;
    g->frame_bytes = (g->coding == SAIL_CODING_YUV420) ? (u32)(w * h * 3 / 2) : (u32)(w * h * 4);
    const u32 size = (g->frame_bytes + SAIL_SND_TAIL + 127u) & ~127u;
    const u32 pp = g->self + 0x40;          /* scratch inside the adapter's private storage */
    vm_write32(pp, 0); vm_write32(pp + 4, 0); vm_write32(pp + 8, 0);
    g_ps3_guest_caller(pAlloc, g->arg, size, 3, pp, 0, 0, 0, 0);
    g->nframes = 0;
    for (int i = 0; i < 3; i++) { g->frames[i] = vm_read32(pp + 4u * i); if (g->frames[i]) g->nframes = i + 1; }
    printf("[cellSail] frames: %d x %u bytes (coding %d, %dx%d) at 0x%08X\n",
           g->nframes, size, g->coding, w, h, g->frames[0]);
    return g->nframes;
}

static void sail_free_frames(SailGfxAdapter* g)
{
    if (!g || !g->nframes || !g->funcs || !g_ps3_guest_caller) return;
    const u32 pFree = g->fn[4];
    const u32 pp = g->self + 0x40;
    for (int i = 0; i < 3; i++) vm_write32(pp + 4u * i, g->frames[i]);
    if (pFree) g_ps3_guest_caller(pFree, g->arg, (u64)g->nframes, pp, 0, 0, 0, 0, 0);
    g->nframes = 0;
    memset(g->frames, 0, sizeof(g->frames));
}

/* Open the descriptor's file for player slot sp (at OpenStream). */
static void sail_movie_open_for(int sp, const char* uri)
{
    sail_movie_stop();
    if (!sail_movies_on() || !uri || !*uri) return;
    const char* path = uri;
    if (!strncmp(path, "x-cell-fs://", 12)) path += 12;
    char host[1024];
    if (cellfs_translate_path(path, host, sizeof(host)) != 0) snprintf(host, sizeof(host), "%s", path);
    sail_movie* m = sail_movie_open(host);
    if (!m) { printf("[cellSail] cannot decode %s -- the stream ends at once\n", host); return; }
    s_mv.m = m; s_mv.slot = sp; s_mv.cur = -1; s_mv.tag = 0; s_mv.snd_pos = 0; s_mv.snd_buf = 0;
    /* One movie at a time: the last graphics adapter set up is the one. */
    s_mv.gfx = 0;
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) if (s_gfx[i].self && s_gfx[i].funcs) s_mv.gfx = &s_gfx[i];
    if (s_mv.gfx && sail_movie_width(m)) {
        if (!s_mv.gfx->nframes) sail_alloc_frames(s_mv.gfx, sail_movie_width(m), sail_movie_height(m));
        if (s_mv.gfx->nframes) s_mv.snd_buf = s_mv.gfx->frames[0] + s_mv.gfx->frame_bytes;
    }
}

static inline u8 sail_clamp8(int v) { return (u8)(v < 0 ? 0 : v > 255 ? 255 : v); }

/* NV12 (tight) -> the title's coding, into guest memory at dst. */
static void sail_convert(const SailGfxAdapter* g, const unsigned char* nv12, u32 dst)
{
    const int w = g->w, h = g->h;
    u8* out = (u8*)vm_base + dst;
    const unsigned char* uv = nv12 + (size_t)w * h;
    if (g->coding == SAIL_CODING_YUV420) {
        memcpy(out, nv12, (size_t)w * h);
        u8* U = out + (size_t)w * h; u8* V = U + (size_t)(w / 2) * (h / 2);
        for (int y = 0; y < h / 2; y++)
            for (int x = 0; x < w / 2; x++) {
                U[y * (w / 2) + x] = uv[y * w + 2 * x];
                V[y * (w / 2) + x] = uv[y * w + 2 * x + 1];
            }
        return;
    }
    /* BT.601, limited range (what MPEG-4 Part 2 and DivX encode). */
    const u8 a = (u8)g->alpha;
    for (int y = 0; y < h; y++) {
        const unsigned char* Y = nv12 + (size_t)y * w;
        const unsigned char* C = uv + (size_t)(y / 2) * w;
        u8* o = out + (size_t)y * w * 4;
        for (int x = 0; x < w; x++) {
            const int c = 298 * (Y[x] - 16), d = C[x & ~1] - 128, e = C[(x & ~1) + 1] - 128;
            const u8 r = sail_clamp8((c + 409 * e + 128) >> 8);
            const u8 gg = sail_clamp8((c - 100 * d - 208 * e + 128) >> 8);
            const u8 b = sail_clamp8((c + 516 * d + 128) >> 8);
            if (g->coding == SAIL_CODING_ARGB) { o[0] = a; o[1] = r; o[2] = gg; o[3] = b; }
            else                               { o[0] = r; o[1] = gg; o[2] = b; o[3] = a; }
            o += 4;
        }
    }
}

/* Lifecycle */

s32 cellSailInit(void)
{
    printf("[cellSail] Init()\n");
    if (s_initialized)
        return (s32)CELL_SAIL_ERROR_ALREADY_INITIALIZED;
    memset(s_players, 0, sizeof(s_players));
    s_initialized = 1;
    return CELL_OK;
}

s32 cellSailTerm(void)
{
    printf("[cellSail] Term()\n");
    s_initialized = 0;
    return CELL_OK;
}

/* Player management */

s32 cellSailPlayerCreate(const CellSailPlayerAttribute* attr,
                           const CellSailPlayerResource* resource,
                           CellSailPlayerCallback callback,
                           void* callbackArg,
                           CellSailPlayerHandle* handle)
{
    (void)attr; (void)resource;
    printf("[cellSail] PlayerCreate()\n");

    if (!s_initialized) return (s32)CELL_SAIL_ERROR_NOT_INITIALIZED;
    if (!handle) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) {
        if (!s_players[i].in_use) {
            memset(&s_players[i], 0, sizeof(SailPlayer));
            s_players[i].in_use = 1;
            s_players[i].state = CELL_SAIL_PLAYER_STATE_INITIALIZED;
            s_players[i].callback = callback;
            s_players[i].callbackArg = callbackArg;
            vm_write32((u32)(uintptr_t)handle, (u32)i);   /* guest out-param */
            return CELL_OK;
        }
    }
    return (s32)CELL_SAIL_ERROR_OUT_OF_MEMORY;
}

s32 cellSailPlayerDestroy(CellSailPlayerHandle handle)
{
    printf("[cellSail] PlayerDestroy(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].in_use = 0;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_CLOSED;
    return CELL_OK;
}

/* CellSailEvent is {be32 major; be32 minor} -- one 64-bit register, major in
 * the high half. Async player calls report completion as major=2
 * (CALL_COMPLETED) with minor = the CellSailPlayerCall id that finished; a
 * title that drives SAIL asynchronously blocks until its handler sees that.
 * We run every call synchronously, so notify right after each one. */
#define SAIL_EV_CALL_COMPLETED 2
enum { SAIL_CALL_BOOT = 1, SAIL_CALL_OPEN_STREAM = 2, SAIL_CALL_CLOSE_STREAM = 3,
       SAIL_CALL_START = 10, SAIL_CALL_STOP = 11 };

static void sail_notify(int slot, u32 major, u32 minor, u64 arg0, u64 arg1)
{
    if (slot < 0 || !g_ps3_guest_caller) return;
    const u32 cb = s_sail_players[slot].callback_ea;
    if (!cb) return;
    g_ps3_guest_caller(cb, s_sail_players[slot].arg_ea,
                       ((u64)major << 32) | minor, arg0, arg1, 0, 0, 0, 0);
}

/* The player also reports every state it passes through: major=3
 * (PLAYER_STATE_CHANGED), minor 0, arg0 = the new state in the SDK's
 * numbering -- not the CELL_SAIL_PLAYER_STATE_* values in cellSail.h, which
 * are this file's own bookkeeping. Midway Arcade Origins maps arg0 through a
 * nine-entry table (CLOSED=2, OPENED=4, RUNNING=6, ...) and opens nothing until
 * boot has reported CLOSED; without these events it boots the player and then
 * waits on a black screen for good. */
#define SAIL_EV_STATE_CHANGED 3
enum { SAIL_ST_BOOT_TRANSITION = 1, SAIL_ST_CLOSED = 2, SAIL_ST_OPEN_TRANSITION = 3,
       SAIL_ST_OPENED = 4, SAIL_ST_START_TRANSITION = 5, SAIL_ST_RUNNING = 6,
       SAIL_ST_STOP_TRANSITION = 7, SAIL_ST_CLOSE_TRANSITION = 8 };

/* One synchronous call: transition state, completion, settled state.
 * The call token belongs to completion arg1, not either state notification. */
static void sail_call(int slot, u32 call, u32 via, u32 to, u64 userParam)
{
    sail_notify(slot, SAIL_EV_STATE_CHANGED, 0, via, 0);
    sail_notify(slot, SAIL_EV_CALL_COMPLETED, call, 0, userParam);
    sail_notify(slot, SAIL_EV_STATE_CHANGED, 0, to, 0);
}

s32 cellSailPlayerBoot(CellSailPlayerHandle handle, u64 userParam)
{
    printf("[cellSail] PlayerBoot(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_RUNNING;
    sail_call(_sp, SAIL_CALL_BOOT, SAIL_ST_BOOT_TRANSITION, SAIL_ST_CLOSED, userParam);
    return CELL_OK;
}

s32 cellSailPlayerOpenStream(CellSailPlayerHandle handle, const char* path)
{
    /* path is a GUEST address; printing it as a host char* faulted here the
     * same way it did in CreateDescriptor. Copy it out of guest memory. */
    char path_host[512];
    path_host[0] = '\0';
    if (path) {
        u32 ea = (u32)(uintptr_t)path;
        size_t n = 0;
        while (n < sizeof(path_host) - 1) {
            char c = (char)vm_read8(ea + (u32)n);
            if (!c) break;
            path_host[n++] = c;
        }
        path_host[n] = '\0';
    }
    printf("[cellSail] PlayerOpenStream(%u, \"%s\") - stub\n", handle,
           path ? path_host : "null");
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    /* The host decodes the file the player's descriptor names (the title
     * passes "" here and the URI in CreateDescriptor). */
    sail_movie_open_for(_sp, (path && path_host[0]) ? path_host : sail_desc_uri((u32)handle));
    sail_call(_sp, SAIL_CALL_OPEN_STREAM, SAIL_ST_OPEN_TRANSITION, SAIL_ST_OPENED, 0);
    return CELL_OK;
}

s32 cellSailPlayerCloseStream(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    sail_call(_sp, SAIL_CALL_CLOSE_STREAM, SAIL_ST_CLOSE_TRANSITION, SAIL_ST_CLOSED, 0);
    return CELL_OK;
}

s32 cellSailPlayerStart(CellSailPlayerHandle handle)
{
    printf("[cellSail] PlayerStart(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_RUNNING;
    /* Immediately signal finished since we don't play anything */
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_FINISHED;
    sail_call(_sp, SAIL_CALL_START, SAIL_ST_START_TRANSITION, SAIL_ST_RUNNING, 0);
    /* A movie the host is playing ends when its length has passed, reported
     * from the title's adapter polls like SAIL_EOS_ON_POLL below. */
    if (s_mv.m && s_mv.slot == _sp) {
        s_mv.t0_ms = ps3_ms_now();
        s_mv.running = 1;
        s_sail_eos_at[_sp] = s_mv.t0_ms + (unsigned long long)(sail_movie_duration_us(s_mv.m) / 1000) + 50;
        printf("[cellSail] playing (%.2f s)\n", sail_movie_duration_us(s_mv.m) / 1e6);
        return CELL_OK;
    }
    /* SAIL_EOS_ON_POLL=1: report the end of the stream later, from the title's
     * next adapter poll (GetFrame / UpdateAvSync) at least 100 ms on, instead
     * of inside this call. A title that stores RUNNING into its own
     * copy of the player state after Start/SetRepeatMode return and then waits
     * for OPENED: an OPENED sent here was overwritten and the logo never
     * ended. Off by default: titles that wait for it inside Start keep the
     * old behaviour. */
    static int s_eos_on_poll = -1;
    if (s_eos_on_poll < 0) s_eos_on_poll = getenv("SAIL_EOS_ON_POLL") ? 1 : 0;
    if (s_eos_on_poll) {
        extern unsigned long long ps3_ms_now(void);
        s_sail_eos_at[_sp] = ps3_ms_now() + 100;
        return CELL_OK;
    }
    /* Nothing decodes the stream, so it has already ended: a finished stream
     * drops the player back to OPENED, which is what a title waiting out a
     * movie (Midway's logo AVIs) watches for. */
    sail_notify(_sp, SAIL_EV_STATE_CHANGED, 0, SAIL_ST_STOP_TRANSITION, 0);
    sail_notify(_sp, SAIL_EV_STATE_CHANGED, 0, SAIL_ST_OPENED, 0);
    return CELL_OK;
}

/* Deliver a deferred end of stream (SAIL_EOS_ON_POLL) once it is due. Called
 * from the adapter polls a title makes every frame while a movie plays. */
static void sail_eos_poll(void)
{
    extern unsigned long long ps3_ms_now(void);
    const unsigned long long now = ps3_ms_now();
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) {
        if (!s_sail_eos_at[i] || now < s_sail_eos_at[i] || !s_sail_players[i].in_use) continue;
        s_sail_eos_at[i] = 0;
        printf("[cellSail] end of stream (deferred) for player %d\n", i);
        if (s_mv.slot == i) s_mv.running = 0;
        sail_notify(i, SAIL_EV_STATE_CHANGED, 0, SAIL_ST_STOP_TRANSITION, 0);
        sail_notify(i, SAIL_EV_STATE_CHANGED, 0, SAIL_ST_OPENED, 0);
    }
}

s32 cellSailPlayerStop(CellSailPlayerHandle handle)
{
    printf("[cellSail] PlayerStop(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_FINISHED;
    if (s_mv.slot == _sp) { s_mv.running = 0; s_sail_eos_at[_sp] = 0; }
    sail_call(_sp, SAIL_CALL_STOP, SAIL_ST_STOP_TRANSITION, SAIL_ST_OPENED, 0);
    return CELL_OK;
}

s32 cellSailPlayerPause(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_PAUSE;
    return CELL_OK;
}

s32 cellSailPlayerGetState(CellSailPlayerHandle handle, s32* state)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!state) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)state, (u32)s_sail_players[_sp].state);
    return CELL_OK;
}

s32 cellSailPlayerGetStreamNum(CellSailPlayerHandle handle, u32* streamNum)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!streamNum) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)streamNum, 0);   /* no streams in stub */
    return CELL_OK;
}

s32 cellSailPlayerGetStreamInfo(CellSailPlayerHandle handle, u32 streamIndex,
                                  CellSailStreamInfo* info)
{
    (void)streamIndex;
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!info) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return (s32)CELL_SAIL_ERROR_NOT_FOUND;
}

/* r3 is a CellSailPlayer* -- a GUEST POINTER, the same `pSelf` that
 * cellSailPlayerInitialize2 registered -- not a small integer handle. Validating
 * it as an index (`handle >= CELL_SAIL_PLAYER_MAX`) rejects every real call,
 * because a guest pointer is always larger than the table size.
 *
 * That is not a harmless error return: Tokyo Jungle sets its sound adapter
 * during audio init, and on failure abandons the whole path -- it closes a
 * resource it never opened ("sgxResClose unknown ID (0)"), terminates its SGX
 * sound system, and then blocks forever joining an audio thread that is not
 * going to exit. Look the player up the way Initialize2 stored it. */
s32 cellSailPlayerSetSoundAdapter(CellSailPlayerHandle handle, u32 index, void* adapter)
{
    (void)index; (void)adapter;
    u32 self_ea = (u32)handle;
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++)
        if (s_sail_players[i].in_use && s_sail_players[i].self_ea == self_ea)
            return CELL_OK;
    return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
}

/* r3 is a CellSailPlayer* -- a GUEST POINTER, the same `pSelf` that
 * cellSailPlayerInitialize2 registered -- not a small integer handle. Validating
 * it as an index (`handle >= CELL_SAIL_PLAYER_MAX`) rejects every real call,
 * because a guest pointer is always larger than the table size.
 *
 * That is not a harmless error return: Tokyo Jungle sets its sound adapter
 * during audio init, and on failure abandons the whole path -- it closes a
 * resource it never opened ("sgxResClose unknown ID (0)"), terminates its SGX
 * sound system, and then blocks forever joining an audio thread that is not
 * going to exit. Look the player up the way Initialize2 stored it. */
s32 cellSailPlayerSetGraphicsAdapter(CellSailPlayerHandle handle, u32 index, void* adapter)
{
    (void)index; (void)adapter;
    u32 self_ea = (u32)handle;
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++)
        if (s_sail_players[i].in_use && s_sail_players[i].self_ea == self_ea)
            return CELL_OK;
    return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
}

s32 cellSailPlayerCancel(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_FINISHED;
    return CELL_OK;
}

s32 cellSailPlayerIsPaused(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return 0;
    return (s_sail_players[_sp].state == CELL_SAIL_PLAYER_STATE_PAUSE) ? 1 : 0;
}

s32 cellSailPlayerSetRepeatMode(CellSailPlayerHandle handle, s32 repeatMode, void* command)
{
    (void)command;
    printf("[cellSail] SetRepeatMode(%u, %d)\n", handle, repeatMode);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    /* Real ABI returns the (now-set) repeat mode in r3, not an error code. */
    return repeatMode;
}

/* ---------------------------------------------------------------------------
 * Descriptor management
 * -----------------------------------------------------------------------*/

#define MAX_DESCRIPTORS 16

typedef struct {
    int in_use;
    s32 streamType;
    CellSailPlayerHandle player;
    char uri[512];
} SailDescriptor;

static SailDescriptor s_descs[MAX_DESCRIPTORS];

static const char* sail_desc_uri(u32 player)
{
    for (int i = MAX_DESCRIPTORS - 1; i >= 0; i--)
        if (s_descs[i].in_use && (u32)s_descs[i].player == player && s_descs[i].uri[0]) return s_descs[i].uri;
    return 0;
}

s32 cellSailPlayerCreateDescriptor(CellSailPlayerHandle handle,
                                     s32 streamType, void* mediaInfo,
                                     char* uri,
                                     CellSailDescriptorHandle* desc)
{
    (void)mediaInfo;

    /* uri is a GUEST address. Printing it as a host char* faulted the moment
     * the title opened a movie -- the log line stopped mid-token at
     * 'uri="' and the process died with 0xC0000005. Copy it out of guest
     * memory first. The descriptor has always had a uri[512] field; nothing
     * ever filled it, so the URI was discarded even when it did not crash. */
    char uri_host[512];
    uri_host[0] = '\0';
    if (uri) {
        u32 ea = (u32)(uintptr_t)uri;
        size_t n = 0;
        while (n < sizeof(uri_host) - 1) {
            char c = (char)vm_read8(ea + (u32)n);
            if (!c) break;
            uri_host[n++] = c;
        }
        uri_host[n] = '\0';
    }

    printf("[cellSail] CreateDescriptor(player=%u, type=%d, uri=\"%s\")\n", handle,
           streamType, uri ? uri_host : "null");
    if (!desc) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    for (int i = 0; i < MAX_DESCRIPTORS; i++) {
        if (!s_descs[i].in_use) {
            memset(&s_descs[i], 0, sizeof(SailDescriptor));
            s_descs[i].in_use = 1;
            s_descs[i].streamType = streamType;
            s_descs[i].player = handle;
            snprintf(s_descs[i].uri, sizeof(s_descs[i].uri), "%s", uri_host);
            vm_write32((u32)(uintptr_t)desc, (u32)i);   /* guest out-param */
            return CELL_OK;
        }
    }
    return (s32)CELL_SAIL_ERROR_OUT_OF_MEMORY;
}

s32 cellSailPlayerDestroyDescriptor(CellSailPlayerHandle handle,
                                      CellSailDescriptorHandle desc)
{
    (void)handle;
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_descs[desc].in_use = 0;
    return CELL_OK;
}

s32 cellSailPlayerAddDescriptor(CellSailPlayerHandle handle,
                                  CellSailDescriptorHandle desc)
{
    (void)handle;
    printf("[cellSail] AddDescriptor(player=%u, desc=%u)\n", handle, desc);
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailPlayerRemoveDescriptor(CellSailPlayerHandle handle,
                                     CellSailDescriptorHandle desc)
{
    (void)handle;
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorCreateDatabase(CellSailDescriptorHandle desc,
                                       void* dbAddr, u32 dbSize, u64 arg)
{
    (void)dbAddr; (void)dbSize; (void)arg;
    printf("[cellSail] DescriptorCreateDatabase(desc=%u)\n", desc);
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorDestroyDatabase(CellSailDescriptorHandle desc)
{
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorGetStreamType(CellSailDescriptorHandle desc, s32* type)
{
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use || !type)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)type, (u32)s_descs[desc].streamType);
    return CELL_OK;
}

s32 cellSailDescriptorSetAutoSelection(CellSailDescriptorHandle desc, s32 enable)
{
    (void)enable;
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorGetUri(CellSailDescriptorHandle desc, char* uri, u32 maxLen)
{
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use || !uri)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    strncpy(GUEST_PTR(uri, char*), s_descs[desc].uri, maxLen - 1);
    uri[maxLen - 1] = '\0';
    return CELL_OK;
}

/* Memory allocator */

/* cellSailPlayerInitialize2(pSelf, pAllocator, pCallback, pUserParam,
 *                          pAttribute, pResource)  -- NID 0x23654375
 *
 * This was unimplemented, so it took the generic "return success and touch
 * nothing" path. That is the worst answer for an initialiser: the guest
 * believes its player struct is set up and reads whatever happened to be in
 * that memory. Tokyo Jungle then dereferenced one of those stale words and
 * faulted reading guest 0x10F000B4, well past the end of main memory.
 *
 * The struct fields are not documented here, and guessing offsets to write
 * would just trade one wrong value for another. What IS safe and sufficient is
 * to ZERO it: every pointer the guest reads out is then NULL, which callers
 * check, instead of a garbage address that faults. The handles we were given
 * are kept on our side, keyed by the struct EA, so no guest layout is assumed.
 */
#define SAIL_PLAYER_CLEAR   0x200u   /* well clear of the caller's next object */
#define SAIL_ADAPTER_CLEAR  0x100u


static void sail_zero_guest(u32 ea, u32 n)
{
    for (u32 i = 0; i < n; i += 4) vm_write32(ea + i, 0);
}

s32 cellSailPlayerInitialize2(u32 pSelf, u32 pAllocator, u32 pCallback,
                              u32 pUserParam, u32 pAttribute, u32 pResource)
{
    (void)pAttribute; (void)pResource;
    printf("[cellSail] PlayerInitialize2(self=0x%08X alloc=0x%08X cb=0x%08X)\n",
           pSelf, pAllocator, pCallback);
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    /* PS3_NO_SAIL=1: report that the player cannot be created. On hardware
     * this call spawns the SAIL player's own worker threads, and a title that
     * then waits for one of them to signal readiness will wait forever against
     * a stub that spawns nothing. Failing cleanly lets a title take its
     * "no video" path instead of deadlocking during setup. */
    if (getenv("PS3_NO_SAIL")) {
        printf("[cellSail] PS3_NO_SAIL -- reporting player creation failure\n");
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    }

    sail_zero_guest(pSelf, SAIL_PLAYER_CLEAR);

    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) {
        if (!s_sail_players[i].in_use) {
            s_sail_players[i].in_use       = 1;
            s_sail_players[i].self_ea      = pSelf;
            s_sail_players[i].allocator_ea = pAllocator;
            s_sail_players[i].callback_ea  = pCallback;
            s_sail_players[i].arg_ea       = pUserParam;
            break;
        }
    }
    return CELL_OK;
}

/* cellSailSoundAdapterInitialize(pSelf, pCallbacks, pArg) -- NID 0x3D0D3B72.
 * Same reasoning: zero it rather than leave the guest reading stale memory. */
s32 cellSailSoundAdapterInitialize(u32 pSelf, u32 pCallbacks, u32 pArg)
{
    printf("[cellSail] SoundAdapterInitialize(self=0x%08X funcs=0x%08X)\n",
           pSelf, pCallbacks);
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    sail_zero_guest(pSelf, SAIL_ADAPTER_CLEAR);
    return CELL_OK;
}

/* cellSailGraphicsAdapterInitialize(pSelf, pCallbacks, pArg)
 *
 * The video-side twin of cellSailSoundAdapterInitialize, and it was missing --
 * so the import returned whatever was in r3 and the guest read an adapter it
 * believed was initialised. Same treatment as the sound adapter: zero the guest
 * struct so every pointer the title reads back is NULL (which callers check)
 * rather than a stale address that faults or is silently followed. */
s32 cellSailGraphicsAdapterInitialize(u32 pSelf, u32 pCallbacks, u32 pArg)
{
    printf("[cellSail] GraphicsAdapterInitialize(self=0x%08X funcs=0x%08X arg=0x%08X)\n",
           pSelf, pCallbacks, pArg);
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    sail_zero_guest(pSelf, SAIL_ADAPTER_CLEAR);
    /* Keep the callbacks: host playback allocates its frames through them
     * (CellSailGraphicsAdapterFuncs: makeup, cleanup, formatChanged, alloc, free). */
    SailGfxAdapter* g = sail_gfx_new(pSelf);
    if (g) {
        g->funcs = pCallbacks; g->arg = pArg;
        for (int i = 0; i < 5; i++) g->fn[i] = pCallbacks ? vm_read32(pCallbacks + 4u * i) : 0;
    }
    return CELL_OK;
}

/* cellSailGraphicsAdapterSetPreferredFormat(pSelf, pFormat)
 *
 * CellSailVideoFormat: coding s8 @0, width s16 @4, height s16 @6, pitch @8,
 * alpha @0xC. Host playback converts its frames to this coding. */
s32 cellSailGraphicsAdapterSetPreferredFormat(u32 pSelf, u32 pFormat)
{
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    SailGfxAdapter* g = sail_gfx(pSelf);
    if (g && pFormat) {
        const int coding = (s8)vm_read8(pFormat);
        if (coding == SAIL_CODING_ARGB || coding == SAIL_CODING_RGBA || coding == SAIL_CODING_YUV420)
            g->coding = coding;
        const u32 alpha = vm_read32(pFormat + 0xC);
        g->alpha = alpha <= 255 ? alpha : 255;
        printf("[cellSail] GraphicsAdapterSetPreferredFormat(self=0x%08X coding=%d %dx%d alpha=%u)\n",
               pSelf, coding, (s16)vm_read16(pFormat + 4), (s16)vm_read16(pFormat + 6), alpha);
    }
    return CELL_OK;
}

s32 cellSailMemAllocatorInitialize(CellSailMemAllocator* allocator,
                                     CellSailMemAllocatorFuncs* pFuncs)
{
    /* Both parameters are GUEST addresses, and both structs live in GUEST
     * memory with 32-bit big-endian pointers. Dereferencing them as host
     * structs was wrong twice over: the address was never translated, and the
     * host struct has 64-bit fields, so even the offsets did not line up. The
     * values stored here are guest OPD addresses that SAIL calls back into. */
    u32 alloc_ea = (u32)(uintptr_t)allocator;
    u32 funcs_ea = (u32)(uintptr_t)pFuncs;
    printf("[cellSail] MemAllocatorInitialize(allocator=0x%08X funcs=0x%08X)\n",
           alloc_ea, funcs_ea);
    if (!alloc_ea) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (funcs_ea) {
        vm_write32(alloc_ea + 0, vm_read32(funcs_ea + 0));   /* pAlloc */
        vm_write32(alloc_ea + 4, vm_read32(funcs_ea + 4));   /* pFree  */
    }
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Futures, adapters and the rest of the player API a title drives a movie
 * with, as a player that decodes nothing (no frames ever arrive, the stream
 * has already ended). Layouts and semantics follow RPCS3 cellSail.h/.cpp.
 *
 * CellSailFuture: mutex u32 @0, cond u32 @4, flags u32 @8, result s32 @0xC,
 * userParam u64 @0x10. A title passes one as a call's userParam, its player
 * callback stores the event in +0x10 and calls cellSailFutureSet on
 * PLAYER_CALL_COMPLETED, and it waits in cellSailFutureGet. A title that
 * plays its logo movie this way polled the missing adapter calls for ever.
 * -----------------------------------------------------------------------*/
#define SAIL_ERR_EMPTY 0x80610705u   /* SDK: CELL_SAIL_ERROR_EMPTY, no frame/sample yet */

s32 cellSailFutureInitialize(u32 pSelf)
{
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    sail_zero_guest(pSelf, 0x18);
    return CELL_OK;
}

s32 cellSailFutureFinalize(u32 pSelf)
{
    return pSelf ? CELL_OK : (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
}

/* cellSailFutureReset(pSelf, wait) -- NID 0x0C4CB439 (registered by NID in the
 * port: the toolkit's database lacks the name). */
s32 cellSailFutureReset(u32 pSelf, u32 wait)
{
    (void)wait;
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32(pSelf + 0x8, 0);
    vm_write32(pSelf + 0xC, 0);
    return CELL_OK;
}

s32 cellSailFutureSet(u32 pSelf, s32 result)
{
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32(pSelf + 0xC, (u32)result);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    vm_write32(pSelf + 0x8, 1);
    return CELL_OK;
}

/* cellSailFutureGet(pSelf, timeout_us, pResult): every player call here runs
 * synchronously and completes its future before it returns, so the future is
 * normally set already; wait a little for one set on another thread. */
s32 cellSailFutureGet(u32 pSelf, u64 timeout, u32 pResult)
{
    if (!pSelf) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    unsigned waited_ms = 0;
    const unsigned cap_ms = (timeout && timeout / 1000 < 100) ? (unsigned)(timeout / 1000) : 100;
    while (!vm_read32(pSelf + 0x8) && waited_ms < cap_ms) {
#ifdef _WIN32
        extern __declspec(dllimport) void __stdcall Sleep(unsigned long);
        Sleep(1);
#else
        struct timespec ts = {0, 1000000}; nanosleep(&ts, 0);
#endif
        waited_ms++;
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (pResult) vm_write32(pResult, vm_read32(pSelf + 0xC));
    return CELL_OK;
}

/* CellSailGraphicsFrameInfo / CellSailSoundFrameInfo: buffer @0, sessionId @4,
 * tag @8, status @0xC, pts u64 @0x10 (90 kHz). */
static void sail_write_frame_info(u32 pInfo, u32 buffer, u32 tag, unsigned long long pts90)
{
    if (!pInfo) return;
    vm_write32(pInfo + 0x0, buffer);
    vm_write32(pInfo + 0x4, SAIL_SESSION_ID);
    vm_write32(pInfo + 0x8, tag);
    vm_write32(pInfo + 0xC, 0);
    vm_write64(pInfo + 0x10, pts90);
}

/* The frame due now, converted into the next of the title's buffers; the same
 * frame again until the next one is due. Returns the frame's size in bytes
 * (titles may take >= 256 as success), or EMPTY with nothing playing. */
s32 cellSailGraphicsAdapterGetFrame(u32 pSelf, u32 pInfo)
{
    sail_eos_poll();
    SailGfxAdapter* g = s_mv.gfx;
    if (!s_mv.running || !s_mv.m || !g || g->self != pSelf || !g->nframes) return (s32)SAIL_ERR_EMPTY;
    const long long t_us = (long long)(ps3_ms_now() - s_mv.t0_ms) * 1000;
    const unsigned char* nv12 = 0;
    if (sail_movie_video_at(s_mv.m, t_us, &nv12) == 1 && nv12) {
        s_mv.cur = (s_mv.cur + 1) % g->nframes;
        sail_convert(g, nv12, g->frames[s_mv.cur]);
        s_mv.tag++;
        s_mv.pts90 = (unsigned long long)t_us * 9 / 100;
    }
    if (s_mv.cur < 0) return (s32)SAIL_ERR_EMPTY;
    sail_write_frame_info(pInfo, g->frames[s_mv.cur], s_mv.tag, s_mv.pts90);
    return (s32)g->frame_bytes;
}

/* The next `samples` frames of the soundtrack (float stereo, 48 kHz, big-endian
 * in guest memory). Returns the number of samples, EMPTY past the end. */
s32 cellSailSoundAdapterGetFrame(u32 pSelf, u32 samples, u32 pInfo)
{
    (void)pSelf;
    sail_eos_poll();
    unsigned total = 0;
    const float* pcm = s_mv.m ? sail_movie_audio(s_mv.m, &total) : 0;
    if (!s_mv.running || !pcm || !s_mv.snd_buf || s_mv.snd_pos >= total) return (s32)SAIL_ERR_EMPTY;
    if (samples == 0 || samples > SAIL_SND_SAMPLES * 4u) samples = SAIL_SND_SAMPLES;
    if (samples > SAIL_SND_TAIL / 8u) samples = SAIL_SND_TAIL / 8u;
    u32* out = (u32*)((u8*)vm_base + s_mv.snd_buf);
    for (u32 i = 0; i < samples * 2u; i++) {
        const unsigned k = s_mv.snd_pos * 2u + i;
        float f = k < total * 2u ? pcm[k] : 0.0f;
        u32 w; memcpy(&w, &f, 4);
        out[i] = (w >> 24) | ((w >> 8) & 0xFF00u) | ((w << 8) & 0xFF0000u) | (w << 24);
    }
    sail_write_frame_info(pInfo, s_mv.snd_buf, s_mv.snd_pos / samples,
                          (unsigned long long)s_mv.snd_pos * 90000ull / 48000ull);
    s_mv.snd_pos += samples;
    return (s32)samples;
}

/* pts (90 kHz) -> time position (microseconds). */
s32 cellSailGraphicsAdapterPtsToTimePosition(u32 pSelf, u64 pts, u32 pTime)
{
    (void)pSelf;
    if (pTime) vm_write64(pTime, pts * 100u / 9u);
    return CELL_OK;
}
s32 cellSailSoundAdapterPtsToTimePosition(u32 pSelf, u64 pts, u32 pTime)
{
    (void)pSelf;
    if (pTime) vm_write64(pTime, pts * 100u / 9u);
    return CELL_OK;
}

s32 cellSailGraphicsAdapterUpdateAvSync(u32 pSelf, u64 pts) { (void)pSelf; (void)pts; sail_eos_poll(); return CELL_OK; }
s32 cellSailSoundAdapterUpdateAvSync(u32 pSelf, u64 pts)    { (void)pSelf; (void)pts; return CELL_OK; }
/* CellSailVideoFormat: coding s8 @0, scan @1, bitsPerColor @2, frameRate @3,
 * width s16 @4, height s16 @6, pitch @8, alpha @0xC, colorMatrix @0x10,
 * colorRange @0x11. */
s32 cellSailGraphicsAdapterGetFormat(u32 pSelf, u32 pFormat)
{
    SailGfxAdapter* g = sail_gfx(pSelf);
    if (!pFormat || !g || !g->w) return CELL_OK;
    for (u32 i = 0; i < 0x20; i += 4) vm_write32(pFormat + i, 0);
    vm_write8(pFormat + 0, (u8)g->coding);
    vm_write8(pFormat + 2, 8);
    vm_write16(pFormat + 4, (u16)g->w);
    vm_write16(pFormat + 6, (u16)g->h);
    vm_write32(pFormat + 8, g->coding == SAIL_CODING_YUV420 ? (u32)g->w : (u32)g->w * 4u);
    vm_write32(pFormat + 0xC, g->alpha);
    vm_write8(pFormat + 0x11, 1);
    return CELL_OK;
}
/* CellSailAudioFormat: coding s8 @0, chNum s8 @1, sampleNum s16 @2, fs @4,
 * chLayout @8, reserved0 @0xC (-1), reserved1 s64 @0x10 (-1). */
s32 cellSailSoundAdapterGetFormat(u32 pSelf, u32 pFormat)
{
    (void)pSelf;
    if (!pFormat) return CELL_OK;
    vm_write8(pFormat + 0, 1);                       /* float LPCM */
    vm_write8(pFormat + 1, 2);
    vm_write16(pFormat + 2, (u16)SAIL_SND_SAMPLES);
    vm_write32(pFormat + 4, 48000);
    vm_write32(pFormat + 8, 0);
    vm_write32(pFormat + 0xC, 0xFFFFFFFFu);
    vm_write64(pFormat + 0x10, ~0ull);
    return CELL_OK;
}
/* The title's free callback gets the frames back here, and the movie closes. */
s32 cellSailGraphicsAdapterFinalize(u32 pSelf)
{
    SailGfxAdapter* g = sail_gfx(pSelf);
    if (g) {
        if (s_mv.gfx == g) { sail_movie_stop(); s_mv.gfx = 0; }
        sail_free_frames(g);
        memset(g, 0, sizeof(*g));
    }
    return CELL_OK;
}
s32 cellSailSoundAdapterFinalize(u32 pSelf)    { (void)pSelf; return CELL_OK; }
/* cellSailSoundAdapterSetPreferredFormat -- NID 0x1C9D5E5A (port-registered). */
s32 cellSailSoundAdapterSetPreferredFormat(u32 pSelf, u32 pFormat) { (void)pSelf; (void)pFormat; return CELL_OK; }
/* cellSailAviMovieGetMovieInfo -- NID 0x6E83F5C0 (port-registered). Zero, or
 * with a movie open: scale @0x1C, rate @0x20 and length @0x24 in frames, which
 * a title turns into the length in ms (length * scale * 1000 / rate). */
s32 cellSailAviMovieGetMovieInfo(u32 pSelf, u32 pInfo)
{
    (void)pSelf;
    if (!pInfo || !s_mv.m) return CELL_OK;
    const long long us = sail_movie_duration_us(s_mv.m);
    vm_write32(pInfo + 0x14, (u32)sail_movie_width(s_mv.m));
    vm_write32(pInfo + 0x18, (u32)sail_movie_height(s_mv.m));
    vm_write32(pInfo + 0x1C, 1000);
    vm_write32(pInfo + 0x20, 1000000);
    vm_write32(pInfo + 0x24, (u32)(us / 1000));     /* in 1 ms "frames" */
    return CELL_OK;
}

s32 cellSailPlayerSetParameter(u32 pSelf, s32 type, u64 param0, u64 param1)
{ (void)pSelf; (void)type; (void)param0; (void)param1; return CELL_OK; }

s32 cellSailPlayerFinalize(u32 pSelf)
{
    const int sp = sail_player_slot(pSelf);
    if (sp >= 0) {
        if (s_mv.slot == sp) sail_movie_stop();
        s_sail_eos_at[sp] = 0;
        s_sail_players[sp].in_use = 0;
    }
    return CELL_OK;
}

/* PAUSE_STATE_CHANGED (event 7), arg0 = the paused flag. */
s32 cellSailPlayerSetPaused(u32 pSelf, s32 paused)
{
    const int sp = sail_player_slot(pSelf);
    if (sp < 0) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    sail_notify(sp, 7, 0, (u64)(paused ? 1 : 0), 0);
    return CELL_OK;
}
