/*
 * ps3recomp - the running title's trophies, for the port's own UI
 *
 * sceNpTrophy (the HLE) reads the title's trophy pack when the game registers
 * its context, keeps the unlocks in gamedata/trophies, and tells the port
 * about each new trophy through the listener below, so a port can show its
 * own notification (the console's "You have earned a trophy." pop-up is
 * drawn by the system software, which a recompilation does not have).
 */

#ifndef PS3RECOMP_TROPHY_HOST_H
#define PS3RECOMP_TROPHY_HOST_H

#include <stddef.h>
#include <stdint.h>
#include "trophy_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrophyUnlockEvent {
    char           np_comm_id[16];
    int            id;
    char           grade;            /* 'P' 'G' 'S' 'B' */
    int            hidden;
    char           name[TROPHY_NAME_SIZE];       /* in the console's language */
    char           detail[TROPHY_DETAIL_SIZE];
    uint64_t       at;               /* Unix seconds */
    int            preview;          /* a test's (PS3_TROPHY_PREVIEW): not recorded */
    const uint8_t* icon_png;         /* TROPnnn.PNG; valid during the call only */
    size_t         icon_size;
} TrophyUnlockEvent;

typedef void (*trophy_unlock_listener)(const TrophyUnlockEvent* e, void* user);

/* Called on the game's thread that unlocked it, once per trophy (a platinum
 * the unlock completes comes as a second call). Copy what is kept. */
void trophy_host_set_listener(trophy_unlock_listener fn, void* user);

/* Where the trophy packs are looked for when the game registers: a folder
 * holding PS3_GAME/TROPDIR (or TROPDIR). Default: the disc, through the VFS
 * (/dev_bdvd/PS3_GAME/TROPDIR, then $PS3_VFS_ROOT). */
void trophy_host_set_pack_dir(const char* dir);

/* The folder the unlocks are saved in ("gamedata/trophies", or
 * $PS3_GAMEDATA_ROOT/trophies). */
const char* trophy_host_storage_dir(void);

/* A copy of the registered title's pack and unlocks (for an in-game list).
 * 0 if one is registered; free the pack with trophy_pack_free. */
int trophy_host_snapshot(TrophyPack** pack, TrophyUnlocks* unlocks);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_TROPHY_HOST_H */
