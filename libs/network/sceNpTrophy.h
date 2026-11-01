/*
 * ps3recomp - sceNpTrophy HLE
 *
 * Trophy/achievement system: create contexts, unlock trophies, query progress.
 */

#ifndef PS3RECOMP_SCE_NP_TROPHY_H
#define PS3RECOMP_SCE_NP_TROPHY_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"
#include "sceNp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Error codes (the firmware's: RPCS3's sceNpTrophy.h lists the same)
 * -----------------------------------------------------------------------*/
#define SCE_NP_TROPHY_ERROR_ALREADY_INITIALIZED     0x80022901
#define SCE_NP_TROPHY_ERROR_NOT_INITIALIZED         0x80022902
#define SCE_NP_TROPHY_ERROR_NOT_SUPPORTED           0x80022903
#define SCE_NP_TROPHY_ERROR_CONTEXT_NOT_REGISTERED  0x80022904
#define SCE_NP_TROPHY_ERROR_OUT_OF_MEMORY           0x80022905
#define SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT        0x80022906
#define SCE_NP_TROPHY_ERROR_EXCEEDS_MAX             0x80022907
#define SCE_NP_TROPHY_ERROR_INSUFFICIENT            0x80022909
#define SCE_NP_TROPHY_ERROR_UNKNOWN_CONTEXT         0x8002290A
#define SCE_NP_TROPHY_ERROR_INVALID_FORMAT          0x8002290B
#define SCE_NP_TROPHY_ERROR_INVALID_CONTEXT         0x8002290E
#define SCE_NP_TROPHY_ERROR_PROCESSING_ABORTED      0x8002290F
#define SCE_NP_TROPHY_ERROR_ABORT                   0x80022910
#define SCE_NP_TROPHY_ERROR_UNKNOWN_HANDLE          0x80022911
#define SCE_NP_TROPHY_ERROR_LOCKED                  0x80022912
#define SCE_NP_TROPHY_ERROR_HIDDEN                  0x80022913
#define SCE_NP_TROPHY_ERROR_CANNOT_UNLOCK_PLATINUM  0x80022914
#define SCE_NP_TROPHY_ERROR_ALREADY_UNLOCKED        0x80022915
#define SCE_NP_TROPHY_ERROR_INVALID_HANDLE          0x80022917
#define SCE_NP_TROPHY_ERROR_INVALID_NP_COMM_ID      0x80022918
#define SCE_NP_TROPHY_ERROR_CONF_DOES_NOT_EXIST     0x8002291B
#define SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID       0x80022920
#define SCE_NP_TROPHY_ERROR_UNKNOWN_TROPHY_ID       0x80022921
#define SCE_NP_TROPHY_ERROR_UNKNOWN_FILE            0x80022923
#define SCE_NP_TROPHY_ERROR_UNKNOWN                 0x800229FF

/* the names this file used before it had the firmware's */
#define SCE_NP_TROPHY_ERROR_CONTEXT_ALREADY_REG     SCE_NP_TROPHY_ERROR_INVALID_CONTEXT
#define SCE_NP_TROPHY_ERROR_PLATINUM_CANNOT_UNLOCK  SCE_NP_TROPHY_ERROR_CANNOT_UNLOCK_PLATINUM

/* ---------------------------------------------------------------------------
 * Constants
 * -----------------------------------------------------------------------*/
#define SCE_NP_TROPHY_INVALID_CONTEXT       ((SceNpTrophyContext)0)
#define SCE_NP_TROPHY_INVALID_HANDLE        ((SceNpTrophyHandle)0)
#define SCE_NP_TROPHY_INVALID_TROPHY_ID     ((SceNpTrophyId)-1)

#define SCE_NP_TROPHY_MAX_NUM_TROPHIES      128
#define SCE_NP_TROPHY_NAME_MAX_SIZE         128
#define SCE_NP_TROPHY_DESC_MAX_SIZE         1024
#define SCE_NP_TROPHY_GAME_TITLE_MAX_SIZE   128
#define SCE_NP_TROPHY_GAME_DESC_MAX_SIZE    1024

#define SCE_NP_TROPHY_MAX_CONTEXTS          4
#define SCE_NP_TROPHY_MAX_HANDLES           4

/* Trophy grades */
#define SCE_NP_TROPHY_GRADE_UNKNOWN         0
#define SCE_NP_TROPHY_GRADE_PLATINUM        1
#define SCE_NP_TROPHY_GRADE_GOLD            2
#define SCE_NP_TROPHY_GRADE_SILVER          3
#define SCE_NP_TROPHY_GRADE_BRONZE          4

/* ---------------------------------------------------------------------------
 * Types
 * -----------------------------------------------------------------------*/
typedef s32 SceNpTrophyContext;
typedef s32 SceNpTrophyHandle;
typedef s32 SceNpTrophyId;

/* The guest's layouts (every integer big-endian). */
typedef struct SceNpTrophyDetails {
    u32  trophyId;
    u32  trophyGrade;
    char name[SCE_NP_TROPHY_NAME_MAX_SIZE];
    char description[SCE_NP_TROPHY_DESC_MAX_SIZE];
    u8   hidden;
    u8   padding[3];
} SceNpTrophyDetails;   /* 1164 bytes */

typedef struct SceNpTrophyData {
    u64  timestamp;     /* CellRtcTick: microseconds since 0001-01-01; 0 if locked */
    u32  trophyId;
    u8   unlocked;
    u8   padding[3];
} SceNpTrophyData;      /* 16 bytes */

typedef struct SceNpTrophyGameDetails {
    u32  numTrophies;
    u32  numPlatinum;
    u32  numGold;
    u32  numSilver;
    u32  numBronze;
    char title[SCE_NP_TROPHY_GAME_TITLE_MAX_SIZE];
    char description[SCE_NP_TROPHY_GAME_DESC_MAX_SIZE];
    u8   reserved[4];
} SceNpTrophyGameDetails;   /* 1176 bytes */

typedef struct SceNpTrophyGameData {
    u32  unlockedTrophies;
    u32  unlockedPlatinum;
    u32  unlockedGold;
    u32  unlockedSilver;
    u32  unlockedBronze;
} SceNpTrophyGameData;

/* Bitfield for trophy unlock state: 1 bit per trophy */
typedef struct SceNpTrophyFlagArray {
    u32 flag[SCE_NP_TROPHY_MAX_NUM_TROPHIES / 32];
} SceNpTrophyFlagArray;

/* ---------------------------------------------------------------------------
 * Functions
 * -----------------------------------------------------------------------*/

s32 sceNpTrophyInit(void* poolPtr, u32 poolSize, u32 containerId, u64 options);
s32 sceNpTrophyTerm(void);

s32 sceNpTrophyCreateContext(SceNpTrophyContext* context,
                             const SceNpCommunicationId* commId,
                             const SceNpCommunicationSignature* commSign,
                             u64 options);
s32 sceNpTrophyDestroyContext(SceNpTrophyContext context);

s32 sceNpTrophyCreateHandle(SceNpTrophyHandle* handle);
s32 sceNpTrophyDestroyHandle(SceNpTrophyHandle handle);

/* Real ABI: (context, handle, statusCb, arg, options).  statusCb is a guest
 * OPD for SceNpTrophyStatusCallback(context, status, completed, total, arg);
 * the game blocks on registration-complete which is signalled via this cb. */
s32 sceNpTrophyRegisterContext(SceNpTrophyContext context,
                               SceNpTrophyHandle handle,
                               u32 statusCb,
                               u32 arg,
                               u64 options);

/* SceNpTrophyStatus values fired to the status callback */
#define SCE_NP_TROPHY_STATUS_UNKNOWN            0
#define SCE_NP_TROPHY_STATUS_NOT_INSTALLED      1
#define SCE_NP_TROPHY_STATUS_DATA_CORRUPT       2
#define SCE_NP_TROPHY_STATUS_INSTALLED          3
#define SCE_NP_TROPHY_STATUS_REQUIRES_UPDATE    4
#define SCE_NP_TROPHY_STATUS_PROCESSING_COMPLETE 8

s32 sceNpTrophyGetRequiredDiskSpace(SceNpTrophyContext context,
                                    SceNpTrophyHandle handle,
                                    u64* reqSpace, u64 options);

s32 sceNpTrophyGetGameInfo(SceNpTrophyContext context,
                           SceNpTrophyHandle handle,
                           SceNpTrophyGameDetails* details,
                           SceNpTrophyGameData* data);

s32 sceNpTrophyGetTrophyInfo(SceNpTrophyContext context,
                             SceNpTrophyHandle handle,
                             SceNpTrophyId trophyId,
                             SceNpTrophyDetails* details,
                             SceNpTrophyData* data);

s32 sceNpTrophyUnlockTrophy(SceNpTrophyContext context,
                            SceNpTrophyHandle handle,
                            SceNpTrophyId trophyId,
                            SceNpTrophyId* platinumId);

s32 sceNpTrophyGetTrophyUnlockState(SceNpTrophyContext context,
                                    SceNpTrophyHandle handle,
                                    SceNpTrophyFlagArray* flags,
                                    u32* count);

s32 sceNpTrophyGetGameProgress(SceNpTrophyContext context,
                               SceNpTrophyHandle handle,
                               s32* percentage);

s32 sceNpTrophyGetGameIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
                           void* buffer, u32* size);
s32 sceNpTrophyGetTrophyIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
                             SceNpTrophyId trophyId, void* buffer, u32* size);
s32 sceNpTrophySetSoundLevel(SceNpTrophyContext context, SceNpTrophyHandle handle,
                             u32 level, u64 options);
s32 sceNpTrophyAbortHandle(SceNpTrophyHandle handle);

/* Set the trophy storage directory (defaults to "gamedata/trophies") */
void sceNpTrophySetStoragePath(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_SCE_NP_TROPHY_H */
