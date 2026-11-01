/*
 * ps3recomp - a title's trophy pack and the player's unlocks, on the host
 *
 * The trophy pack is the title's own PS3_GAME/TROPDIR/<NP comm id>/TROPHY.TRP:
 * a plain archive (big-endian header, magic 0xDCA24D00) of
 *   TROPCONF.SFM      the trophies' ids, grades, hidden flags, platinum links
 *   TROP.SFM          the same with names and details, in the title's own language
 *   TROP_xx.SFM       a translation; xx is the console language (CELL_SYSUTIL_LANG_*)
 *   ICON0.PNG         the set's icon (320x176)
 *   TROPnnn.PNG       each trophy's icon (240x240)
 * The SFM files are XML (a signature comment first).
 *
 * The unlocks are the runtime's own JSON file (sceNpTrophy.c writes it):
 *   { "np_comm_id": "NPWR04682_00",
 *     "trophies": [ { "id": 1, "timestamp": 1761900000 }, ... ] }
 * one trophy a line, the time in Unix seconds.
 *
 * Plain C with no guest memory, so a port's launcher reads the same files the
 * running game does.
 */

#ifndef PS3RECOMP_TROPHY_PACK_H
#define PS3RECOMP_TROPHY_PACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TROPHY_PACK_MAX          128   /* ids 0..127 (SCE_NP_TROPHY_FLAG_SETSIZE) */
#define TROPHY_PACK_MAX_GROUPS   64
#define TROPHY_PACK_MAX_ENTRIES  512
#define TROPHY_NAME_SIZE         128   /* SCE_NP_TROPHY_NAME_MAX_SIZE */
#define TROPHY_DETAIL_SIZE       1024  /* SCE_NP_TROPHY_DESCR_MAX_SIZE */

/* The console languages that name the TROP_xx.SFM files (CELL_SYSUTIL_LANG_*). */
#define TROPHY_LANG_DEFAULT      (-1)  /* TROP.SFM, the title's own */
#define TROPHY_LANG_JAPANESE     0
#define TROPHY_LANG_ENGLISH_US   1
#define TROPHY_LANG_ENGLISH_GB   18

typedef struct TrophyPackTrophy {
    unsigned char valid;      /* the pack lists this id */
    unsigned char hidden;
    char          grade;      /* 'P' 'G' 'S' 'B' */
    int           pid;        /* the platinum it counts toward; -1 none */
    int           group;      /* -1 the base game, else the group's number ("001" = 1) */
    char          name[TROPHY_NAME_SIZE];
    char          detail[TROPHY_DETAIL_SIZE];
} TrophyPackTrophy;

typedef struct TrophyPackGroup {
    int  id;
    char name[TROPHY_NAME_SIZE];
    char detail[TROPHY_DETAIL_SIZE];
} TrophyPackGroup;

typedef struct TrophyPackEntry {
    char     name[33];
    uint64_t offset, size;
} TrophyPackEntry;

typedef struct TrophyPack {
    char np_comm_id[16];       /* "NPWR04682_00" */
    char set_version[16];      /* "01.05" */
    char title[TROPHY_NAME_SIZE];
    char detail[TROPHY_DETAIL_SIZE];
    char language_file[16];    /* the SFM the names came from */
    int  count;                /* trophies listed */
    int  end;                  /* highest id + 1 */
    int  platinum;             /* the platinum's id, or -1 */
    TrophyPackTrophy trophies[TROPHY_PACK_MAX];   /* by id */
    int  group_count;
    TrophyPackGroup groups[TROPHY_PACK_MAX_GROUPS];
    /* the archive, for the icons */
    char path[1024];
    int  entry_count;
    TrophyPackEntry entries[TROPHY_PACK_MAX_ENTRIES];
} TrophyPack;

/* Read a TROPHY.TRP. `language` (CELL_SYSUTIL_LANG_*, or TROPHY_LANG_DEFAULT)
 * picks TROP_xx.SFM; without one, the other English, then TROP.SFM, then the
 * names-less TROPCONF.SFM ("Trophy 12"). NULL if the file is not a trophy pack.
 * Free with trophy_pack_free. */
TrophyPack* trophy_pack_load(const char* trp_path, int language);
void        trophy_pack_free(TrophyPack* pack);

/* The trophy `id`, or NULL if the pack has none. */
const TrophyPackTrophy* trophy_pack_trophy(const TrophyPack* pack, int id);

/* An entry's bytes ("ICON0.PNG"), malloc'd into *out. 0 on success. */
int trophy_pack_read(const TrophyPack* pack, const char* entry, uint8_t** out, size_t* size);

/* A trophy's PNG (TROPnnn.PNG), or with id < 0 the set's (ICON0.PNG). */
int trophy_pack_icon(const TrophyPack* pack, int id, uint8_t** png, size_t* size);

/* 'P' -> SCE_NP_TROPHY_GRADE_PLATINUM (1) ... 'B' -> BRONZE (4); 0 unknown. */
int trophy_grade_number(char grade);
/* "Platinum", "Gold", "Silver", "Bronze" */
const char* trophy_grade_name(char grade);

/* Find a title's pack under a disc or game folder (the one holding
 * PS3_GAME/, or PS3_GAME itself, or a PSN game's folder): `dir`/[PS3_GAME/]
 * TROPDIR/<np_comm_id>/TROPHY.TRP. `np_comm_id` NULL or "" takes the first
 * pack in TROPDIR. 0 and the path in `out` if there is one. */
int trophy_pack_find(const char* dir, const char* np_comm_id, char* out, size_t cap);

/* ---- the unlocks ---------------------------------------------------------- */

typedef struct TrophyUnlocks {
    unsigned char unlocked[TROPHY_PACK_MAX];
    uint64_t      at[TROPHY_PACK_MAX];      /* Unix seconds */
} TrophyUnlocks;

/* Read the runtime's file (absent: nothing unlocked, and 0). 0 on success. */
int trophy_unlocks_load(const char* json_path, TrophyUnlocks* out);
/* Write it (through a temporary file and a rename). 0 on success. */
int trophy_unlocks_save(const char* json_path, const TrophyUnlocks* u, const char* np_comm_id);

/* The file sceNpTrophy keeps a title's unlocks in, under `storage_dir`
 * (the runtime's "gamedata/trophies" or $PS3_GAMEDATA_ROOT/trophies):
 * <dir>/<the comm id without its _NN>.json, as the runtime always named it. */
void trophy_unlocks_path(const char* storage_dir, const char* np_comm_id, char* out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_TROPHY_PACK_H */
