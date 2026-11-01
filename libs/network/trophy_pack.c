/*
 * ps3recomp - a title's trophy pack and the player's unlocks (trophy_pack.h)
 */

#include "trophy_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

/* ---------------------------------------------------------------------------
 * Files (UTF-8 paths on every host)
 * -----------------------------------------------------------------------*/

static FILE* tp_fopen(const char* path, const char* mode)
{
#ifdef _WIN32
    wchar_t wp[1024], wm[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 1024)) return NULL;
    if (!MultiByteToWideChar(CP_UTF8, 0, mode, -1, wm, 8)) return NULL;
    return _wfopen(wp, wm);
#else
    return fopen(path, mode);
#endif
}

static int tp_rename(const char* from, const char* to)
{
#ifdef _WIN32
    wchar_t wf[1024], wt[1024];
    if (!MultiByteToWideChar(CP_UTF8, 0, from, -1, wf, 1024)) return -1;
    if (!MultiByteToWideChar(CP_UTF8, 0, to, -1, wt, 1024)) return -1;
    return MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(from, to);
#endif
}

static int tp_is_file(const char* path)
{
    FILE* f = tp_fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static uint32_t be32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static uint64_t be64(const uint8_t* p) { return ((uint64_t)be32(p) << 32) | be32(p + 4); }

/* ---------------------------------------------------------------------------
 * The SFM's XML: just what the trophy configuration uses
 * -----------------------------------------------------------------------*/

static void put_utf8(char** o, char* end, unsigned cp)
{
    char b[4]; int n;
    if (cp < 0x80) { b[0] = (char)cp; n = 1; }
    else if (cp < 0x800) { b[0] = (char)(0xC0 | (cp >> 6)); b[1] = (char)(0x80 | (cp & 0x3F)); n = 2; }
    else if (cp < 0x10000) { b[0] = (char)(0xE0 | (cp >> 12)); b[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); b[2] = (char)(0x80 | (cp & 0x3F)); n = 3; }
    else { b[0] = (char)(0xF0 | (cp >> 18)); b[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); b[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); b[3] = (char)(0x80 | (cp & 0x3F)); n = 4; }
    if (*o + n > end) return;
    memcpy(*o, b, (size_t)n);
    *o += n;
}

/* [s, e) with the entities decoded and the outer whitespace dropped, into out */
static void xml_text(const char* s, const char* e, char* out, size_t cap)
{
    if (!cap) return;
    while (s < e && (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')) s++;
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) e--;
    char* o = out;
    char* end = out + cap - 1;
    while (s < e && o < end) {
        if (*s == '&') {
            const char* semi = memchr(s, ';', (size_t)(e - s));
            if (semi && semi - s <= 10) {
                const size_t n = (size_t)(semi - s - 1);
                const char* nm = s + 1;
                unsigned cp = 0; int ok = 1;
                if (n == 3 && !strncmp(nm, "amp", 3)) cp = '&';
                else if (n == 2 && !strncmp(nm, "lt", 2)) cp = '<';
                else if (n == 2 && !strncmp(nm, "gt", 2)) cp = '>';
                else if (n == 4 && !strncmp(nm, "quot", 4)) cp = '"';
                else if (n == 4 && !strncmp(nm, "apos", 4)) cp = '\'';
                else if (n >= 2 && nm[0] == '#') {
                    char num[12] = {0};
                    memcpy(num, nm + 1, n - 1 < 11 ? n - 1 : 11);
                    cp = (num[0] == 'x' || num[0] == 'X') ? (unsigned)strtoul(num + 1, NULL, 16)
                                                           : (unsigned)strtoul(num, NULL, 10);
                    ok = cp > 0 && cp < 0x110000;
                } else ok = 0;
                if (ok) { put_utf8(&o, end, cp); s = semi + 1; continue; }
            }
        }
        *o++ = *s++;
    }
    *o = 0;
}

/* The next "<tag" (followed by a space, '>' or '/') in [p, end), or NULL. */
static const char* xml_find(const char* p, const char* end, const char* tag)
{
    const size_t n = strlen(tag);
    for (; p && p + n + 1 < end; p++) {
        p = memchr(p, '<', (size_t)(end - p));
        if (!p || p + n + 1 >= end) return NULL;
        if (!strncmp(p + 1, tag, n)) {
            const char c = p[1 + n];
            if (c == ' ' || c == '>' || c == '/' || c == '\t' || c == '\r' || c == '\n') return p;
        }
    }
    return NULL;
}

/* An attribute of the tag at [t, te) (te at its '>'), into out. 1 if found. */
static int xml_attr(const char* t, const char* te, const char* name, char* out, size_t cap)
{
    const size_t n = strlen(name);
    for (const char* p = t + 1; p + n + 2 < te; p++) {
        if ((p[-1] == ' ' || p[-1] == '\t' || p[-1] == '\n' || p[-1] == '\r') && !strncmp(p, name, n)) {
            const char* q = p + n;
            while (q < te && (*q == ' ' || *q == '\t')) q++;
            if (q >= te || *q != '=') continue;
            q++;
            while (q < te && (*q == ' ' || *q == '\t')) q++;
            if (q >= te || (*q != '"' && *q != '\'')) continue;
            const char quote = *q++;
            const char* v = q;
            while (q < te && *q != quote) q++;
            xml_text(v, q, out, cap);
            return 1;
        }
    }
    return 0;
}

/* The text of the first <tag>...</tag> in [p, end), into out. 1 if found. */
static int xml_child(const char* p, const char* end, const char* tag, char* out, size_t cap)
{
    const char* t = xml_find(p, end, tag);
    if (!t) return 0;
    const char* te = memchr(t, '>', (size_t)(end - t));
    if (!te) return 0;
    if (te[-1] == '/') { if (cap) out[0] = 0; return 1; }
    char close[48];
    snprintf(close, sizeof close, "</%s>", tag);
    const char* c = te + 1;
    const size_t cn = strlen(close);
    while (c + cn <= end && strncmp(c, close, cn)) c++;
    if (c + cn > end) return 0;
    xml_text(te + 1, c, out, cap);
    return 1;
}

/* One <tag ...> element: its attributes' range [*t, *te) and its body
 * [*b, *be) (empty when it closes itself); the position after it. */
static const char* xml_element(const char* p, const char* end, const char* tag,
                               const char** t, const char** te, const char** b, const char** be)
{
    *t = xml_find(p, end, tag);
    if (!*t) return NULL;
    *te = memchr(*t, '>', (size_t)(end - *t));
    if (!*te) return NULL;
    if ((*te)[-1] == '/') { *b = *be = *te; return *te + 1; }
    char close[48];
    snprintf(close, sizeof close, "</%s>", tag);
    const size_t cn = strlen(close);
    const char* c = *te + 1;
    while (c + cn <= end && strncmp(c, close, cn)) c++;
    if (c + cn > end) return NULL;
    *b = *te + 1; *be = c;
    return c + cn;
}

static void parse_sfm(TrophyPack* pk, const char* x, size_t n)
{
    const char* end = x + n;
    /* past the signature comment */
    const char* root = xml_find(x, end, "trophyconf");
    if (root) x = root;
    char buf[64];
    xml_child(x, end, "npcommid", pk->np_comm_id, sizeof pk->np_comm_id);
    xml_child(x, end, "trophyset-version", pk->set_version, sizeof pk->set_version);
    xml_child(x, end, "title-name", pk->title, sizeof pk->title);
    xml_child(x, end, "title-detail", pk->detail, sizeof pk->detail);

    const char *t, *te, *b, *be;
    for (const char* p = x; p && (p = xml_element(p, end, "group", &t, &te, &b, &be)) != NULL;) {
        if (pk->group_count >= TROPHY_PACK_MAX_GROUPS) break;
        if (!xml_attr(t, te, "id", buf, sizeof buf)) continue;
        TrophyPackGroup* g = &pk->groups[pk->group_count++];
        memset(g, 0, sizeof *g);
        g->id = atoi(buf);
        if (b < be) {
            xml_child(b, be, "name", g->name, sizeof g->name);
            xml_child(b, be, "detail", g->detail, sizeof g->detail);
        }
    }
    for (const char* p = x; p && (p = xml_element(p, end, "trophy", &t, &te, &b, &be)) != NULL;) {
        if (!xml_attr(t, te, "id", buf, sizeof buf)) continue;
        const int id = atoi(buf);
        if (id < 0 || id >= TROPHY_PACK_MAX) continue;
        TrophyPackTrophy* tr = &pk->trophies[id];
        if (!tr->valid) pk->count++;
        tr->valid = 1;
        if (xml_attr(t, te, "hidden", buf, sizeof buf)) tr->hidden = buf[0] == 'y' || buf[0] == 'Y';
        if (xml_attr(t, te, "ttype", buf, sizeof buf)) tr->grade = buf[0];
        tr->pid = xml_attr(t, te, "pid", buf, sizeof buf) ? atoi(buf) : -1;
        tr->group = xml_attr(t, te, "gid", buf, sizeof buf) ? atoi(buf) : -1;
        if (b < be) {
            xml_child(b, be, "name", tr->name, sizeof tr->name);
            xml_child(b, be, "detail", tr->detail, sizeof tr->detail);
        }
        if (id + 1 > pk->end) pk->end = id + 1;
        if (tr->grade == 'P' && pk->platinum < 0) pk->platinum = id;
    }
}

/* ---------------------------------------------------------------------------
 * The archive
 * -----------------------------------------------------------------------*/

static const TrophyPackEntry* find_entry(const TrophyPack* pk, const char* name)
{
    for (int i = 0; i < pk->entry_count; i++)
#ifdef _WIN32
        if (!_stricmp(pk->entries[i].name, name)) return &pk->entries[i];
#else
        if (!strcasecmp(pk->entries[i].name, name)) return &pk->entries[i];
#endif
    return NULL;
}

int trophy_pack_read(const TrophyPack* pk, const char* entry, uint8_t** out, size_t* size)
{
    *out = NULL;
    if (size) *size = 0;
    const TrophyPackEntry* e = find_entry(pk, entry);
    if (!e || e->size > 64u * 1024 * 1024) return -1;
    FILE* f = tp_fopen(pk->path, "rb");
    if (!f) return -1;
    uint8_t* buf = (uint8_t*)malloc((size_t)e->size + 1);
    int ok = buf != NULL;
#ifdef _WIN32
    ok = ok && _fseeki64(f, (__int64)e->offset, SEEK_SET) == 0;
#else
    ok = ok && fseeko(f, (off_t)e->offset, SEEK_SET) == 0;
#endif
    ok = ok && fread(buf, 1, (size_t)e->size, f) == (size_t)e->size;
    fclose(f);
    if (!ok) { free(buf); return -1; }
    buf[e->size] = 0;   /* the SFMs are read as text */
    *out = buf;
    if (size) *size = (size_t)e->size;
    return 0;
}

int trophy_pack_icon(const TrophyPack* pk, int id, uint8_t** png, size_t* size)
{
    char name[32];
    if (id < 0) snprintf(name, sizeof name, "ICON0.PNG");
    else snprintf(name, sizeof name, "TROP%03d.PNG", id);
    return trophy_pack_read(pk, name, png, size);
}

TrophyPack* trophy_pack_load(const char* trp_path, int language)
{
    FILE* f = tp_fopen(trp_path, "rb");
    if (!f) return NULL;
    uint8_t h[0x40];
    if (fread(h, 1, sizeof h, f) != sizeof h || be32(h) != 0xDCA24D00u) { fclose(f); return NULL; }
    uint32_t count = be32(h + 0x10), esize = be32(h + 0x14);
    if (esize < 0x40 || esize > 0x1000 || count > TROPHY_PACK_MAX_ENTRIES) { fclose(f); return NULL; }
    TrophyPack* pk = (TrophyPack*)calloc(1, sizeof *pk);
    if (!pk) { fclose(f); return NULL; }
    snprintf(pk->path, sizeof pk->path, "%s", trp_path);
    pk->platinum = -1;
    uint8_t e[0x1000];
    for (uint32_t i = 0; i < count; i++) {
        if (fread(e, 1, esize, f) != esize) break;
        TrophyPackEntry* en = &pk->entries[pk->entry_count++];
        memcpy(en->name, e, 32);
        en->name[32] = 0;
        en->offset = be64(e + 0x20);
        en->size = be64(e + 0x28);
    }
    fclose(f);

    /* the names: the console language's, the other English, the title's own */
    char want[4][24];
    int nw = 0;
    if (language >= 0) {
        snprintf(want[nw++], sizeof want[0], "TROP_%02d.SFM", language);
        if (language == TROPHY_LANG_ENGLISH_US) snprintf(want[nw++], sizeof want[0], "TROP_%02d.SFM", TROPHY_LANG_ENGLISH_GB);
        if (language == TROPHY_LANG_ENGLISH_GB) snprintf(want[nw++], sizeof want[0], "TROP_%02d.SFM", TROPHY_LANG_ENGLISH_US);
    }
    snprintf(want[nw++], sizeof want[0], "TROP.SFM");
    snprintf(want[nw++], sizeof want[0], "TROPCONF.SFM");
    for (int i = 0; i < nw; i++) {
        uint8_t* x = NULL;
        size_t n = 0;
        if (trophy_pack_read(pk, want[i], &x, &n) != 0) continue;
        parse_sfm(pk, (const char*)x, n);
        free(x);
        if (pk->count > 0) {
            memcpy(pk->language_file, want[i], sizeof pk->language_file - 1);
            break;
        }
    }
    if (pk->count == 0) { free(pk); return NULL; }
    for (int id = 0; id < pk->end; id++) {
        TrophyPackTrophy* t = &pk->trophies[id];
        if (t->valid && !t->name[0]) snprintf(t->name, sizeof t->name, "Trophy %d", id);
    }
    return pk;
}

void trophy_pack_free(TrophyPack* pk) { free(pk); }

const TrophyPackTrophy* trophy_pack_trophy(const TrophyPack* pk, int id)
{
    if (!pk || id < 0 || id >= TROPHY_PACK_MAX || !pk->trophies[id].valid) return NULL;
    return &pk->trophies[id];
}

int trophy_grade_number(char g)
{
    switch (g) {
    case 'P': return 1;
    case 'G': return 2;
    case 'S': return 3;
    case 'B': return 4;
    default:  return 0;
    }
}

const char* trophy_grade_name(char g)
{
    switch (g) {
    case 'P': return "Platinum";
    case 'G': return "Gold";
    case 'S': return "Silver";
    case 'B': return "Bronze";
    default:  return "Trophy";
    }
}

/* The first folder in `dir` (a TROPDIR), into out; 0 if there is one. */
static int first_subdir(const char* dir, char* out, size_t cap)
{
#ifdef _WIN32
    wchar_t pat[1024];
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s/*", dir);
    if (!MultiByteToWideChar(CP_UTF8, 0, tmp, -1, pat, 1024)) return -1;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int found = -1;
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] != L'.') {
            if (WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, out, (int)cap, NULL, NULL)) found = 0;
        }
    } while (found != 0 && FindNextFileW(h, &fd));
    FindClose(h);
    return found;
#else
    DIR* d = opendir(dir);
    if (!d) return -1;
    int found = -1;
    struct dirent* de;
    while (found != 0 && (de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        char p[1024];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
        if (stat(p, &st) == 0 && S_ISDIR(st.st_mode)) { snprintf(out, cap, "%s", de->d_name); found = 0; }
    }
    closedir(d);
    return found;
#endif
}

int trophy_pack_find(const char* dir, const char* np_comm_id, char* out, size_t cap)
{
    static const char* const under[] = { "PS3_GAME/TROPDIR", "TROPDIR" };
    for (size_t i = 0; i < sizeof under / sizeof under[0]; i++) {
        char td[1024], id[64];
        snprintf(td, sizeof td, "%s/%s", dir, under[i]);
        if (np_comm_id && *np_comm_id) snprintf(id, sizeof id, "%s", np_comm_id);
        else if (first_subdir(td, id, sizeof id) != 0) continue;
        snprintf(out, cap, "%s/%s/TROPHY.TRP", td, id);
        if (tp_is_file(out)) return 0;
    }
    return -1;
}

/* ---------------------------------------------------------------------------
 * The unlocks
 * -----------------------------------------------------------------------*/

void trophy_unlocks_path(const char* storage_dir, const char* np_comm_id, char* out, size_t cap)
{
    char base[16];
    snprintf(base, sizeof base, "%s", np_comm_id ? np_comm_id : "");
    char* us = strchr(base, '_');
    if (us) *us = 0;   /* NPWR04682_00 -> NPWR04682 */
    snprintf(out, cap, "%s/%s.json", storage_dir, base);
}

int trophy_unlocks_load(const char* json_path, TrophyUnlocks* u)
{
    memset(u, 0, sizeof *u);
    FILE* fp = tp_fopen(json_path, "rb");
    if (!fp) return 0;   /* nothing yet */
    char line[512];
    while (fgets(line, sizeof line, fp)) {
        const char* id_pos = strstr(line, "\"id\":");
        if (!id_pos) continue;
        unsigned tid = 0;
        unsigned long long ts = 0;
        if (sscanf(id_pos, "\"id\": %u", &tid) != 1 || tid >= TROPHY_PACK_MAX) continue;
        const char* ts_pos = strstr(line, "\"timestamp\":");
        if (ts_pos) sscanf(ts_pos, "\"timestamp\": %llu", &ts);
        u->unlocked[tid] = 1;
        u->at[tid] = (uint64_t)ts;
    }
    fclose(fp);
    return 0;
}

int trophy_unlocks_save(const char* json_path, const TrophyUnlocks* u, const char* np_comm_id)
{
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", json_path);
    FILE* fp = tp_fopen(tmp, "wb");
    if (!fp) return -1;
    fprintf(fp, "{\n");
    if (np_comm_id && *np_comm_id) fprintf(fp, "  \"np_comm_id\": \"%s\",\n", np_comm_id);
    fprintf(fp, "  \"trophies\": [");
    int first = 1;
    for (int i = 0; i < TROPHY_PACK_MAX; i++) {
        if (!u->unlocked[i]) continue;
        fprintf(fp, "%s\n    { \"id\": %d, \"timestamp\": %llu }", first ? "" : ",", i, (unsigned long long)u->at[i]);
        first = 0;
    }
    fprintf(fp, "\n  ]\n}\n");
    const int ok = fflush(fp) == 0;
    fclose(fp);
    if (!ok || tp_rename(tmp, json_path) != 0) { remove(tmp); return -1; }
    return 0;
}
