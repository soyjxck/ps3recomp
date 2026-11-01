/* cc -std=gnu17 -I include libs/network/tests/test_trophy_pack.c
 *   libs/network/trophy_pack.c libs/system/cellSysutil.c
 *   -pthread -o /tmp/test_trophy_pack && /tmp/test_trophy_pack
 *
 * A made-up title's trophy pack (four trophies: a platinum, a bronze and a
 * hidden silver linked to it, a DLC gold that is not) through the HLE: the
 * names and grades from the pack, a hidden trophy's details withheld, the
 * platinum earned with the last linked trophy, the unlocks kept across a
 * restart, the icons, the port's listener. */
#include "../sceNpTrophy.c"
#include <assert.h>
#include <sys/stat.h>
uint8_t* vm_base;
uint32_t ppu_vm_size;
int g_resv_store_active;
uint32_t g_ww_lo, g_ww_hi;
const char* ppu_vfs_root = ".";
void ppu_resv_break_store(uint64_t ea) { (void)ea; }
void ps3_ww_report_inline(uint32_t a, uint64_t v, int w) { (void)a; (void)v; (void)w; }
int spu_coh_is_reserved(uint32_t a) { (void)a; return 0; }
void spu_coh_notify_write(uint32_t a) { (void)a; }
void spu_lockline_lock(void) {}
void spu_lockline_unlock(void) {}
/* cellSysutilCheckCallback's other pumps */
void cellMsgDialog_tick(void) {}
void np_psnr_pump(void) {}
void ps3_np_eula_tick(void) {}

static void status_cb(u32 opd, u64 context, u64 status, u64 completed, u64 total,
                      u64 arg, u64 a, u64 b, u64 c)
{
    (void)opd; (void)context; (void)status; (void)completed; (void)total; (void)arg; (void)a; (void)b; (void)c;
}

static int heard[8], nheard;
static char heard_grade[8];
static size_t heard_icon[8];
static void listener(const TrophyUnlockEvent* e, void* user)
{
    assert(user == (void*)0x1234);
    assert(nheard < 8);
    heard_grade[nheard] = e->grade;
    heard_icon[nheard] = e->icon_size;
    heard[nheard++] = e->id;
}

static const char kConf[] =
    "<!--Sce-Np-Trophy-Signature: 00-->\n"
    "<trophyconf version=\"1.1\" policy=\"large\">\n"
    " <npcommid>NPWR99999_00</npcommid>\n <trophyset-version>01.00</trophyset-version>\n"
    " <group id=\"001\"/>\n"
    " <trophy id=\"000\" hidden=\"no\" ttype=\"P\" pid=\"-1\"/>\n"
    " <trophy id=\"001\" hidden=\"no\" ttype=\"B\" pid=\"000\"/>\n"
    " <trophy id=\"002\" hidden=\"yes\" ttype=\"S\" pid=\"000\"/>\n"
    " <trophy id=\"003\" hidden=\"no\" ttype=\"G\" pid=\"-1\" gid=\"001\"/>\n"
    "</trophyconf>\n";
static const char kNames[] =
    "<!--Sce-Np-Trophy-Signature: 00-->\n"
    "<trophyconf version=\"1.1\" policy=\"large\">\n"
    " <npcommid>NPWR99999_00</npcommid>\n <trophyset-version>01.00</trophyset-version>\n"
    " <title-name>Test &amp; Title</title-name>\n <title-detail>Made up</title-detail>\n"
    " <group id=\"001\">\n  <name>Extra</name>\n  <detail>The DLC</detail>\n </group>\n"
    " <trophy id=\"000\" hidden=\"no\" ttype=\"P\" pid=\"-1\">\n  <name>All of It</name>\n  <detail>Everything.</detail>\n </trophy>\n"
    " <trophy id=\"001\" hidden=\"no\" ttype=\"B\" pid=\"000\">\n  <name>Bronze One</name>\n  <detail>A &lt;first&gt; step.</detail>\n </trophy>\n"
    " <trophy id=\"002\" hidden=\"yes\" ttype=\"S\" pid=\"000\">\n  <name>Secret</name>\n  <detail>Hush.</detail>\n </trophy>\n"
    " <trophy id=\"003\" hidden=\"no\" ttype=\"G\" pid=\"-1\" gid=\"001\">\n  <name>Extra Gold</name>\n  <detail>The DLC's.</detail>\n </trophy>\n"
    "</trophyconf>\n";
static const unsigned char kPng[] = { 0x89, 'P', 'N', 'G', 13, 10, 26, 10, 1, 2, 3, 4, 5, 6 };

static void be32w(unsigned char* p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void be64w(unsigned char* p, uint64_t v) { be32w(p, (uint32_t)(v >> 32)); be32w(p + 4, (uint32_t)v); }

static void write_trp(const char* path)
{
    const char* names[7] = { "TROPCONF.SFM", "TROP.SFM", "ICON0.PNG", "TROP000.PNG", "TROP001.PNG", "TROP002.PNG", "TROP003.PNG" };
    const void* data[7] = { kConf, kNames, kPng, kPng, kPng, kPng, kPng };
    const size_t sizes[7] = { sizeof kConf - 1, sizeof kNames - 1, sizeof kPng, sizeof kPng, sizeof kPng, sizeof kPng, sizeof kPng };
    unsigned char hdr[0x40 + 7 * 0x40];
    memset(hdr, 0, sizeof hdr);
    be32w(hdr, 0xDCA24D00);
    be32w(hdr + 4, 1);
    be32w(hdr + 0x10, 7);
    be32w(hdr + 0x14, 0x40);
    uint64_t off = sizeof hdr;
    for (int i = 0; i < 7; i++) {
        unsigned char* e = hdr + 0x40 + i * 0x40;
        memcpy(e, names[i], strlen(names[i]));
        be64w(e + 0x20, off);
        be64w(e + 0x28, sizes[i]);
        off += sizes[i];
    }
    be64w(hdr + 8, off);
    FILE* f = fopen(path, "wb");
    assert(f);
    fwrite(hdr, 1, sizeof hdr, f);
    for (int i = 0; i < 7; i++) fwrite(data[i], 1, sizes[i], f);
    fclose(f);
}

static int open_title(void)
{
    assert(sceNpTrophyInit(NULL, 0, 0, 0) == CELL_OK);
    assert(sceNpTrophyCreateContext((void*)0x200, (void*)0x100, NULL, 0) == CELL_OK);
    assert(sceNpTrophyCreateHandle((void*)0x204) == CELL_OK);
    assert(sceNpTrophyRegisterContext(1, 1, 0x300, 0, 0) == CELL_OK);
    cellSysutilCheckCallback();
    return 1;
}

int main(void)
{
    char root[] = "/tmp/ps3recomp-trophypack-XXXXXX";
    assert(mkdtemp(root));
    char p[600];
    snprintf(p, sizeof p, "%s/PS3_GAME", root); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/PS3_GAME/TROPDIR", root); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/PS3_GAME/TROPDIR/NPWR99999_00", root); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/PS3_GAME/TROPDIR/NPWR99999_00/TROPHY.TRP", root);
    write_trp(p);
    setenv("PS3_VFS_ROOT", root, 1);
    setenv("XDG_CONFIG_HOME", root, 1);   /* no account: nothing is sent */
    unsetenv("RECOMP_TROPHIES_TOKEN");

    vm_base = calloc(1, 1 << 20); assert(vm_base);
    snprintf((char*)vm_base + 0x400, 512, "%s/trophies", root);
    sceNpTrophySetStoragePath((void*)0x400);
    memcpy(vm_base + 0x100, "NPWR99999", 9);   /* num 0 */
    g_ps3_guest_caller = status_cb;
    trophy_host_set_listener(listener, (void*)0x1234);

    open_title();
    assert(s_contexts[1].pack && s_contexts[1].total_trophies == 4);

    /* the set */
    assert(sceNpTrophyGetGameInfo(1, 1, (void*)0x1000, (void*)0x2000) == CELL_OK);
    assert(vm_read32(0x1000) == 4 && vm_read32(0x1004) == 1 && vm_read32(0x1008) == 1 &&
           vm_read32(0x100C) == 1 && vm_read32(0x1010) == 1);
    assert(!strcmp((char*)vm_base + 0x1014, "Test & Title"));
    assert(!strcmp((char*)vm_base + 0x1014 + 128, "Made up"));
    assert(vm_read32(0x2000) == 0);

    /* a trophy, a hidden one, bad ids */
    assert(sceNpTrophyGetTrophyInfo(1, 1, 1, (void*)0x3000, (void*)0x4000) == CELL_OK);
    assert(vm_read32(0x3000) == 1 && vm_read32(0x3004) == SCE_NP_TROPHY_GRADE_BRONZE);
    assert(!strcmp((char*)vm_base + 0x3008, "Bronze One"));
    assert(!strcmp((char*)vm_base + 0x3008 + 128, "A <first> step."));
    assert(vm_base[0x3008 + 128 + 1024] == 0);   /* not hidden */
    assert(vm_read64(0x4000) == 0 && vm_read32(0x4008) == 1 && vm_base[0x400C] == 0);
    assert(sceNpTrophyGetTrophyInfo(1, 1, 2, (void*)0x3000, NULL) == SCE_NP_TROPHY_ERROR_HIDDEN);
    assert(sceNpTrophyGetTrophyInfo(1, 1, 4, (void*)0x3000, NULL) == SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID);
    assert(sceNpTrophyGetTrophyIcon(1, 1, 1, NULL, (void*)0x5000) == SCE_NP_TROPHY_ERROR_LOCKED);
    assert(sceNpTrophyGetTrophyIcon(1, 1, 2, NULL, (void*)0x5000) == SCE_NP_TROPHY_ERROR_HIDDEN);

    /* unlocking: the platinum only through the others */
    assert(sceNpTrophyUnlockTrophy(1, 1, 0, (void*)0x6000) == SCE_NP_TROPHY_ERROR_CANNOT_UNLOCK_PLATINUM);
    assert(sceNpTrophyUnlockTrophy(1, 1, 1, (void*)0x6000) == CELL_OK);
    assert(vm_read32(0x6000) == 0xFFFFFFFFu);
    assert(sceNpTrophyUnlockTrophy(1, 1, 1, (void*)0x6000) == SCE_NP_TROPHY_ERROR_ALREADY_UNLOCKED);
    assert(sceNpTrophyUnlockTrophy(1, 1, 3, (void*)0x6000) == CELL_OK);   /* the DLC's: no link */
    assert(vm_read32(0x6000) == 0xFFFFFFFFu);
    assert(sceNpTrophyUnlockTrophy(1, 1, 2, (void*)0x6000) == CELL_OK);   /* the last linked one */
    assert(vm_read32(0x6000) == 0);
    assert(nheard == 4 && heard[0] == 1 && heard[1] == 3 && heard[2] == 2 && heard[3] == 0);
    assert(heard_grade[0] == 'B' && heard_grade[3] == 'P' && heard_icon[0] == sizeof kPng);

    assert(sceNpTrophyGetTrophyInfo(1, 1, 2, (void*)0x3000, (void*)0x4000) == CELL_OK);   /* earned: shown */
    assert(vm_base[0x3008 + 128 + 1024] == 1 && vm_base[0x400C] == 1);
    assert(vm_read64(0x4000) > 62135596800000000ULL);
    assert(sceNpTrophyGetGameInfo(1, 1, NULL, (void*)0x2000) == CELL_OK);
    assert(vm_read32(0x2000) == 4 && vm_read32(0x2004) == 1 && vm_read32(0x2010) == 1);
    assert(sceNpTrophyGetTrophyUnlockState(1, 1, (void*)0x7000, (void*)0x7100) == CELL_OK);
    assert(vm_read32(0x7000) == 0xF && vm_read32(0x7100) == 4);
    assert(sceNpTrophyGetGameProgress(1, 1, (void*)0x7200) == CELL_OK && vm_read32(0x7200) == 100);

    /* the icons: the size, then the bytes */
    vm_write32(0x5000, 0);
    assert(sceNpTrophyGetTrophyIcon(1, 1, 2, (void*)0x8000, (void*)0x5000) == CELL_OK);
    assert(vm_read32(0x5000) == sizeof kPng && vm_base[0x8000] == 0);
    assert(sceNpTrophyGetTrophyIcon(1, 1, 2, (void*)0x8000, (void*)0x5000) == CELL_OK);
    assert(!memcmp(vm_base + 0x8000, kPng, sizeof kPng));
    vm_write32(0x5000, 4096);
    assert(sceNpTrophyGetGameIcon(1, 1, (void*)0x9000, (void*)0x5000) == CELL_OK && vm_read32(0x5000) == sizeof kPng);

    /* the snapshot a port's in-game list reads */
    TrophyPack* snap = NULL;
    TrophyUnlocks su;
    assert(trophy_host_snapshot(&snap, &su) == 0 && snap && snap->count == 4 && su.unlocked[0] && su.unlocked[3]);
    trophy_pack_free(snap);

    /* kept across a restart */
    assert(sceNpTrophyTerm() == CELL_OK);
    nheard = 0;
    setenv("PS3_TROPHY_PREVIEW", "3", 1);
    open_title();
    assert(nheard == 1 && heard[0] == 3);   /* the preview: heard, not recorded again */
    assert(trophy_count_unlocked(&s_contexts[1]) == 4);
    assert(sceNpTrophyUnlockTrophy(1, 1, 3, NULL) == SCE_NP_TROPHY_ERROR_ALREADY_UNLOCKED);
    assert(sceNpTrophyTerm() == CELL_OK);

    char cmd[700];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    assert(system(cmd) == 0);
    free(vm_base);
    puts("Trophy pack: names, grades, hidden details, platinum, icons and saved unlocks");
}
