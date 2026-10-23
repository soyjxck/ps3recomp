/*
 * ps3recomp - Memory management syscalls (implementation)
 */

#include "sys_memory.h"
#include "../ps3_log.h"   /* ps3_log_verbose */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/ps3emu/log_first.h"   /* relative: tools/test_*memory*.py compile this file alone */

/* ---------------------------------------------------------------------------
 * Globals
 * -----------------------------------------------------------------------*/
sys_mem_alloc_info       g_sys_mem_allocs[SYS_MEMORY_ALLOC_MAX];
sys_mem_container_info   g_sys_mem_containers[SYS_MEMORY_CONTAINER_MAX];
sys_mmapper_shared_info  g_sys_mmapper_shared[SYS_MMAPPER_SHARED_MAX];

/* Bump allocator for main memory pool.
 * Starts after a reasonable offset to avoid the low addresses used by ELF. */
uint32_t g_sys_mem_bump_ptr = 0;

/* Total user memory size (default 213 MB, after kernel reservation) */
#define SYS_MEM_USER_TOTAL  (213 * 1024 * 1024)

/* Guest window handed out by sys_memory_allocate.
 * 0x40000000+ matches where the real lv2 places these allocations (verified
 * against an RPCS3 boot log of Yakuza: Dead Souls); the window is outside
 * the pre-committed main region, so pages are committed on demand. */
#define SYS_MEM_ALLOC_BASE  0x40000000u
#define SYS_MEM_ALLOC_END   0x50000000u

/* Overflow window, used once the one above is exhausted. A title can need
 * more than the PS3's ~213 MB here, because the HLE does not reproduce every
 * way the real system bounds or recycles memory; failing it with ENOMEM ends
 * the run (UE3 treats it as fatal) where real hardware would have carried on.
 * 0x70000000..0x80000000 sits above sys_vm and below the sign bit, so a guest
 * that compares pointers as signed still sees a positive one. */
#define SYS_MEM_OVERFLOW_BASE 0x70000000u
#define SYS_MEM_OVERFLOW_END  0x80000000u

/* SYS_MEM_OVERFLOW_END=<hex>: move the end of the overflow window (an
 * experiment knob; past 0x80000000 a signed pointer compare misfires). */
static uint32_t sys_mem_overflow_end(void)
{
    static uint32_t s_ovf_end = 0;
    if (!s_ovf_end) { const char* e = getenv("SYS_MEM_OVERFLOW_END");
                      s_ovf_end = e ? (uint32_t)strtoul(e, 0, 16) : SYS_MEM_OVERFLOW_END; }
    return s_ovf_end;
}

/* Whether `addr` is in one of the windows above. Syscall 14 (lv2_register.c)
 * answers such an address as user memory, which lv2 keeps in segment 0x2. */
int sys_memory_in_user_windows(uint32_t addr)
{
    return (addr >= SYS_MEM_ALLOC_BASE && addr < SYS_MEM_ALLOC_END) ||
           (addr >= SYS_MEM_OVERFLOW_BASE && addr < sys_mem_overflow_end());
}

/* LV2 user memory and mapper reservations occupy separate address domains.
 * Mapper reservations (sys_mmapper_allocate_address, and 341's direct blocks)
 * come from 0x11000000..0xC0000000, less every window another owner holds:
 * the loaded image and its TLS, the two user windows above, the HLE heap,
 * audio, sys_vm and the runtime's injected window (mapper_address). The base
 * is where the loader's own answer to syscall 330 used to start, just past
 * the image, so a title is handed the addresses it was handed then:
 * Drakengard 3 reserves 256 MB at a time and maps RSX memory at 0x20000000
 * and 0x30000000. The raw-SPU window at 0x30000000 is not skipped, as it was
 * not then. */
#define SYS_MAPPER_BASE     0x11000000u
#define SYS_MAPPER_END      0xC0000000u
static uint32_t s_mapper_bump_ptr = SYS_MAPPER_BASE;
typedef struct mapper_reservation {
    uint32_t start, end, next;
} mapper_reservation;
static mapper_reservation s_mapper_reservations[SYS_MEMORY_ALLOC_MAX];

static uint32_t s_total_allocated = 0;
static uint64_t (*s_image_query)(uint64_t addr, uint64_t end);

void sys_memory_set_image_query(uint64_t (*query)(uint64_t addr, uint64_t end))
{
    s_image_query = query;
}

/* Freed ranges, reused by sys_memory_allocate. The bump allocator alone never
 * gave a byte back: sys_memory_free retired the record and the window kept
 * growing. Drakengard 3's UE3 heap (FMallocPS3DL, dlmalloc on
 * sys_memory_allocate/free) asked for 640 MB over five minutes of play while
 * keeping far less live; with the 213 MB user pool plus the overflow window
 * exhausted, the allocator reported ENOMEM, UE3's OOM handler parked holding
 * the malloc lock and the game froze. Real hardware recycles freed pages, so
 * the same play never ran out. First-fit, split, coalesced on free; blocks are
 * page multiples so a linear scan over a few hundred entries is nothing. */
#define SYS_MEM_FREE_MAX 1024
static struct { uint32_t addr, size; } s_free[SYS_MEM_FREE_MAX];
static int s_free_n = 0;
static uint32_t s_free_bytes = 0;

/* Under bump_lock: carve `size` bytes at `alignment` out of a free block. */
static uint32_t sys_mem_reuse(uint32_t size, uint32_t alignment)
{
    for (int i = 0; i < s_free_n; i++) {
        uint32_t a = VM_ALIGN_UP(s_free[i].addr, alignment);
        uint32_t end = s_free[i].addr + s_free[i].size;
        if (a < s_free[i].addr || a + size > end || a + size < a) continue;
        uint32_t head = a - s_free[i].addr, tail = end - (a + size);
        s_free_bytes -= size;
        if (head && tail && s_free_n < SYS_MEM_FREE_MAX) {
            s_free[i].size = head;
            s_free[s_free_n].addr = a + size; s_free[s_free_n].size = tail; s_free_n++;
        } else if (head) {
            s_free[i].size = head;            /* tail (if any) is lost: table full */
            if (tail) s_free_bytes -= tail;
        } else if (tail) {
            s_free[i].addr = a + size; s_free[i].size = tail;
        } else {
            s_free[i] = s_free[--s_free_n];
        }
        return a;
    }
    return 0;
}

/* Under bump_lock: give a range back, merging with neighbours. */
static void sys_mem_release(uint32_t addr, uint32_t size)
{
    s_free_bytes += size;
    for (int i = 0; i < s_free_n; i++) {
        if (s_free[i].addr + s_free[i].size == addr) { s_free[i].size += size; addr = s_free[i].addr; size = s_free[i].size;
            s_free[i] = s_free[--s_free_n]; i = -1; continue; }
        if (addr + size == s_free[i].addr) { size += s_free[i].size;
            s_free[i] = s_free[--s_free_n]; i = -1; continue; }
    }
    if (s_free_n < SYS_MEM_FREE_MAX) { s_free[s_free_n].addr = addr; s_free[s_free_n].size = size; s_free_n++; }
    else { s_free_bytes -= size;
           static int _n = 0; if (_n < 4 && ++_n) fprintf(stderr, "[sys_memory] free list full, 0x%X bytes at 0x%08X not recycled\n", size, addr); }
}

/* Guest threads are real host threads; serialize the bump allocator. */
#ifdef _WIN32
static SRWLOCK s_bump_lock = SRWLOCK_INIT;
static void bump_lock(void)   { AcquireSRWLockExclusive(&s_bump_lock); }
static void bump_unlock(void) { ReleaseSRWLockExclusive(&s_bump_lock); }
#else
#include <pthread.h>
static pthread_mutex_t s_bump_mtx = PTHREAD_MUTEX_INITIALIZER;
static void bump_lock(void)   { pthread_mutex_lock(&s_bump_mtx); }
static void bump_unlock(void) { pthread_mutex_unlock(&s_bump_mtx); }
#endif

static void write_be32(uint32_t addr, uint32_t val)
{
    uint32_t* p = (uint32_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
          ((val <<  8) & 0xFF0000) | ((val << 24) & 0xFF000000u);
#endif
    *p = val;
}

/* ---------------------------------------------------------------------------
 * sys_memory_allocate
 *
 * r3 = size
 * r4 = flags (page size)
 * r5 = pointer to receive allocated address (u32*)
 * -----------------------------------------------------------------------*/
int64_t sys_memory_allocate(ppu_context* ctx)
{
    uint64_t requested = LV2_ARG_U64(ctx, 0);
    uint32_t flags     = LV2_ARG_U32(ctx, 1);
    uint32_t addr_out  = LV2_ARG_PTR(ctx, 2);

    if (ps3_log_verbose())   /* Drakengard 3 allocates ten times a second */
        fprintf(stderr, "[sys_memory] allocate(size=0x%llX, flags=0x%X)\n",
                (unsigned long long)requested, flags);

    /* Determine alignment based on page size flags: 1 MB or 64 KB */
    uint32_t alignment = (flags & SYS_MEMORY_PAGE_SIZE_1M) ? 0x100000u : 0x10000u;

    /* The size is a 64-bit argument: one that does not fit 32 bits once
     * rounded up is refused, not truncated. */
    uint64_t rounded = (requested + alignment - 1u) & ~(uint64_t)(alignment - 1u);
    if (!requested || requested > UINT32_MAX || !rounded || rounded > UINT32_MAX)
        return (int64_t)(int32_t)CELL_EINVAL;
    uint32_t size = (uint32_t)rounded;

    bump_lock();

    /* A freed range first: the pages are already committed, so only the
     * contents need clearing to look like fresh ones. */
    { uint32_t r = sys_mem_reuse(size, alignment);
      if (r) {
          int slot = -1;
          for (int i = 0; i < SYS_MEMORY_ALLOC_MAX; i++)
              if (!g_sys_mem_allocs[i].active) { slot = i; break; }
          if (slot >= 0) {
              s_total_allocated += size;
              sys_mem_alloc_info* a = &g_sys_mem_allocs[slot];
              a->active = 1; a->addr = r; a->size = size; a->container_id = 0; a->page_size = alignment;
              bump_unlock();
              memset(vm_to_host(r), 0, size);
              { static unsigned long _n = 0; if (_n++ < 40 || (_n % 200) == 0)
                  fprintf(stderr, "[sys_memory] allocate -> 0x%08X (reused; live=%u MB, free %u MB in %d blocks)\n",
                          r, s_total_allocated >> 20, s_free_bytes >> 20, s_free_n); }
              if (addr_out != 0) write_be32(addr_out, r);
              return CELL_OK;
          }
          sys_mem_release(r, size);   /* no record slot: put it back, fall through to the usual error */
      } }

    /* Initialize bump pointer on first call */
    if (g_sys_mem_bump_ptr == 0)
        g_sys_mem_bump_ptr = SYS_MEM_ALLOC_BASE;

    /* Align bump pointer */
    g_sys_mem_bump_ptr = VM_ALIGN_UP(g_sys_mem_bump_ptr, alignment);

    /* Check if we have room; spill into the overflow window once */
    if (g_sys_mem_bump_ptr < SYS_MEM_OVERFLOW_BASE &&
        g_sys_mem_bump_ptr + size > SYS_MEM_ALLOC_END) {
        fprintf(stderr, "[sys_memory] primary window exhausted at 0x%08X -- continuing in 0x%08X..0x%08X\n",
                g_sys_mem_bump_ptr, SYS_MEM_OVERFLOW_BASE, sys_mem_overflow_end());
        g_sys_mem_bump_ptr = SYS_MEM_OVERFLOW_BASE;
    }
    if (g_sys_mem_bump_ptr + size > (g_sys_mem_bump_ptr >= SYS_MEM_OVERFLOW_BASE
                                     ? sys_mem_overflow_end() : SYS_MEM_ALLOC_END)) {
        bump_unlock();
        fprintf(stderr, "[sys_memory] allocate(size=0x%X) -> ENOMEM: window exhausted (bump 0x%08X)\n",
                size, g_sys_mem_bump_ptr);
        return (int64_t)(int32_t)CELL_ENOMEM;
    }

    /* Find a free allocation slot */
    int slot = -1;
    for (int i = 0; i < SYS_MEMORY_ALLOC_MAX; i++) {
        if (!g_sys_mem_allocs[i].active) { slot = i; break; }
    }
    if (slot < 0) {
        bump_unlock();
        fprintf(stderr, "[sys_memory] allocate(size=0x%X) -> ENOMEM: all %d allocation slots in use\n",
                size, SYS_MEMORY_ALLOC_MAX);
        return (int64_t)(int32_t)CELL_ENOMEM;
    }

    /* SYS_MEM_TRACE=<hex bump>: once the bump pointer passes this, log the guest
     * call chain of every 16th allocation -- who keeps growing the heap. */
    { static uint32_t s_tr = 0xFFFFFFFFu; static unsigned s_trn = 0;
      if (s_tr == 0xFFFFFFFFu) { const char* e = getenv("SYS_MEM_TRACE"); s_tr = e ? (uint32_t)strtoul(e, 0, 16) : 0; }
      if (s_tr && g_sys_mem_bump_ptr >= s_tr && (s_trn++ % 16) == 0 && s_trn < 16 * 400) {
          uint32_t sp = (uint32_t)ctx->gpr[1];
          fprintf(stderr, "[memtrace] size=0x%X lr=%08X chain:", size, (uint32_t)ctx->lr);
          for (int k = 0; k < 16 && sp && sp < 0xF0000000u; k++) {
              sp = __builtin_bswap32(*(uint32_t*)vm_to_host(sp + 4));
              if (!sp || sp >= 0xF0000000u) break;
              fprintf(stderr, " %08X", __builtin_bswap32(*(uint32_t*)vm_to_host(sp + 0x14)));
          }
          fprintf(stderr, "\n");
      } }
    uint32_t alloc_addr = g_sys_mem_bump_ptr;
    g_sys_mem_bump_ptr += size;
    s_total_allocated += size;

    sys_mem_alloc_info* a = &g_sys_mem_allocs[slot];
    a->active       = 1;       /* claim the slot before unlocking */
    a->addr         = alloc_addr;
    a->size         = size;
    a->container_id = 0;
    a->page_size    = alignment;  /* 0x100000 (1M) or 0x10000 (64K) */

    bump_unlock();

    /* Commit the pages (window is outside the pre-committed main region);
     * fresh commits are already zeroed by the OS */
    if (vm_commit(alloc_addr, size) != CELL_OK) {
        a->active = 0;
        return (int64_t)(int32_t)CELL_ENOMEM;
    }

    if (ps3_log_verbose())
        fprintf(stderr, "[sys_memory] allocate -> 0x%08X (live=%u MB)\n", alloc_addr, s_total_allocated >> 20);

    if (addr_out != 0) {
        write_be32(addr_out, alloc_addr);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_memory_free
 *
 * r3 = address
 * -----------------------------------------------------------------------*/
int64_t sys_memory_free(ppu_context* ctx)
{
    uint32_t addr = LV2_ARG_U32(ctx, 0);

    bump_lock();
    for (int i = 0; i < SYS_MEMORY_ALLOC_MAX; i++) {
        if (g_sys_mem_allocs[i].active && g_sys_mem_allocs[i].addr == addr) {
            uint32_t size = g_sys_mem_allocs[i].size;
            s_total_allocated -= size;
            g_sys_mem_allocs[i].active = 0;
            sys_mem_release(addr, size);
            bump_unlock();
            { static unsigned long _n = 0; if (_n++ < 40 || (_n % 200) == 0)
                fprintf(stderr, "[sys_memory] free(0x%08X) size=0x%X (live=%u MB, free %u MB in %d blocks)\n",
                        addr, size, s_total_allocated >> 20, s_free_bytes >> 20, s_free_n); }
            return CELL_OK;
        }
    }
    bump_unlock();
    { static int _n = 0; if (_n < 8 && ++_n) fprintf(stderr, "[sys_memory] free(0x%08X): not an allocation -> EINVAL\n", addr); }

    return (int64_t)(int32_t)CELL_EINVAL;
}

/* ---------------------------------------------------------------------------
 * sys_memory_get_user_memory_size
 *
 * r3 = pointer to output struct: { u32 total, u32 available }
 * -----------------------------------------------------------------------*/
int64_t sys_memory_get_user_memory_size(ppu_context* ctx)
{
    uint32_t out_addr = LV2_ARG_PTR(ctx, 0);

    fprintf(stderr, "[sys_memory] get_user_memory_size()\n");
    if (out_addr != 0) {
        uint32_t total = SYS_MEM_USER_TOTAL;
        uint32_t avail = total - s_total_allocated;
        write_be32(out_addr + 0, total);
        write_be32(out_addr + 4, avail);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_memory_get_page_attribute (syscall 351 / 0x15F)
 *
 * r3 = address to query
 * r4 = pointer to sys_page_attribute_t:
 *        +0x00 attribute   (u64)
 *        +0x08 access_right(u64)
 *        +0x10 page_size   (u32)   <- the field titles actually check
 *        +0x14 pad         (u32)
 *
 * Was unimplemented (fell through to the return-0 stub), so the output struct
 * was never filled -> callers reading page_size got stack garbage. Dantelion2's
 * graphics init (func_009F6D40) queries a 1MB-page buffer and only proceeds to
 * construct a sub-object when page_size == 0x100000; the stub made it take the
 * failure path, leaving that sub-object NULL -> later null-deref crash.
 * -----------------------------------------------------------------------*/
int64_t sys_memory_get_page_attribute(ppu_context* ctx)
{
    uint32_t addr     = LV2_ARG_U32(ctx, 0);
    uint32_t attr_out = LV2_ARG_PTR(ctx, 1);

    /* Report the page size of the allocation containing addr. */
    uint32_t page_size = 0x10000;  /* default 64K */
    for (int i = 0; i < SYS_MEMORY_ALLOC_MAX; i++) {
        if (g_sys_mem_allocs[i].active &&
            addr >= g_sys_mem_allocs[i].addr &&
            addr <  g_sys_mem_allocs[i].addr + g_sys_mem_allocs[i].size) {
            page_size = g_sys_mem_allocs[i].page_size ? g_sys_mem_allocs[i].page_size : 0x10000;
            break;
        }
    }

    if (attr_out != 0) {
        write_be32(attr_out + 0x00, 0);          /* attribute    (hi) */
        write_be32(attr_out + 0x04, 0x00040000); /* attribute    (lo) PROT_READ_WRITE-ish */
        write_be32(attr_out + 0x08, 0);          /* access_right (hi) */
        write_be32(attr_out + 0x0C, 0x00000008); /* access_right (lo) PPU_THREAD-ish */
        write_be32(attr_out + 0x10, page_size);  /* page_size */
        write_be32(attr_out + 0x14, 0);          /* pad */
    }
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_memory_container_create
 *
 * r3 = pointer to receive container ID (u32*)
 * r4 = size
 * -----------------------------------------------------------------------*/
int64_t sys_memory_container_create(ppu_context* ctx)
{
    uint32_t id_out_addr = LV2_ARG_PTR(ctx, 0);
    uint32_t size        = LV2_ARG_U32(ctx, 1);

    int slot = -1;
    for (int i = 0; i < SYS_MEMORY_CONTAINER_MAX; i++) {
        if (!g_sys_mem_containers[i].active) { slot = i; break; }
    }
    if (slot < 0)
        return (int64_t)(int32_t)CELL_EAGAIN;

    sys_mem_container_info* c = &g_sys_mem_containers[slot];
    c->active     = 1;
    c->total_size = size;
    c->used_size  = 0;

    uint32_t container_id = (uint32_t)(slot + 1);
    if (id_out_addr != 0) {
        write_be32(id_out_addr, container_id);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_memory_container_destroy
 *
 * r3 = container_id
 * -----------------------------------------------------------------------*/
int64_t sys_memory_container_destroy(ppu_context* ctx)
{
    uint32_t container_id = LV2_ARG_U32(ctx, 0);

    if (container_id == 0 || container_id > SYS_MEMORY_CONTAINER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_mem_container_info* c = &g_sys_mem_containers[container_id - 1];
    if (!c->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    c->active = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_memory_container_get_size
 *
 * r3 = container_id
 * r4 = pointer to output struct: { u32 total, u32 used }
 * -----------------------------------------------------------------------*/
int64_t sys_memory_container_get_size(ppu_context* ctx)
{
    uint32_t container_id = LV2_ARG_U32(ctx, 0);
    uint32_t out_addr     = LV2_ARG_PTR(ctx, 1);

    if (container_id == 0 || container_id > SYS_MEMORY_CONTAINER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_mem_container_info* c = &g_sys_mem_containers[container_id - 1];
    if (!c->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (out_addr != 0) {
        write_be32(out_addr + 0, c->total_size);
        write_be32(out_addr + 4, c->used_size);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_mmapper_allocate_address
 *
 * r3 = size
 * r4 = flags
 * r5 = alignment
 * r6 = pointer to receive address (u32*)
 * -----------------------------------------------------------------------*/
/* Mapper reservation and legacy direct mapping share their own cursor.
 * Called with the allocator lock held. */
static uint32_t mapper_address(uint64_t requested, uint64_t alignment)
{
    if (!requested || requested > UINT32_MAX || !alignment ||
        alignment > UINT32_MAX || (alignment & (alignment - 1u))) return 0;
    uint64_t addr = ((uint64_t)s_mapper_bump_ptr + alignment - 1u) & ~(alignment - 1u);
    uint64_t size = (requested + alignment - 1u) & ~(alignment - 1u);
    /* A reservation owns its entire aligned span. Recheck after every skip:
     * an alignment adjustment can move it into another runtime-owned window. */
    const uint64_t reserved[][2] = {
        { VM_PPU_TLS_BASE, (uint64_t)VM_PPU_TLS_BASE + VM_PPU_TLS_SIZE },
        { SYS_MEM_ALLOC_BASE, SYS_MEM_ALLOC_END },                   /* sys_memory_allocate's windows */
        { SYS_MEM_OVERFLOW_BASE, sys_mem_overflow_end() },
        { VM_HLE_HEAP_BASE, (uint64_t)VM_HLE_HEAP_BASE + VM_HLE_HEAP_SIZE },
        { VM_AUDIO_BASE, (uint64_t)VM_AUDIO_BASE + VM_AUDIO_SIZE },
        { VM_SYS_VM_BASE, (uint64_t)VM_SYS_VM_BASE + VM_SYS_VM_SIZE },
        { VM_HLE_INJECT_BASE, (uint64_t)VM_HLE_INJECT_BASE + VM_HLE_INJECT_SIZE }
    };
    for (;;) {
        uint64_t next = s_image_query ? s_image_query(addr, addr + size) : addr;
        for (size_t i = 0; i < sizeof(reserved) / sizeof(reserved[0]); ++i)
            if (addr < reserved[i][1] && reserved[i][0] < addr + size &&
                reserved[i][1] > next) next = reserved[i][1];
        if (next == addr) break;
        addr = (next + alignment - 1u) & ~(alignment - 1u);
    }
    if (!size || addr + size > SYS_MAPPER_END || vm_commit((uint32_t)addr, (uint32_t)size) != CELL_OK)
        return 0;
    s_mapper_bump_ptr = (uint32_t)(addr + size);
    return (uint32_t)addr;
}

int64_t sys_mmapper_allocate_address(ppu_context* ctx)
{
    uint64_t size = LV2_ARG_U64(ctx, 0);
    uint64_t alignment = LV2_ARG_U64(ctx, 2);
    uint32_t out = LV2_ARG_PTR(ctx, 3);
    if (!alignment) alignment = 0x10000u;
    if (alignment > UINT32_MAX || (alignment & (alignment - 1u)) || !size || size > UINT32_MAX)
        return (int64_t)(int32_t)CELL_EINVAL;
    bump_lock();
    int slot = -1;
    for (int i = 0; i < SYS_MEMORY_ALLOC_MAX; ++i)
        if (!s_mapper_reservations[i].start) { slot = i; break; }
    uint32_t addr = slot < 0 ? 0 : mapper_address(size, alignment);
    if (addr) {
        mapper_reservation* r = &s_mapper_reservations[slot];
        r->start = r->next = addr; r->end = s_mapper_bump_ptr;
    }
    bump_unlock();
    if (!addr) return (int64_t)(int32_t)CELL_ENOMEM;
    if (out) write_be32(out, addr);
    return CELL_OK;
}

/* Claim a mapped block inside the caller's existing reservation. The loader's
 * search-and-map bridge and the allocator share this ownership bookkeeping,
 * so an unsuccessful request leaves all reservation cursors unchanged. */
int64_t sys_mmapper_search_map_address(uint32_t start, uint32_t size, uint32_t* out)
{
    if (!out || !size) return (int64_t)(int32_t)CELL_EINVAL;
    bump_lock();
    mapper_reservation* found = NULL;
    for (int i = 0; i < SYS_MEMORY_ALLOC_MAX; ++i) {
        mapper_reservation* r = &s_mapper_reservations[i];
        if (r->start && ((!start && !found) || (start >= r->start && start < r->end))) {
            found = r; if (start) break;
        }
    }
    if (!found) {
        bump_unlock(); return (int64_t)(int32_t)CELL_EINVAL;
    }
    uint64_t addr = found->next;
    if (start > addr) addr = start;
    addr = (addr + 0xFFFFFu) & ~0xFFFFFull;
    uint64_t end = (addr + size + 0xFFFFFu) & ~0xFFFFFull;
    if (end > found->end || vm_commit((uint32_t)addr, size) != CELL_OK) {
        bump_unlock(); return (int64_t)(int32_t)CELL_ENOMEM;
    }
    found->next = (uint32_t)end;
    *out = (uint32_t)addr;
    bump_unlock(); return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_mmapper_free_address
 *
 * r3 = address
 * -----------------------------------------------------------------------*/
int64_t sys_mmapper_free_address(ppu_context* ctx)
{
    /* uint32_t addr = LV2_ARG_U32(ctx, 0); */
    (void)ctx;
    /* In our bump allocator we don't actually free, just acknowledge */
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_mmapper_allocate_shared_memory
 *
 * r3 = key
 * r4 = size
 * r5 = flags
 * r6 = pointer to receive shared mem ID (u32*)
 * -----------------------------------------------------------------------*/
int64_t sys_mmapper_allocate_shared_memory(ppu_context* ctx)
{
    uint64_t key      = LV2_ARG_U64(ctx, 0);
    uint32_t size     = LV2_ARG_U32(ctx, 1);
    /* uint32_t flags = LV2_ARG_U32(ctx, 2); */
    uint32_t id_out   = LV2_ARG_PTR(ctx, 3);

    int slot = -1;
    for (int i = 0; i < SYS_MMAPPER_SHARED_MAX; i++) {
        if (!g_sys_mmapper_shared[i].active) { slot = i; break; }
    }
    if (slot < 0)
        return (int64_t)(int32_t)CELL_EAGAIN;

    sys_mmapper_shared_info* s = &g_sys_mmapper_shared[slot];
    s->active = 1;
    s->size   = VM_ALIGN_UP(size, VM_PAGE_SIZE);
    s->key    = key;
    s->addr   = 0;

    uint32_t shm_id = (uint32_t)(slot + 1);
    if (id_out != 0) {
        write_be32(id_out, shm_id);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_mmapper_allocate_memory (lv2 341)
 *
 * r3 = pointer to receive the allocated address (u32*)
 * r4 = size
 *
 * The ABI here was read off the caller rather than a header. GT5P's PDI device
 * worker calls this the moment it starts -- func_00941808 loads r3 with a
 * pointer and r4 with 0x300000, issues the syscall, retries on any non-zero
 * result, and on success immediately does `lwz r29, 0(r29)`, reading back
 * through the pointer it passed in r3.
 *
 * Answering CELL_OK without writing anything is the worst of both worlds: the
 * worker believes it has three megabytes, reads a null buffer back, and exits
 * on the spot -- which leaves the packed-filesystem device with no worker at
 * all and stalls asset loading. Answering an error is worse still, because the
 * caller retries forever.
 *
 * The flat VM has no distinction between reserving and mapping, so this hands
 * back committed guest memory directly and sys_mmapper_allocate_memory_from_
 * container below accepts it as already mapped.
 * -----------------------------------------------------------------------*/
int64_t sys_mmapper_allocate_memory(ppu_context* ctx)
{
    uint32_t addr_out = LV2_ARG_PTR(ctx, 0);
    uint32_t size     = LV2_ARG_U32(ctx, 1);

    if (size == 0)
        return (int64_t)(int32_t)CELL_EINVAL;

    bump_lock();
    uint32_t addr = mapper_address(size, VM_PAGE_SIZE);
    bump_unlock();
    if (!addr) return (int64_t)(int32_t)CELL_ENOMEM;

    if (addr_out != 0)
        write_be32(addr_out, addr);

    { static int n = 0; if (PS3_LOG_FIRST(n, 4))
        fprintf(stderr, "[sys_memory] mmapper_allocate_memory size=0x%X -> 0x%08X\n",
                size, addr); }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_mmapper_allocate_memory_from_container (lv2 342)
 *
 * The same allocation charged to a container. GT5P calls it straight after 341
 * with the address 341 handed back, so with a flat VM there is nothing further
 * to do -- the memory is already committed and addressable.
 * -----------------------------------------------------------------------*/
int64_t sys_mmapper_allocate_memory_from_container(ppu_context* ctx)
{
    (void)ctx;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_mmapper_map_shared_memory
 *
 * r3 = address (where to map)
 * r4 = shared mem ID
 * r5 = flags
 * -----------------------------------------------------------------------*/
int64_t sys_mmapper_map_shared_memory(ppu_context* ctx)
{
    uint32_t addr   = LV2_ARG_U32(ctx, 0);
    uint32_t shm_id = LV2_ARG_U32(ctx, 1);
    /* uint32_t flags = LV2_ARG_U32(ctx, 2); */

    if (shm_id == 0 || shm_id > SYS_MMAPPER_SHARED_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_mmapper_shared_info* s = &g_sys_mmapper_shared[shm_id - 1];
    if (!s->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    /* Commit the memory at the specified address */
    int32_t rc = vm_commit(addr, s->size);
    if (rc != CELL_OK)
        return (int64_t)(int32_t)rc;

    s->addr = addr;
    memset(vm_to_host(addr), 0, s->size);

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Registration
 * -----------------------------------------------------------------------*/
void sys_memory_init(lv2_syscall_table* tbl)
{
    bump_lock();
    memset(g_sys_mem_allocs,     0, sizeof(g_sys_mem_allocs));
    memset(g_sys_mem_containers, 0, sizeof(g_sys_mem_containers));
    memset(g_sys_mmapper_shared, 0, sizeof(g_sys_mmapper_shared));
    g_sys_mem_bump_ptr = 0;
    s_total_allocated  = 0;
    s_free_n = 0;
    s_free_bytes = 0;
    s_mapper_bump_ptr = SYS_MAPPER_BASE;
    memset(s_mapper_reservations, 0, sizeof(s_mapper_reservations));
    bump_unlock();

    lv2_syscall_register(tbl, SYS_MEMORY_ALLOCATE,             sys_memory_allocate);
    lv2_syscall_register(tbl, SYS_MEMORY_FREE,                 sys_memory_free);
    lv2_syscall_register(tbl, SYS_MEMORY_GET_USER_MEMORY_SIZE, sys_memory_get_user_memory_size);
    lv2_syscall_register(tbl, SYS_MEMORY_GET_PAGE_ATTRIBUTE,   sys_memory_get_page_attribute);
    lv2_syscall_register(tbl, SYS_MEMORY_CONTAINER_CREATE,     sys_memory_container_create);
    lv2_syscall_register(tbl, SYS_MEMORY_CONTAINER_DESTROY,    sys_memory_container_destroy);
    lv2_syscall_register(tbl, SYS_MEMORY_CONTAINER_GET_SIZE,   sys_memory_container_get_size);
    lv2_syscall_register(tbl, SYS_MMAPPER_ALLOCATE_ADDRESS,    sys_mmapper_allocate_address);
    lv2_syscall_register(tbl, SYS_MMAPPER_FREE_ADDRESS,        sys_mmapper_free_address);
    lv2_syscall_register(tbl, SYS_MMAPPER_ALLOCATE_SHARED_MEMORY, sys_mmapper_allocate_shared_memory);
    lv2_syscall_register(tbl, SYS_MMAPPER_MAP_SHARED_MEMORY,  sys_mmapper_map_shared_memory);
    lv2_syscall_register(tbl, 341, sys_mmapper_allocate_memory);
    lv2_syscall_register(tbl, 342, sys_mmapper_allocate_memory_from_container);
}
