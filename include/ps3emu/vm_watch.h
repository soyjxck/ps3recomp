/*
 * ps3recomp - write-watch on guest memory (Windows)
 *
 * The draw engine keeps a host copy of every guest texture it has uploaded
 * and has to notice when the title writes the guest bytes. Hashing each
 * cached texture's bytes every frame was a third of the RSX walker's time
 * on Drakengard 3 and the frames where many textures were bound hitched on
 * it. This does what RPCS3's texture cache does instead: the 4 KiB pages
 * under a cached texture are made read-only, a write faults, the vectored
 * handler marks the page written and makes it writable again, and the
 * engine only hashes a texture whose pages were written since it last did.
 *
 * A page is re-protected by the next arm, unless it was written in the last
 * couple of seconds: a page the title writes every frame (a dynamic
 * texture, or data sharing a page with one) would fault every frame, which
 * costs more than the hash does, so such pages are left unwatched and the
 * texture is hashed as before until they go quiet. The same quarantine
 * covers the one thing a fault cannot catch: a write the kernel makes for
 * a host call (ReadFile into guest memory fails on a read-only page instead
 * of faulting), so the host calls vm_watch_touch before those.
 *
 * Off Windows and in the replay harness nothing is initialised and
 * vm_watch_available() is 0: the engine hashes as it always did.
 */
#ifndef PS3EMU_VM_WATCH_H
#define PS3EMU_VM_WATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* main: after the guest VM is reserved. */
void     vm_watch_init(uint8_t* base, uint64_t size);
int      vm_watch_available(void);
/* Protect the pages under [ea, ea+len) that can be protected and return a
 * stamp that changes when any of them is written. *unwatched receives how
 * many of the pages are not protected (recently written, or could not be):
 * a caller with unwatched pages cannot trust the stamp and hashes. */
uint64_t vm_watch_arm(uint32_t ea, uint32_t len, uint32_t* unwatched);
/* The host is about to write [ea, ea+len) with a kernel call. */
void     vm_watch_touch(uint32_t ea, uint32_t len);
/* From the vectored exception handler: a fault at host address `addr`.
 * 1 when it was a watched page (now writable, marked written). */
int      vm_watch_fault(uintptr_t addr, int is_write);
void     vm_watch_stats(uint64_t* faults, uint64_t* pages_protected);
/* Commit the RESERVED pages of host range [p, p+n) read-write and leave the
 * committed ones alone; 1 on success. Every commit of guest memory goes
 * through this: on Windows, MEM_COMMIT over a page that is already committed
 * resets its protection, which would silently re-open a watched page. */
int      vm_commit_reserved(void* p, uint64_t n);

#ifdef __cplusplus
}
#endif
#endif /* PS3EMU_VM_WATCH_H */
