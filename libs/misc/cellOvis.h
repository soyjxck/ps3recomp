/*
 * ps3recomp - cellOvis HLE
 *
 * SPU overlay support (libovis). A title that runs overlaid SPU programs --
 * PhysX on Drakengard 3 is one -- asks for the size of an ELF's overlay table,
 * allocates it, then has the library fill it in. Static recompilation lifts
 * SPU programs whole, so there are no overlays to manage: the table is empty.
 */

#ifndef PS3RECOMP_CELL_OVIS_H
#define PS3RECOMP_CELL_OVIS_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Functions -- the real libovis signatures. Note GetOverlayTableSize RETURNS
 * the size (an int) rather than writing it through a pointer, and the library
 * has no init call. */
s32 cellOvisGetOverlayTableSize(u32 elf_ea);
s32 cellOvisInitializeOverlayTable(u32 ea_ovly_table, u32 elf_ea);
void cellOvisFixSpuSegments(u32 r);
void cellOvisInvalidateOverlappedSegments(u32 r, u32 num);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_OVIS_H */
