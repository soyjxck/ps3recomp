/*
 * ps3recomp - cellOvis HLE implementation
 *
 * Every lifted SPU program is a whole image, so no title needs overlay tables:
 * report an empty one. This also matches what RPCS3 answers.
 *
 * The previous version invented an API -- cellOvisInit plus an out-parameter
 * for the size -- that libovis does not have. Titles never call the made-up
 * init, so GetOverlayTableSize returned CELL_OVIS_ERROR_NOT_INITIALIZED
 * (0x80410701), and since the real function returns a size, Drakengard 3's
 * PhysX took that as one: it asked malloc for 0x80410701 + 0x84 bytes.
 */

#include "cellOvis.h"
#include <stdio.h>

s32 cellOvisGetOverlayTableSize(u32 elf_ea)
{
    static int n = 0;
    if (n++ < 4) printf("[cellOvis] GetOverlayTableSize(elf=0x%08X) -> 0\n", elf_ea);
    return 0;
}

s32 cellOvisInitializeOverlayTable(u32 ea_ovly_table, u32 elf_ea)
{
    static int n = 0;
    if (n++ < 4) printf("[cellOvis] InitializeOverlayTable(table=0x%08X, elf=0x%08X)\n",
                        ea_ovly_table, elf_ea);
    return CELL_OK;
}

void cellOvisFixSpuSegments(u32 r)
{
    (void)r;
}

void cellOvisInvalidateOverlappedSegments(u32 r, u32 num)
{
    (void)r; (void)num;
}
