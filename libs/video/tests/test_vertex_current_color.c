/* Explicit current vertex colors must survive disabled-array fetches.
 * What tells an explicit colour from an unwritten one is the write count on
 * ATTR3's first 4F slot, which the draw engine's method entry keeps
 * (rsx_draw_engine.c, eng_method_watched; rsx_dispatch_method does not count
 * writes). The engine is not linked here: set_current() below counts as it
 * does, and test_rsx_draw_engine.c checks the engine's own counting.
 * Build without a graphics device:
 *   cc -std=gnu17 -O2 -I include -I libs/video \
 *     libs/video/tests/test_vertex_current_color.c libs/video/rsx_vertex_compact.c \
 *     libs/video/rsx_dispatch.c -o /tmp/test_vertex_current_color
 */
#include "rsx_vertex_compact.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* rsx_dispatch.c reads it for RSX_CONST_LOG; rsx_draw_engine.c, not linked
 * here, defines it. */
uint32_t g_rsx_engine_frame;

static int failures;

/* A write of ATTR3's current value, as it arrives through the draw engine:
 * counted on the attribute's first 4F slot whatever the encoding, then stored. */
static void set_current(rsx_dispatch* rsx, u32 method, u32 value)
{
    rsx->seen[0x1C30u >> 2]++;
    rsx_dispatch_method(rsx, method, value);
}

static u32 float_bits(float value)
{
    u32 word;
    memcpy(&word, &value, sizeof(word));
    return word;
}

static void check_fetch(const char* name, rsx_dispatch* rsx, u32 attr,
                        const float expected[4])
{
    rsx_vertex_layout_plan layout;
    rsx_vertex_fetch_plan plan;
    const rsx_vertex_ref refs[2] = {{1, 0}, {5, 0}};
    float direct[4], one[4] = {0}, many[8] = {0};
    rsx_vertex_fetch_default(rsx, attr, direct);
    rsx_vertex_layout_plan_init(&layout, 1u << attr);
    rsx_vertex_fetch_plan_init(&plan, rsx, &layout, NULL, NULL);
    int ok = rsx_vertex_fetch_one(&plan, &refs[0], (u8*)one) &&
        rsx_vertex_fetch_many(&plan, refs, 2, (u8*)many);
    for (u32 i = 0; i < 4; i++) {
        if (direct[i] != expected[i] || one[i] != expected[i] || many[i] != expected[i] ||
            many[i + 4] != expected[i])
            ok = 0;
    }
    if (!ok) {
        printf("FAIL %s: got (%g,%g,%g,%g), expected (%g,%g,%g,%g)\n",
               name, one[0], one[1], one[2], one[3],
               expected[0], expected[1], expected[2], expected[3]);
        failures++;
    }
}

int main(void)
{
    rsx_dispatch* rsx = (rsx_dispatch*)calloc(1, sizeof(*rsx));
    if (!rsx)
        return 2;
    const float white[4] = {1, 1, 1, 1};
    const float black[4] = {0, 0, 0, 1};
    const float color[4] = {0.25f, 0.5f, 0.75f, 1};

    rsx_dispatch_init(rsx, NULL);
    check_fetch("unwritten color compatibility fallback", rsx, 3, white);
    check_fetch("unwritten non-color hardware default", rsx, 4, black);

    /* A shader can swizzle x into output alpha. Replacing this explicit
     * value with white makes transparent geometry opaque. */
    for (u32 i = 0; i < 4; i++)
        set_current(rsx, 0x1C30 + i * 4, float_bits(black[i]));
    check_fetch("explicit 4F color", rsx, 3, black);

    rsx_dispatch_init(rsx, NULL);
    set_current(rsx, 0x194C, 0xFF000000);
    check_fetch("explicit packed 4UB color", rsx, 3, black);

    /* Presence is independent of value, and a write to any component
     * establishes an explicit current attribute. */
    for (u32 lane = 0; lane < 4; lane++) {
        rsx_dispatch_init(rsx, NULL);
        set_current(rsx, 0x1C30 + lane * 4, float_bits(black[lane]));
        check_fetch("single-component explicit color", rsx, 3, black);
    }

    rsx_dispatch_init(rsx, NULL);
    for (u32 i = 0; i < 4; i++)
        set_current(rsx, 0x1C30 + i * 4, float_bits(color[i]));
    check_fetch("other explicit color", rsx, 3, color);
    free(rsx);
    printf("test_vertex_current_color: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
