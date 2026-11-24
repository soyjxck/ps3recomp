#!/usr/bin/env python3
"""Check that the Metal backend's rectangle clears leave the rest of the target."""
import pathlib
import platform
import subprocess
import tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
if platform.system() != "Darwin":
    print("SKIP: Metal requires macOS")
    raise SystemExit(0)
with tempfile.TemporaryDirectory(prefix="metal-rect-clears-") as tmp:
    binary = pathlib.Path(tmp) / "test"
    # The translator is C++17 (ps3types.h); a .cpp on a plain clang line is
    # compiled as C++98, so it is built on its own line.
    translator = pathlib.Path(tmp) / "rsx_shader_msl.o"
    subprocess.run(["clang++", "-std=c++17", "-Iinclude", "-Ilibs/video", "-c",
                    "libs/video/rsx_shader_msl.cpp", "-o", str(translator)],
                   cwd=ROOT, check=True, timeout=120)
    # A clear that finds the record array full submits mid-frame, so the
    # clears reach the engine's commit, and through it the draw engine and a
    # few runtime hooks: those sources, and stand-ins for the rest
    # (metal_test_standins.c).
    subprocess.run(["clang", "-fobjc-arc", "-Iinclude", "-Ilibs/video",
                    "libs/video/tests/test_metal_rect_clears.m",
                    "libs/video/rsx_draw_engine.c", "libs/video/rsx_dispatch.c",
                    "libs/video/rsx_vertex_compact.c", "runtime/platform/env_cache.c",
                    "libs/video/tests/metal_test_standins.c",
                    "libs/video/rsx_texture_layout.c", "libs/video/rsx_vertex_fetch.c",
                    "libs/video/rsx_vp_decompiler.c", "libs/video/rsx_fp_decompiler.c",
                    "libs/video/rsx_tiling.c", str(translator), "-lc++", "-Wl,-dead_strip",
                    "-framework", "Metal", "-framework", "Cocoa",
                    "-framework", "QuartzCore", "-o", str(binary)],
                   cwd=ROOT, check=True, timeout=120)
    subprocess.run([str(binary)], check=True, timeout=20)
