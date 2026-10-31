#!/usr/bin/env python3
"""Real launch/IRQ/guard preservation, with optional temporary bug controls."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SPU = ROOT / 'runtime/spu'
UNITS = ['spu_channels', 'spu_lockstep', 'spu_coherency', 'spu_workload',
         'spurs_policy', 'spurs_job', 'spu_tsp_weak', 'spu_vm_pagemap',
         'spurs_policy_blob_weak', 'spu_interp']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mutants', action='store_true')
args = parser.parse_args()
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('SPU_', 'SPURS_', 'YDKJ_', 'LBP_', 'GH3_'))}
# CI has already configured its compiler matrix before discovering this suite.
# Honor explicit overrides, then use that configured compiler rather than `cc`
# (which would silently run GCC in both Linux matrix entries).
cc = os.environ.get('CC')
cache = ROOT / 'build/CMakeCache.txt'
if not cc and cache.is_file():
    match = re.search(r'^CMAKE_C_COMPILER:[^=]+=([^\r\n]+)$', cache.read_text(), re.M)
    if match:
        configured = match[1]
        # clang-cl uses MSVC driver flags; clang shares its backend/target and
        # accepts the portable flags used by this standalone fixture.
        if Path(configured).stem.lower() == 'clang-cl':
            configured = str(Path(configured).with_name('clang' + Path(configured).suffix))
        cc = configured
if not cc:
    cc = shutil.which('clang') if os.name == 'nt' else 'cc'
assert cc, 'a C compiler is required'
print('drain preservation compiler:', cc, flush=True)


def depth_history(result):
    found = re.search(r'\[spu-depth\] recent drain sites[^:]*:([^\n]*)', result.stderr)
    assert found, result.stdout + result.stderr
    return [int(pc, 16) for pc in found[1].split()]


with tempfile.TemporaryDirectory(prefix='spu-drain-launch-') as td:
    tmp = Path(td)

    def run(name, mutation=None, *, sanitize=False, diagnostics=False, debug=False,
            lockstep=False, depth_only=False):
        # A bug control edits one file of a copy of runtime/spu: the copies
        # include each other first ("spu_context.h" next to the includer), and
        # their ../ includes fall back to the real tree through -I SPU.
        spu = SPU
        if mutation:
            spu = tmp / (name + '-src') / 'runtime/spu'
            (spu / 'tests').mkdir(parents=True)
            for f in SPU.iterdir():
                if f.is_file():
                    shutil.copyfile(f, spu / f.name)
            shutil.copyfile(SPU / 'tests/test_spu_drain_launch.c', spu / 'tests/test_spu_drain_launch.c')
            file, old, new = mutation
            text = (spu / file).read_bytes()
            assert text.count(old.encode()) == 1, (name, file, text.count(old.encode()))
            (spu / file).write_bytes(text.replace(old.encode(), new.encode()))
        exe = tmp / (name + ('.exe' if os.name == 'nt' else ''))
        flags = ['-std=gnu17', '-O0' if debug else '-O2', '-UNDEBUG',
                 '-I' + str(ROOT / 'include'), '-I' + str(SPU),
                 '-I' + str(ROOT / 'runtime/platform')]
        if sanitize:
            flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
        command = [cc, *flags, str(spu / 'tests/test_spu_drain_launch.c'), str(spu / 'spu_drain.c'),
                   *[str(spu / (unit + '.c')) for unit in UNITS],
                   str(ROOT / 'runtime/platform/win32_compat.c'),
                   str(ROOT / 'runtime/platform/env_cache.c'),
                   str(ROOT / 'runtime/spu/tests/fork_runtime_standins.c'),
                   *([] if os.name == 'nt' else ['-pthread', '-lm']), '-o', str(exe)]
        built = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert built.returncode == 0, built.stdout + built.stderr
        testenv = dict(env)
        if diagnostics:
            testenv.update(SPU_STEPTRACE='906', SPU_EXITTRACE='906')
        if lockstep:
            testenv.update(SPU_LOCKSTEP='1', SPU_LOCKSTEP_QUANTUM='8')
        mode = ['lockstep'] if lockstep else []
        if depth_only:
            mode = ['depth']
            testenv['SPU_HOST_DEPTH_MAX'] = '8'
        result = subprocess.run([str(exe), *mode], env=testenv, capture_output=True,
                                text=True, timeout=15)
        print(name, result.returncode, result.stdout.strip(), flush=True)
        if name == 'o2':
            cases = [('8', 'depth', [0xA010, 0xA014, 0xA01C, 0xA020]),
                     ('0x8', 'depth', [0xA010, 0xA014, 0xA01C, 0xA020]),
                     ('0', 'depth-small', [0xA000]),
                     ('1', 'depth-small', [0xA000, 0xA004]),
                     (None, 'depth-default', [0xAFA0, 0xAFA4, 0xBF3C, 0xBF40])]
            for limit, argument, want in cases:
                depthenv = dict(env)
                if limit is not None:
                    depthenv['SPU_HOST_DEPTH_MAX'] = limit
                probe = subprocess.run([str(exe), argument], env=depthenv,
                                       capture_output=True, text=True, timeout=15)
                assert probe.returncode == 0, probe.stdout + probe.stderr
                assert depth_history(probe) == want, probe.stderr
                print('PASS depth ring, limit=' + str(limit), flush=True)
        return result

    modes = [('o2', {}), ('debug', {'debug': True}),
             ('diagnostics', {'diagnostics': True}), ('lockstep', {'lockstep': True})]
    if os.name != 'nt':
        modes.append(('ubsan', {'sanitize': True}))
    for name, options in modes:
        result = run(name, **options)
        assert result.returncode == 0, result.stdout + result.stderr
        if name == 'diagnostics':
            assert '[step]' in result.stderr and '[spu-exit]' in result.stderr, result.stderr
    if args.mutants:
        # Each temporary edit removes a real side effect; expected values are
        # literal history/steps/register assertions in the compiled fixture.
        # The fork's hop runs inline from spu_context.h (spu_hop) unless a
        # debug switch sends it through spu_drain.c's full check, and the
        # history is the context's own ring (dbg_recent).
        controls = {
            'history': ('spu_context.h',
                        'ctx->dbg_recent[ctx->dbg_recent_n++ & 31] = (uint32_t)ctx->pc & SPU_LS_MASK;', ''),
            'steps': ('spu_context.h', 'if (ctx->steps++ && (ctx->pc & SPU_LS_MASK) == 0 &&',
                      'if (ctx->steps && (ctx->pc & SPU_LS_MASK) == 0 &&'),
            'reset': ('spu_context.h', 'spu_check_stack_reset(ctx, tf);', '(void)tf;'),
            'irq': ('spu_drain.c', 'if (ctx->int_enable && (ctx->event_status & ctx->event_mask)) {',
                    'if (0) {'),
            'flow': ('spu_context.h', 'g_pm_flow_buf[g_pm_flow_n++] = ctx->pc;', '(void)ctx;'),
            'lockstep': ('spu_context.h',
                         'if (__builtin_expect(g_yz_lockstep_on != 0, 0)) yz_lockstep_tick(ctx);',
                         '(void)ctx;'),
        }
        for name, control in controls.items():
            result = run('mutant-' + name, control)
            assert result.returncode != 0 and 'FAIL:' in result.stdout, result.stdout + result.stderr
            print('PASS rejected bug control:', name, flush=True)
        result = run('mutant-half-ring', ('spu_drain.c', 'if (ctx->host_depth < s_max / 2) return;',
                                          'if (ctx->host_depth < s_max) return;'),
                     depth_only=True)
        assert result.returncode != 0 or depth_history(result) != [0xA010, 0xA014, 0xA01C, 0xA020]
        print('PASS rejected bug control: half-ring', flush=True)
    print('PASS real drain launch preservation', flush=True)
