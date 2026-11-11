"""Exercise the lifetime of LAMMPS Kokkos sanity instances."""
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import easybuild.easyblocks.l.lammps as lammps
from easybuild.easyblocks.generic.cmakemake import CMakeMake
from easybuild.easyblocks.l.lammps import EB_LAMMPS, translate_lammps_version


class _Config(dict):
    """Supply the toolchain and parallel properties without initializing a build."""


class LammpsKokkosCleanup(unittest.TestCase):
    """Release the instance before process-wide Kokkos resources."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('atm', 'balance', 'msst'):
            path = self.root / 'examples' / name / ('in.' + name)
            path.parent.mkdir(parents=True)
            path.write_text('run 100\n')

    def commands(self, version='30Sep2026', cuda=True, kokkos=True):
        block = object.__new__(EB_LAMMPS)
        block.cfg = _Config(version=version, sanity_check_test_inputs=['atm', 'balance', 'msst'], kokkos=kokkos)
        block.cfg.parallel = 8
        mpi = mock.Mock(side_effect=lambda command, count: 'mpirun -n %s %s' % (count, command))
        block.cfg.toolchain = SimpleNamespace(options={'usempi': True}, mpi_cmd_for=mpi)
        block.cur_version = translate_lammps_version(version)
        block.installdir = str(self.root)
        block.cuda = cuda
        block.nvidia_gpu_found = True
        block.dry_run = False
        block.log = mock.Mock()
        with mock.patch.object(lammps.tempfile, 'mkdtemp', return_value=str(self.root / 'execution')), \
                mock.patch.object(lammps, 'which', return_value='/usr/bin/nvidia-smi'), \
                mock.patch.object(lammps, 'get_avail_core_count', return_value=8), \
                mock.patch.object(lammps, 'get_software_version', return_value=None), \
                mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                mock.patch.object(CMakeMake, 'sanity_check_step') as base:
            EB_LAMMPS.sanity_check_step(block)
        return base.call_args.kwargs['custom_commands'], mpi

    def kokkos_code(self, **kwargs):
        commands, mpi = self.commands(**kwargs)
        code = [shlex.split(command)[-1] for command in commands if '"-sf", "kk"' in command]
        self.assertEqual(len(code), 1)
        self.assertEqual(mpi.call_count, len(commands))
        for call in mpi.call_args_list:
            self.assertEqual(call.args[1], 4)
        return code[0]

    def execute(self, code, fail_file=False):
        events = []

        class Instance:
            def __init__(self, cmdargs=None):
                events.append(('open', cmdargs))
                self.lib = SimpleNamespace(
                    lammps_kokkos_finalize=lambda: events.append(('kokkos_finalize',)),
                    lammps_mpi_finalize=lambda: events.append(('mpi_finalize',)),
                    lammps_plugin_finalize=lambda: events.append(('plugin_finalize',)),
                )

            def file(self, path):
                events.append(('file', path))
                if fail_file:
                    raise RuntimeError('simulation failure')

            def close(self):
                events.append(('close',))

            def finalize(self):
                events.append(('full_finalize',))

        with mock.patch.dict(sys.modules, {'lammps': SimpleNamespace(lammps=Instance)}):
            exec(code, {})
        return events

    def test_modern_gpu_closes_instance_before_kokkos_and_retains_four_mpi_ranks(self):
        code = self.kokkos_code()
        self.assertEqual(self.execute(code), [
            ('open', ['-sf', 'kk', '-k', 'on', 'g', '1']),
            ('file', str(self.root / 'examples' / 'msst' / 'in.msst')),
            ('close',), ('kokkos_finalize',), ('mpi_finalize',), ('plugin_finalize',),
        ])

    def test_modern_cpu_kokkos_uses_the_same_resource_order(self):
        code = self.kokkos_code(cuda=False)
        self.assertEqual(self.execute(code), [
            ('open', ['-sf', 'kk', '-k', 'on']),
            ('file', str(self.root / 'examples' / 'msst' / 'in.msst')),
            ('close',), ('kokkos_finalize',), ('mpi_finalize',), ('plugin_finalize',),
        ])

    def test_first_modern_release_preserves_external_python_lifetime(self):
        code = self.kokkos_code(version='22Jul2025')
        self.assertEqual(self.execute(code)[2:], [
            ('close',), ('kokkos_finalize',), ('mpi_finalize',), ('plugin_finalize',),
        ])

    def test_legacy_release_retains_full_finalize(self):
        code = self.kokkos_code(version='29Aug2024')
        self.assertEqual(self.execute(code)[2:], [('full_finalize',)])

    def test_release_without_finalize_requirement_keeps_its_command(self):
        code = self.kokkos_code(version='29Oct2020')
        self.assertEqual(len(self.execute(code)), 2)

    def test_all_default_input_files_remain_in_the_sanity_commands(self):
        commands, _ = self.commands()
        for name, command in zip(('atm', 'balance', 'msst'), commands):
            code = shlex.split(command)[-1]
            self.assertEqual(self.execute(code), [
                ('open', None), ('file', str(self.root / 'examples' / name / ('in.' + name))),
            ])

    def test_other_accelerators_do_not_invoke_kokkos_cleanup(self):
        commands, _ = self.commands(kokkos=False)
        self.assertFalse(any('lammps_kokkos_finalize' in command for command in commands))
        self.assertTrue(any('"-sf", "gpu"' in command for command in commands))

    def test_simulation_failure_is_propagated(self):
        with self.assertRaisesRegex(RuntimeError, 'simulation failure'):
            self.execute(self.kokkos_code(), fail_file=True)


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(LammpsKokkosCleanup)


if __name__ == '__main__':
    unittest.main()
