"""Synthetic ELF and command tests for the selected ORCA MPI runtime paths."""
import os
import shlex
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import easybuild.tools.options as eboptions
from easybuild.easyblocks.generic.packedbinary import PackedBinary
from easybuild.easyblocks.o.orca import EB_ORCA, orca_elf_load_info, orca_permanent_rpaths
from easybuild.tools import config
from easybuild.tools.build_log import EasyBuildError


class OrcaMpiRpath(unittest.TestCase):
    """Preserve vendor paths and ELF safety without invoking proprietary binaries."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        options = eboptions.parse_options(args=[])
        config.init(options.options, options.get_options_by_section('config'))
        config.init_build_options(build_options={'extended_dry_run': False})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def elf(self, relative='bin/orca mpi', elf_type=3, segments=((0, 0x400000, 0x200000),)):
        """Write an ELF64 header and load table with no executable machine code."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        ident = b'\x7fELF' + bytes([2, 1, 1]) + b'\x00' * 9
        header = struct.pack('<HHIQQQIHHHHHH', elf_type, 62, 1, 0x400000, 64, 0, 0,
                             64, 56, len(segments), 0, 0, 0)
        program_headers = b''.join(struct.pack('<IIQQQQQQ', 1, 5, offset, vaddr, vaddr, 1, 1, align)
                                   for offset, vaddr, align in segments)
        path.write_bytes(ident + header + program_headers)
        return path

    def patch_commands(self, current, expected, dynamic='(RPATH)', recorded=None):
        """Supply tool output while keeping every command visible to assertions."""
        print_calls = []

        def command_output(command, **kwargs):
            arguments = shlex.split(command)
            if arguments[:2] == ['patchelf', '--print-rpath']:
                print_calls.append(command)
                output = current if len(print_calls) == 1 else (expected if recorded is None else recorded)
            elif arguments[:2] == ['readelf', '-d']:
                output = dynamic
            else:
                output = ''
            return SimpleNamespace(output=output)

        return mock.Mock(side_effect=command_output)

    def test_permanent_paths_keep_site_mpi_and_compiler_paths(self):
        """Probe paths retain selected runtime directories and reject build and relative entries."""
        entries = ':'.join(['$ORIGIN/lib', '', 'relative/lib', '/scratch/build/orca/lib',
                            '/scratch/build/orca-other/lib', '/scratch/source/lib', '/tmp/probe/lib',
                            '/site/OpenMPI/lib', '/site/GCCcore/lib64', '/site/OpenMPI/lib'])
        self.assertEqual(
            orca_permanent_rpaths(entries, ['/scratch/build/orca', '/scratch/source', '/tmp']),
            ['/scratch/build/orca-other/lib', '/site/OpenMPI/lib', '/site/GCCcore/lib64'],
        )

    def test_temporary_symlink_alias_is_filtered(self):
        """An absolute symlink alias cannot carry a build directory into the installed RPATH."""
        build = self.root / 'build'
        build.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(build, target_is_directory=True)
        self.assertEqual(orca_permanent_rpaths(str(alias / 'lib'), [str(build)]), [])

    def test_probe_uses_active_mpicc_and_flags_without_executing_the_result(self):
        """Only the MPI link and RPATH inspection run, with scratch directories removed."""
        build = self.root / 'build with spaces'
        build.mkdir()
        block = SimpleNamespace(builddir=str(build), cfg={'start_dir': str(self.root / 'source')})
        commands = mock.Mock(side_effect=[SimpleNamespace(output=''), SimpleNamespace(
            output='%s:/site/OpenMPI/lib:/site/GCCcore/lib64:$ORIGIN/lib:/tmp/probe/lib\n' % build)])
        environment = {'MPICC': 'mpicc -cc=gcc', 'CPPFLAGS': '-DORCA_MPI_PROBE=1',
                       'CFLAGS': '-O2', 'LDFLAGS': '-Wl,--as-needed', 'TMPDIR': '/scratch/job-tmp'}
        with mock.patch.dict(os.environ, environment), mock.patch(
                'easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            self.assertEqual(EB_ORCA._toolchain_mpi_rpath(block), ['/site/OpenMPI/lib', '/site/GCCcore/lib64'])
        self.assertEqual(commands.call_count, 2)
        command = shlex.split(commands.call_args_list[0][0][0])
        self.assertEqual(command[:2], ['mpicc', '-cc=gcc'])
        self.assertIn('-DORCA_MPI_PROBE=1', command)
        self.assertIn('-O2', command)
        self.assertIn('-Wl,--as-needed', command)
        self.assertIn(str(build / 'eb_orca_mpi_rpath.c'), command)
        self.assertEqual(command[command.index('-o') + 1], str(build / 'eb_orca_mpi_rpath'))
        self.assertEqual(commands.call_args_list[0][1]['work_dir'], str(build))
        self.assertEqual(shlex.split(commands.call_args_list[1][0][0]),
                         ['patchelf', '--print-rpath', str(build / 'eb_orca_mpi_rpath')])
        source = (build / 'eb_orca_mpi_rpath.c').read_text()
        self.assertIn('MPI_Init(&argc, &argv)', source)
        self.assertIn('MPI_Finalize()', source)

    def test_probe_rejects_missing_mpicc(self):
        environment = dict(os.environ)
        environment.pop('MPICC', None)
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                'easybuild.easyblocks.o.orca.run_shell_cmd') as command:
            with self.assertRaisesRegex(EasyBuildError, 'MPICC is required'):
                EB_ORCA._toolchain_mpi_rpath(SimpleNamespace())
        command.assert_not_called()

    def test_probe_rejects_paths_that_are_only_temporary(self):
        build = self.root / 'build'
        build.mkdir()
        block = SimpleNamespace(builddir=str(build), cfg={'start_dir': str(self.root / 'source')})
        commands = mock.Mock(side_effect=[SimpleNamespace(output=''),
                                         SimpleNamespace(output=str(build / 'lib') + ':$ORIGIN/lib')])
        with mock.patch.dict(os.environ, {'MPICC': 'mpicc'}), mock.patch(
                'easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            with self.assertRaisesRegex(EasyBuildError, 'did not provide permanent RPATH entries'):
                EB_ORCA._toolchain_mpi_rpath(block)

    def test_elf_load_information_reads_the_real_program_headers(self):
        path = self.elf(elf_type=2, segments=((0, 0x400000, 0x200000), (0x1000, 0x601000, 0x1000)))
        self.assertEqual(orca_elf_load_info(str(path)),
                         (2, [(0, 0x400000, 0x200000), (0x1000, 0x601000, 0x1000)]))

    def test_pie_rewrite_keeps_vendor_paths_and_large_load_alignment(self):
        """The patch retains $ORIGIN and vendor paths, quotes names, and requests the ELF alignment."""
        path = self.elf()
        expected = '$ORIGIN/lib:/vendor/lib:/site/OpenMPI/lib:/site/GCCcore/lib64'
        commands = self.patch_commands('$ORIGIN/lib:/vendor/lib', expected)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path),
                                     ['/site/OpenMPI/lib', '/vendor/lib', '/site/GCCcore/lib64'])
        patch = [shlex.split(call[0][0]) for call in commands.call_args_list if '--set-rpath' in call[0][0]]
        self.assertEqual(patch, [['patchelf', '--page-size', '2097152', '--force-rpath',
                                  '--set-rpath', expected, str(path)]])

    def test_fixed_address_rewrite_avoids_a_large_page_size(self):
        """Fixed-address images retain a valid mapped address without a large page-size request."""
        path = self.elf(elf_type=2)
        expected = '$ORIGIN/lib:/site/OpenMPI/lib'
        commands = self.patch_commands('$ORIGIN/lib', expected)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])
        patch = [shlex.split(call[0][0]) for call in commands.call_args_list if '--set-rpath' in call[0][0]]
        self.assertEqual(patch, [['patchelf', '--force-rpath', '--set-rpath', expected, str(path)]])

    def test_fixed_address_rewrite_rejects_a_page_zero_load(self):
        path = self.elf(elf_type=2)
        commands = self.patch_commands('$ORIGIN/lib', '$ORIGIN/lib:/site/OpenMPI/lib')
        load_info = [(2, [(0, 0x400000, 0x200000)]), (2, [(0, 0, 0x1000)])]
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands), mock.patch(
                'easybuild.easyblocks.o.orca.orca_elf_load_info', side_effect=load_info):
            with self.assertRaisesRegex(EasyBuildError, 'below 0x10000'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])

    def test_rewrite_rejects_misaligned_pie_loads(self):
        path = self.elf()
        commands = self.patch_commands('$ORIGIN/lib', '$ORIGIN/lib:/site/OpenMPI/lib')
        load_info = [(3, [(0, 0x400000, 0x200000)]), (3, [(1, 0x400000, 0x200000)])]
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands), mock.patch(
                'easybuild.easyblocks.o.orca.orca_elf_load_info', side_effect=load_info):
            with self.assertRaisesRegex(EasyBuildError, 'alignment was not preserved'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])

    def test_rewrite_rejects_lost_vendor_paths(self):
        path = self.elf()
        commands = self.patch_commands('$ORIGIN/lib:/vendor/lib', '$ORIGIN/lib:/vendor/lib:/site/OpenMPI/lib',
                                       recorded='/site/OpenMPI/lib')
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            with self.assertRaisesRegex(EasyBuildError, 'entries were not preserved'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])

    def test_existing_matching_dt_rpath_requires_no_rewrite(self):
        path = self.elf()
        commands = self.patch_commands('$ORIGIN/lib:/site/OpenMPI/lib', '$ORIGIN/lib:/site/OpenMPI/lib')
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])
        self.assertEqual(commands.call_count, 2)
        self.assertFalse(any('--set-rpath' in call[0][0] for call in commands.call_args_list))

    def test_existing_runpath_is_promoted_to_inherited_dt_rpath(self):
        path = self.elf()
        expected = '$ORIGIN/lib:/site/OpenMPI/lib'
        output = self.patch_commands(expected, expected)
        sections = iter(['(RUNPATH)', '(RPATH)'])

        def command_output(command, **kwargs):
            if shlex.split(command)[:2] == ['readelf', '-d']:
                return SimpleNamespace(output=next(sections))
            return output(command, **kwargs)

        commands = mock.Mock(side_effect=command_output)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])
        patch = [shlex.split(call[0][0]) for call in commands.call_args_list if '--set-rpath' in call[0][0]]
        self.assertEqual(len(patch), 1)
        self.assertIn('--force-rpath', patch[0])
        self.assertEqual(patch[0][patch[0].index('--set-rpath') + 1], expected)

    def test_rewrite_rejects_a_reduced_load_alignment(self):
        path = self.elf()
        commands = self.patch_commands('$ORIGIN/lib', '$ORIGIN/lib:/site/OpenMPI/lib')
        load_info = [(3, [(0, 0x400000, 0x200000)]), (3, [(0, 0x400000, 0x1000)])]
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands), mock.patch(
                'easybuild.easyblocks.o.orca.orca_elf_load_info', side_effect=load_info):
            with self.assertRaisesRegex(EasyBuildError, 'alignment was not preserved'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])

    def test_rewrite_rejects_lost_load_headers(self):
        path = self.elf()
        commands = self.patch_commands('$ORIGIN/lib', '$ORIGIN/lib:/site/OpenMPI/lib')
        load_info = [(3, [(0, 0x400000, 0x200000)]), (3, [])]
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands), mock.patch(
                'easybuild.easyblocks.o.orca.orca_elf_load_info', side_effect=load_info):
            with self.assertRaisesRegex(EasyBuildError, 'load structure was not preserved'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(), str(path), ['/site/OpenMPI/lib'])

    def test_consumer_selection_keeps_only_direct_mpi_elf_dependencies(self):
        """Text, symlink aliases, static files, and other dynamic consumers require no MPI rewrite."""
        mpi = self.elf('bin/orca_startup_mpi')
        shared = self.elf('lib/liborca_mpi.so.6')
        other = self.elf('bin/orca')
        static = self.elf('bin/orca_static')
        (self.root / 'bin' / 'orca_script').write_text('#!/bin/sh\n')
        (self.root / 'bin' / 'orca_alias').symlink_to(mpi.name)
        self.elf('datasets/not_a_runtime')

        def dynamic_section(command, **kwargs):
            path = shlex.split(command)[-1]
            if path in (str(mpi), str(shared)):
                output = '0x1 (NEEDED) Shared library: [libmpi.so.40]'
            elif path == str(other):
                output = '0x1 (NEEDED) Shared library: [liborca_tools.so.6]'
            elif path == str(static):
                output = 'There is no dynamic section in this file.'
            else:
                self.fail('Unexpected ELF inspection: %s' % path)
            return SimpleNamespace(output=output)

        commands = mock.Mock(side_effect=dynamic_section)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            self.assertEqual(EB_ORCA._mpi_consumers(SimpleNamespace(installdir=str(self.root))),
                             sorted([str(mpi), str(shared)]))
        self.assertEqual(commands.call_count, 4)

    def test_filtering_library_path_requires_the_selected_mpi_rpath(self):
        block = object.__new__(EB_ORCA)
        options = {'rpath': False, 'filter_env_vars': ['LD_LIBRARY_PATH']}
        with mock.patch.object(PackedBinary, 'post_processing_step') as parent, mock.patch(
                'easybuild.easyblocks.o.orca.build_option', side_effect=options.__getitem__), mock.patch(
                'easybuild.easyblocks.o.orca.which', return_value='/usr/bin/patchelf'), mock.patch.object(
                EB_ORCA, '_mpi_consumers', return_value=['/orca/bin/orca_mpi']), mock.patch.object(
                EB_ORCA, '_toolchain_mpi_rpath', return_value=['/site/OpenMPI/lib']) as probe, mock.patch.object(
                EB_ORCA, '_append_mpi_rpath') as patch:
            EB_ORCA.post_processing_step(block)
        parent.assert_called_once_with()
        probe.assert_called_once_with()
        patch.assert_called_once_with('/orca/bin/orca_mpi', ['/site/OpenMPI/lib'])

    def test_disabled_rpath_without_environment_filtering_does_not_link_a_probe(self):
        block = object.__new__(EB_ORCA)
        options = {'rpath': False, 'filter_env_vars': []}
        with mock.patch.object(PackedBinary, 'post_processing_step'), mock.patch(
                'easybuild.easyblocks.o.orca.build_option', side_effect=options.__getitem__), mock.patch.object(
                EB_ORCA, '_mpi_consumers') as consumers, mock.patch.object(EB_ORCA, '_toolchain_mpi_rpath') as probe:
            EB_ORCA.post_processing_step(block)
        consumers.assert_not_called()
        probe.assert_not_called()


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(OrcaMpiRpath)


if __name__ == '__main__':
    unittest.main()
