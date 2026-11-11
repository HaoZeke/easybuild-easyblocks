"""Synthetic ELF regressions for pairing the ORCA MPI libc and interpreter."""
import os
import shlex
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import easybuild.tools.options as eboptions
from easybuild.easyblocks.o.orca import EB_ORCA, orca_elf_interpreter, orca_elf_load_info
from easybuild.tools import config
from easybuild.tools.build_log import EasyBuildError


class OrcaMpiInterpreter(unittest.TestCase):
    """Inspect synthetic program headers without executing a compiler or vendor binary."""

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
        self.loader = '/site/compat/lib64/ld-linux-x86-64.so.2'
        self.host_loader = '/lib64/ld-linux-x86-64.so.2'
        self.runtime = ['/site/OpenMPI/lib', '/site/compat/lib64']

    def elf(self, path, interpreter=None, elf_type=3):
        """Write load and interpreter headers with no executable machine code."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = os.fsencode(interpreter) + b'\0' if interpreter is not None else b''
        count = 2 if interpreter is not None else 1
        offset = 64 + 56 * count
        ident = b'\x7fELF' + bytes([2, 1, 1]) + b'\0' * 9
        header = struct.pack('<HHIQQQIHHHHHH', elf_type, 62, 1, 0x400000, 64, 0, 0,
                             64, 56, count, 0, 0, 0)
        load = struct.pack('<IIQQQQQQ', 1, 5, 0, 0x400000, 0x400000,
                           offset + len(payload), offset + len(payload), 0x200000)
        interp = b''
        if interpreter is not None:
            interp = struct.pack('<IIQQQQQQ', 3, 4, offset, 0x400000 + offset,
                                 0x400000 + offset, len(payload), len(payload), 1)
        path.write_bytes(ident + header + load + interp + payload)
        return str(path)

    def probe(self, interpreter):
        """The mocked MPI compiler emits a real synthetic ELF program-header table."""
        build = self.root / 'build'
        build.mkdir()
        block = SimpleNamespace(builddir=str(build), cfg={'start_dir': str(self.root / 'source')})

        def output(command, **kwargs):
            arguments = shlex.split(command)
            if arguments[0] == 'mpicc':
                self.elf(arguments[arguments.index('-o') + 1], interpreter)
                return SimpleNamespace(output='')
            self.assertEqual(arguments[:2], ['patchelf', '--print-rpath'])
            return SimpleNamespace(output=':'.join(self.runtime))

        commands = mock.Mock(side_effect=output)
        is_file = os.path.isfile

        def runtime_file(path):
            return path in ('/site/compat/lib64/libc.so.6', self.loader) or is_file(path)

        with mock.patch.dict(os.environ, {'MPICC': 'mpicc'}), mock.patch(
                'easybuild.easyblocks.o.orca.os.path.isfile', side_effect=runtime_file), mock.patch(
                'easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            result = EB_ORCA._toolchain_mpi_rpath(block)
        return block, result, commands

    def patch_commands(self, path, current, expected, apply_interpreter=True):
        """Keep RPATH output and the synthetic PT_INTERP consistent with the patch command."""
        patched = []

        def output(command, **kwargs):
            arguments = shlex.split(command)
            if arguments[:2] == ['patchelf', '--print-rpath']:
                return SimpleNamespace(output=expected if patched else current)
            if arguments[:2] == ['readelf', '-d']:
                return SimpleNamespace(output='(RPATH)')
            self.assertEqual(arguments[0], 'patchelf')
            self.assertIn('--set-rpath', arguments)
            patched.append(arguments)
            if '--set-interpreter' in arguments and apply_interpreter:
                elf_type = struct.unpack_from('<H', Path(path).read_bytes(), 16)[0]
                self.elf(path, arguments[arguments.index('--set-interpreter') + 1], elf_type)
            return SimpleNamespace(output='')

        return mock.Mock(side_effect=output), patched

    def test_program_headers_distinguish_an_executable_from_a_shared_library(self):
        executable = self.elf(self.root / 'orca_mpi', self.host_loader)
        shared = self.elf(self.root / 'liborca_mpi.so')
        self.assertEqual(orca_elf_interpreter(executable), self.host_loader)
        self.assertIsNone(orca_elf_interpreter(shared))
        self.assertEqual(orca_elf_load_info(executable), (3, [(0, 0x400000, 0x200000)]))

    def test_probe_derives_interpreter_from_the_active_mpi_link(self):
        block, entries, commands = self.probe(self.loader)
        self.assertEqual(entries, self.runtime)
        self.assertEqual(block._mpi_interpreter, self.loader)
        self.assertEqual(commands.call_count, 2)

    def test_probe_requires_an_interpreter_when_runtime_paths_select_libc(self):
        with self.assertRaisesRegex(EasyBuildError, 'permanent ELF interpreter'):
            self.probe(None)

    def test_probe_rejects_a_temporary_interpreter(self):
        with self.assertRaisesRegex(EasyBuildError, 'permanent ELF interpreter'):
            self.probe(str(self.root / 'build' / 'ld-linux.so'))

    def test_pie_pairs_libc_rpath_with_the_mpi_probe_interpreter(self):
        path = self.elf(self.root / 'orca mpi', self.host_loader)
        current = '$ORIGIN/lib:/vendor/lib'
        expected = current + ':' + ':'.join(self.runtime)
        commands, patched = self.patch_commands(path, current, expected)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(_mpi_interpreter=self.loader), path, self.runtime)
        self.assertEqual(patched, [['patchelf', '--page-size', '2097152', '--force-rpath',
                                    '--set-rpath', expected, '--set-interpreter', self.loader, path]])
        self.assertEqual(orca_elf_interpreter(path), self.loader)

    def test_matching_rpath_still_repairs_a_different_interpreter(self):
        path = self.elf(self.root / 'orca_mpi', self.host_loader)
        expected = '$ORIGIN/lib:' + ':'.join(self.runtime)
        commands, patched = self.patch_commands(path, expected, expected)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(_mpi_interpreter=self.loader), path, self.runtime)
        self.assertEqual(len(patched), 1)
        self.assertIn('--set-interpreter', patched[0])
        self.assertEqual(orca_elf_interpreter(path), self.loader)

    def test_fixed_address_pairing_does_not_request_a_large_page_size(self):
        path = self.elf(self.root / 'orca_mpi', self.host_loader, elf_type=2)
        expected = '$ORIGIN/lib:' + ':'.join(self.runtime)
        commands, patched = self.patch_commands(path, '$ORIGIN/lib', expected)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(_mpi_interpreter=self.loader), path, self.runtime)
        self.assertEqual(len(patched), 1)
        self.assertNotIn('--page-size', patched[0])
        self.assertIn('--set-interpreter', patched[0])
        self.assertEqual(orca_elf_interpreter(path), self.loader)

    def test_shared_library_does_not_acquire_an_interpreter(self):
        path = self.elf(self.root / 'liborca_mpi.so')
        expected = '$ORIGIN/lib:' + ':'.join(self.runtime)
        commands, patched = self.patch_commands(path, '$ORIGIN/lib', expected)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            EB_ORCA._append_mpi_rpath(SimpleNamespace(_mpi_interpreter=self.loader), path, self.runtime)
        self.assertEqual(len(patched), 1)
        self.assertNotIn('--set-interpreter', patched[0])
        self.assertIsNone(orca_elf_interpreter(path))

    def test_rewrite_refuses_an_unmodified_interpreter(self):
        path = self.elf(self.root / 'orca_mpi', self.host_loader)
        expected = '$ORIGIN/lib:' + ':'.join(self.runtime)
        commands, _ = self.patch_commands(path, '$ORIGIN/lib', expected, apply_interpreter=False)
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            with self.assertRaisesRegex(EasyBuildError, 'ELF interpreter was not preserved'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(_mpi_interpreter=self.loader), path, self.runtime)

    def test_interpreter_rewrite_retains_the_fixed_address_load_check(self):
        path = self.elf(self.root / 'orca_mpi', self.host_loader, elf_type=2)
        expected = '$ORIGIN/lib:' + ':'.join(self.runtime)
        commands, _ = self.patch_commands(path, expected, expected)
        load_info = [(2, [(0, 0x400000, 0x200000)]), (2, [(0, 0, 0x1000)])]
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands), mock.patch(
                'easybuild.easyblocks.o.orca.orca_elf_load_info', side_effect=load_info):
            with self.assertRaisesRegex(EasyBuildError, 'below 0x10000'):
                EB_ORCA._append_mpi_rpath(SimpleNamespace(_mpi_interpreter=self.loader), path, self.runtime)


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(OrcaMpiInterpreter)


if __name__ == '__main__':
    unittest.main()
