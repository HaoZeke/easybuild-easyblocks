"""Replacement RPATH uses the ORCA load-layout and interpreter safeguards."""
import shlex
import unittest
from unittest import mock

from easybuild.easyblocks.o.orca import orca_elf_replace_rpath
from easybuild.tools.build_log import EasyBuildError
from test.easyblocks import test_orca_mpi_interpreter as interpreter_fixture
from test.easyblocks import test_orca_mpi_rpath as rpath_fixture


class OrcaElfReplaceRpath(unittest.TestCase):
    def setUp(self):
        self.fixture = rpath_fixture.OrcaMpiRpath('test_elf_load_information_reads_the_real_program_headers')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_replacement_preserves_order_origin_and_large_page_alignment(self):
        path = self.fixture.elf()
        current = '$ORIGIN/lib:/cvmfs/physical/mpi/lib:/vendor/lib'
        entries = ['$ORIGIN/lib', '/logical/mpi/lib', '/vendor/lib']
        commands = self.fixture.patch_commands(current, ':'.join(entries))
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            orca_elf_replace_rpath(str(path), entries)
        writes = [shlex.split(call.args[0]) for call in commands.call_args_list if '--set-rpath' in call.args[0]]
        self.assertEqual(writes, [['patchelf', '--page-size', '2097152', '--force-rpath',
                                  '--set-rpath', ':'.join(entries), str(path)]])

    def test_matching_replacement_does_not_write(self):
        path = self.fixture.elf()
        entries = ['$ORIGIN/lib', '/logical/mpi/lib']
        commands = self.fixture.patch_commands(':'.join(entries), ':'.join(entries))
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            orca_elf_replace_rpath(str(path), entries)
        self.assertEqual(commands.call_count, 2)
        self.assertFalse(any('--set-rpath' in call.args[0] for call in commands.call_args_list))

    def test_invalid_entries_refuse_before_writing(self):
        path = self.fixture.elf()
        for entries in ([], [''], ['/one:/two'], ['bad\0path']):
            with self.subTest(entries=entries):
                commands = self.fixture.patch_commands('$ORIGIN/lib', '$ORIGIN/lib')
                with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
                    with self.assertRaisesRegex(EasyBuildError, 'Invalid replacement'):
                        orca_elf_replace_rpath(str(path), entries)
                self.assertFalse(any('--set-rpath' in call.args[0] for call in commands.call_args_list))

    def test_replacement_rejects_lost_or_changed_entries(self):
        path = self.fixture.elf()
        commands = self.fixture.patch_commands('/physical/mpi/lib', '/logical/mpi/lib', recorded='/other/lib')
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            with self.assertRaisesRegex(EasyBuildError, 'entries were not preserved'):
                orca_elf_replace_rpath(str(path), ['/logical/mpi/lib'])

    def test_replacement_rejects_fixed_address_page_zero_and_misaligned_loads(self):
        for elf_type, patched, message in [(2, (0, 0, 0x1000), 'below 0x10000'),
                                         (3, (1, 0x400000, 0x200000), 'alignment was not preserved'),
                                         (3, (0, 0x400000, 0x1000), 'alignment was not preserved')]:
            with self.subTest(elf_type=elf_type, patched=patched):
                path = self.fixture.elf(elf_type=elf_type)
                commands = self.fixture.patch_commands('/physical/mpi/lib', '/logical/mpi/lib')
                layout = [(elf_type, [(0, 0x400000, 0x200000)]), (elf_type, [patched])]
                with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands), mock.patch(
                        'easybuild.easyblocks.o.orca.orca_elf_load_info', side_effect=layout):
                    with self.assertRaisesRegex(EasyBuildError, message):
                        orca_elf_replace_rpath(str(path), ['/logical/mpi/lib'])

    def test_replacement_keeps_the_selected_interpreter(self):
        fixture = interpreter_fixture.OrcaMpiInterpreter('test_program_headers_distinguish_an_executable_from_a_shared_library')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        path = fixture.elf(fixture.root / 'orca_mpi', fixture.loader)
        commands, writes = fixture.patch_commands(path, '/physical/mpi/lib', '/logical/mpi/lib')
        with mock.patch('easybuild.easyblocks.o.orca.run_shell_cmd', commands):
            orca_elf_replace_rpath(path, ['/logical/mpi/lib'], fixture.loader)
        self.assertEqual(len(writes), 1)
        self.assertNotIn('--set-interpreter', writes[0])


def suite(loader):
    return loader.loadTestsFromTestCase(OrcaElfReplaceRpath)


if __name__ == '__main__':
    unittest.main()
