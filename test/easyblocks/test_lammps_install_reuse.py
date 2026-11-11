"""LAMMPS example and potential installation with preserved destinations."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from easybuild.easyblocks.generic.cmakemake import CMakeMake
from easybuild.easyblocks.l.lammps import EB_LAMMPS, translate_lammps_version
from easybuild.tools.build_log import EasyBuildError


class LammpsInstallReuse(unittest.TestCase):
    """Exercise installation copies against real fresh and populated directories."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.install = self.root / 'install'
        for name in ('src', 'examples/melt', 'potentials', 'python'):
            (self.source / name).mkdir(parents=True, exist_ok=True)
        (self.source / 'src/version.h').write_text('#define LAMMPS_VERSION "30 Sep 2026"\n')
        (self.source / 'LICENSE').write_text('license fixture\n')
        (self.source / 'examples/melt/in.melt').write_text('run 100\n')
        (self.source / 'potentials/pair.table').write_text('potential fixture\n')
        (self.source / 'examples/missing.link').symlink_to('missing-input')
        self.block = object.__new__(EB_LAMMPS)
        self.block.log = mock.Mock()
        self.block.cfg = {'start_dir': str(self.source), 'keeppreviousinstall': False}
        self.block.installdir = str(self.install)
        self.block.builddir = str(self.root / 'build')
        self.block.cur_version = translate_lammps_version('30Sep2026')

    def install_step(self, keep):
        self.block.cfg['keeppreviousinstall'] = keep
        with mock.patch.object(CMakeMake, 'install_step') as base, \
                mock.patch('easybuild.easyblocks.l.lammps.get_software_version', return_value='3.14.2'), \
                mock.patch('easybuild.easyblocks.l.lammps.set_py_env_vars'), \
                mock.patch('easybuild.easyblocks.l.lammps.run_shell_cmd') as package:
            self.block.install_step()
            base.assert_called_once_with()
            package.assert_called_once()

    def assert_installed(self):
        self.assertEqual((self.install / 'examples/melt/in.melt').read_text(), 'run 100\n')
        self.assertEqual((self.install / 'potentials/pair.table').read_text(), 'potential fixture\n')
        self.assertTrue((self.install / 'examples/missing.link').is_symlink())
        self.assertEqual(os.readlink(self.install / 'examples/missing.link'), 'missing-input')
        self.assertEqual((self.install / 'src/version.h').read_text(),
                         (self.source / 'src/version.h').read_text())
        self.assertEqual((self.install / 'LICENSE').read_text(), 'license fixture\n')

    def test_fresh_install_without_preservation_copies_both_trees(self):
        self.install_step(False)
        self.assert_installed()

    def test_preservation_also_allows_fresh_destination_trees(self):
        self.install_step(True)
        self.assert_installed()

    def test_preserved_destination_trees_merge_and_refresh_selected_files(self):
        for name in ('examples/melt', 'potentials'):
            (self.install / name).mkdir(parents=True, exist_ok=True)
        (self.install / 'examples/melt/in.melt').write_text('old input\n')
        (self.install / 'potentials/pair.table').write_text('old potential\n')
        (self.install / 'examples/retained.txt').write_text('retained example\n')
        (self.install / 'potentials/retained.txt').write_text('retained potential\n')
        self.install_step(True)
        self.assert_installed()
        self.assertEqual((self.install / 'examples/retained.txt').read_text(), 'retained example\n')
        self.assertEqual((self.install / 'potentials/retained.txt').read_text(), 'retained potential\n')

    def test_existing_destination_without_preservation_is_refused(self):
        for name in ('examples', 'potentials'):
            with self.subTest(destination=name), tempfile.TemporaryDirectory() as directory:
                self.install = Path(directory) / 'install'
                self.block.installdir = str(self.install)
                (self.install / name).mkdir(parents=True)
                marker = self.install / name / 'retained.txt'
                marker.write_text('retained content\n')
                with self.assertRaises(EasyBuildError):
                    self.install_step(False)
                self.assertEqual(marker.read_text(), 'retained content\n')


def suite(loader):
    """Return the installation-copy regression cases."""
    return loader.loadTestsFromTestCase(LammpsInstallReuse)


if __name__ == '__main__':
    unittest.main()
