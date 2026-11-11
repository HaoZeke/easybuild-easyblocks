"""Tests for the ORCA vendor library path below its installed executables."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import easybuild.tools.options as eboptions
from easybuild.easyblocks.generic.makecp import MakeCp
from easybuild.easyblocks.o.orca import EB_ORCA
from easybuild.tools import config


class OrcaLibraryLayout(unittest.TestCase):
    """Keep the vendor $ORIGIN/lib search path usable after installation."""

    @classmethod
    def setUpClass(cls):
        """Initialize the options used by the EasyBuild filesystem helpers."""
        super().setUpClass()
        options = eboptions.parse_options(args=[])
        config.init(options.options, options.get_options_by_section('config'))
        config.init_build_options(build_options={'extended_dry_run': False})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary_content = b'synthetic ORCA executable fixture\n'
        self.library_content = b'synthetic ORCA shared library fixture\n'

    def install_fixture(self, name='orca', libraries=('liborca_fixture.so.6',), existing_entry=None,
                        binary_subdir='bin'):
        """Materialize a synthetic vendor installation through the MakeCp boundary."""
        installdir = self.root / name
        easyblock = SimpleNamespace(
            cfg={'files_to_copy': [(['orca'], binary_subdir)]},
            installdir=str(installdir),
            version='6.1.1',
            log=mock.Mock(),
        )

        def install_vendor_files(block):
            bindir = Path(block.installdir) / binary_subdir
            bindir.mkdir(parents=True)
            (bindir / 'orca').write_bytes(self.binary_content)
            for library in libraries:
                libpath = Path(block.installdir) / 'lib' / library
                libpath.parent.mkdir(exist_ok=True)
                libpath.write_bytes(self.library_content)
            binlib = bindir / 'lib'
            if existing_entry == 'directory':
                binlib.mkdir()
                (binlib / 'vendor-marker').write_text('vendor directory\n')
            elif existing_entry == 'file':
                binlib.write_text('vendor file\n')
            elif existing_entry == 'symlink':
                vendor_lib = Path(block.installdir) / 'vendor-lib'
                vendor_lib.mkdir()
                (vendor_lib / 'vendor-marker').write_text('vendor target\n')
                binlib.symlink_to('../vendor-lib')
            elif existing_entry == 'dangling-symlink':
                binlib.symlink_to('../missing-vendor-lib')

        with mock.patch.object(MakeCp, 'install_step', autospec=True, side_effect=install_vendor_files) as install:
            EB_ORCA.install_step(easyblock)
        install.assert_called_once_with(easyblock)
        self.assertEqual((installdir / binary_subdir / 'orca').read_bytes(), self.binary_content)
        return installdir

    def test_vendor_origin_library_path_resolves_to_the_installed_library(self):
        """Executables find the installed library through a relative directory link."""
        installdir = self.install_fixture()
        binlib = installdir / 'bin' / 'lib'
        library = installdir / 'lib' / 'liborca_fixture.so.6'
        resolved_library = binlib / library.name
        self.assertTrue(binlib.is_symlink())
        self.assertEqual(os.readlink(str(binlib)), '../lib')
        self.assertEqual(resolved_library.resolve(), library.resolve())
        self.assertTrue(os.path.samefile(str(resolved_library), str(library)))
        self.assertEqual(library.read_bytes(), self.library_content)

    def test_relative_library_path_survives_installation_relocation(self):
        """Moving the installation preserves access through the vendor search path."""
        installdir = self.install_fixture()
        relocated = self.root / 'relocated'
        installdir.rename(relocated)
        self.assertTrue(os.path.samefile(
            str(relocated / 'bin' / 'lib' / 'liborca_fixture.so.6'),
            str(relocated / 'lib' / 'liborca_fixture.so.6'),
        ))
        self.assertEqual((relocated / 'bin' / 'orca').read_bytes(), self.binary_content)

    def test_existing_bin_library_entries_are_preserved(self):
        """A vendor directory, file, or symlink retains its original contents or target."""
        for kind in ('directory', 'file', 'symlink', 'dangling-symlink'):
            with self.subTest(kind=kind):
                installdir = self.install_fixture(name=kind, existing_entry=kind)
                binlib = installdir / 'bin' / 'lib'
                if kind == 'directory':
                    self.assertFalse(binlib.is_symlink())
                    self.assertEqual((binlib / 'vendor-marker').read_text(), 'vendor directory\n')
                elif kind == 'file':
                    self.assertFalse(binlib.is_symlink())
                    self.assertEqual(binlib.read_text(), 'vendor file\n')
                elif kind == 'symlink':
                    self.assertTrue(binlib.is_symlink())
                    self.assertEqual(os.readlink(str(binlib)), '../vendor-lib')
                    self.assertEqual((binlib / 'vendor-marker').read_text(), 'vendor target\n')
                else:
                    self.assertTrue(binlib.is_symlink())
                    self.assertEqual(os.readlink(str(binlib)), '../missing-vendor-lib')
                    self.assertFalse(binlib.exists())

    def test_custom_root_executable_layout_has_no_bin_library_link(self):
        """A custom executable location retains its sibling library directory."""
        installdir = self.install_fixture(binary_subdir='')
        self.assertFalse((installdir / 'bin').exists())
        self.assertEqual((installdir / 'orca').read_bytes(), self.binary_content)
        self.assertEqual((installdir / 'lib' / 'liborca_fixture.so.6').read_bytes(), self.library_content)

    def test_static_layout_without_orca_libraries_has_no_bin_library_link(self):
        """A static executable requires no compatibility link or invented library directory."""
        installdir = self.install_fixture(libraries=())
        self.assertFalse(os.path.lexists(str(installdir / 'bin' / 'lib')))
        self.assertFalse((installdir / 'lib').exists())

    def test_other_libraries_do_not_trigger_an_orca_library_link(self):
        """A library directory containing other vendor files does not require the ORCA link."""
        installdir = self.install_fixture(libraries=('libother_fixture.so.6',))
        self.assertFalse(os.path.lexists(str(installdir / 'bin' / 'lib')))
        self.assertEqual((installdir / 'lib' / 'libother_fixture.so.6').read_bytes(), self.library_content)


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(OrcaLibraryLayout)


if __name__ == '__main__':
    unittest.main()
