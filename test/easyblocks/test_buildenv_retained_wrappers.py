"""BuildEnv wrapper preservation with complete content and metadata checks."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from easybuild.easyblocks.generic.buildenv import BuildEnv
from easybuild.easyblocks.generic.bundle import Bundle
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.config import init_build_options


class BuildEnvRetainedWrappers(unittest.TestCase):
    """Exercise fresh installation and strict reuse against real wrapper trees."""

    def setUp(self):
        init_build_options()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / 'build'
        self.install = self.root / 'install'
        self.source = self.build / 'rpath_wrappers'
        self.destination = self.install / 'bin/rpath_wrappers'
        (self.source / 'gcc_wrapper').mkdir(parents=True)
        (self.source / 'rpath_args.py').write_text('print("rpath")\n')
        (self.source / 'gcc_wrapper/gcc').write_text('PYTHON_EXE=/build/python\nexec "$PYTHON_EXE" "$@"\n')
        self.block = object.__new__(BuildEnv)
        self.block.log = mock.Mock()
        self.block.cfg = {'python_executable': 'python3', 'keeppreviousinstall': True, 'build_env_vars': True}
        self.block.installdir = str(self.install)
        self.block.rpath_wrappers_dir = str(self.build)

    def install_step(self):
        with mock.patch.object(Bundle, 'install_step') as base:
            self.block.install_step()
            base.assert_called_once_with()

    def populate(self):
        self.destination.mkdir(parents=True)
        (self.destination / 'gcc_wrapper').mkdir()
        (self.destination / 'rpath_args.py').write_bytes((self.source / 'rpath_args.py').read_bytes())
        (self.destination / 'gcc_wrapper/gcc').write_text('PYTHON_EXE=python3\nexec "$PYTHON_EXE" "$@"\n')
        for path in self.destination.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o555)

    def snapshot(self):
        result = {}
        for path in [self.destination] + sorted(self.destination.rglob('*')):
            value = path.lstat()
            result[str(path.relative_to(self.destination))] = (
                value.st_ino, value.st_mode, value.st_uid, value.st_gid, value.st_mtime_ns,
                path.read_bytes() if path.is_file() and not path.is_symlink() else None,
            )
        return result

    def test_fresh_install_is_unchanged_with_preservation_enabled_or_disabled(self):
        for keep in (False, True):
            with self.subTest(keep=keep):
                self.block.cfg['keeppreviousinstall'] = keep
                self.block.installdir = str(self.root / ('fresh-' + str(keep)))
                self.block.rpath_wrappers_dir = str(self.build)
                self.install_step()
                wrapper = Path(self.block.installdir) / 'bin/rpath_wrappers/gcc_wrapper/gcc'
                self.assertEqual(wrapper.read_text(), 'PYTHON_EXE=python3\nexec "$PYTHON_EXE" "$@"\n')
                self.assertEqual(self.block.rpath_wrappers_dir, str(Path(self.block.installdir) / 'bin'))

    def test_identical_literal_tree_is_preserved(self):
        self.populate()
        self.block.cfg['python_executable'] = None
        (self.source / 'gcc_wrapper/gcc').write_bytes((self.destination / 'gcc_wrapper/gcc').read_bytes())
        before = self.snapshot()
        self.install_step()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.block.rpath_wrappers_dir, str(self.install / 'bin'))

    def test_normalized_python_tree_is_preserved_and_module_uses_installed_wrappers(self):
        self.populate()
        before = self.snapshot()
        self.install_step()
        self.assertEqual(self.snapshot(), before)
        self.block.toolchain = SimpleNamespace(is_system_toolchain=lambda: False, vars={})
        self.block.module_load_environment = SimpleNamespace(PATH=[])
        with mock.patch.object(Bundle, 'make_module_step', return_value='module'):
            self.assertEqual(self.block.make_module_step(), 'module')
        self.assertEqual(self.block.module_load_environment.PATH, [str(self.destination / 'gcc_wrapper')])

    def test_content_difference_is_refused_without_writes(self):
        self.populate()
        (self.destination / 'gcc_wrapper/gcc').chmod(0o755)
        (self.destination / 'gcc_wrapper/gcc').write_text('different compiler\n')
        before = self.snapshot()
        with self.assertRaisesRegex(EasyBuildError, 'Retained RPATH wrappers differ'):
            self.install_step()
        self.assertEqual(self.snapshot(), before)

    def test_extra_and_missing_entries_are_refused(self):
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                self.install = Path(directory) / 'install'
                self.destination = self.install / 'bin/rpath_wrappers'
                self.block.installdir = str(self.install)
                self.block.rpath_wrappers_dir = str(self.build)
                self.populate()
                if extra:
                    path = self.destination / 'extra'
                    path.write_text('extra')
                    path.chmod(0o555)
                else:
                    (self.destination / 'rpath_args.py').unlink()
                before = self.snapshot()
                with self.assertRaisesRegex(EasyBuildError, 'Retained RPATH wrappers differ'):
                    self.install_step()
                self.assertEqual(self.snapshot(), before)

    def test_symlink_is_refused_without_following_it(self):
        self.populate()
        outside = self.root / 'outside'
        outside.write_text('outside contents')
        link = self.destination / 'gcc_wrapper/foreign'
        link.symlink_to(outside)
        with self.assertRaisesRegex(EasyBuildError, 'Unsafe retained wrapper metadata'):
            self.install_step()
        self.assertEqual(outside.read_text(), 'outside contents')
        self.assertTrue(link.is_symlink())

    def test_hard_link_is_refused(self):
        self.populate()
        original = self.destination / 'gcc_wrapper/gcc'
        os.link(original, self.root / 'alias')
        with self.assertRaisesRegex(EasyBuildError, 'multiple hard links'):
            self.install_step()

    def test_unsafe_permissions_and_missing_execution_are_refused(self):
        for mode in (0o777, 0o644):
            with self.subTest(mode=mode):
                self.block.rpath_wrappers_dir = str(self.build)
                wrapper = self.destination / 'gcc_wrapper/gcc'
                if not self.destination.exists():
                    self.populate()
                wrapper.chmod(mode)
                before = self.snapshot()
                with self.assertRaisesRegex(EasyBuildError, 'Unsafe retained wrapper metadata'):
                    self.install_step()
                self.assertEqual(self.snapshot(), before)

    def test_preservation_is_required_for_existing_destination(self):
        self.populate()
        self.block.cfg['keeppreviousinstall'] = False
        before = self.snapshot()
        with self.assertRaisesRegex(EasyBuildError, 'Target location .* already exists'):
            self.install_step()
        self.assertEqual(self.snapshot(), before)

    def test_linked_wrapper_root_is_refused(self):
        self.install.mkdir()
        (self.install / 'bin').symlink_to(self.root / 'external', target_is_directory=True)
        external = self.root / 'external/rpath_wrappers'
        external.mkdir(parents=True)
        (external / 'gcc_wrapper').mkdir()
        (external / 'rpath_args.py').write_text('print("rpath")\n')
        (external / 'gcc_wrapper/gcc').write_text('PYTHON_EXE=python3\nexec "$PYTHON_EXE" "$@"\n')
        with self.assertRaisesRegex(EasyBuildError, 'inside the installation directory'):
            self.install_step()


def suite(loader):
    """Return the retained-wrapper regression cases."""
    return loader.loadTestsFromTestCase(BuildEnvRetainedWrappers)


if __name__ == '__main__':
    unittest.main()
