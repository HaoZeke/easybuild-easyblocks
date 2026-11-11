"""LAMMPS example merges preserve links without writing through destination links."""

import os
from pathlib import Path
import tempfile
import unittest

from easybuild.easyblocks.l.lammps import copy_lammps_examples
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.config import init_build_options


class LammpsExampleSymlinks(unittest.TestCase):
    def setUp(self):
        init_build_options()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source' / 'examples'
        self.destination = self.root / 'install' / 'examples'
        self.source.mkdir(parents=True)
        self.destination.mkdir(parents=True)

    def test_matching_kim_link_can_be_dangling_only_in_the_install(self):
        relative = Path('kim/plugin/LAMMPSInterfacePlugin.cmake')
        target = '../../../cmake/Modules/LAMMPSInterfacePlugin.cmake'
        source_link = self.source / relative
        destination_link = self.destination / relative
        source_link.parent.mkdir(parents=True)
        destination_link.parent.mkdir(parents=True)
        module = self.source.parent / 'cmake/Modules/LAMMPSInterfacePlugin.cmake'
        module.parent.mkdir(parents=True)
        module.write_text('plugin definition\n')
        source_link.symlink_to(target)
        destination_link.symlink_to(target)
        before = destination_link.lstat()
        self.assertTrue(source_link.exists())
        self.assertFalse(destination_link.exists())

        copy_lammps_examples(self.source, self.destination, dirs_exist_ok=True)

        self.assertTrue(destination_link.is_symlink())
        self.assertEqual(os.readlink(destination_link), target)
        self.assertFalse(destination_link.exists())
        self.assertEqual(destination_link.lstat().st_ino, before.st_ino)
        self.assertEqual(module.read_text(), 'plugin definition\n')

    def test_readable_source_link_is_copied_as_a_link_into_an_existing_tree(self):
        (self.source / 'input').write_text('run 100\n')
        (self.source / 'selected').symlink_to('input')

        copy_lammps_examples(self.source, self.destination, dirs_exist_ok=True)

        self.assertTrue((self.destination / 'selected').is_symlink())
        self.assertEqual(os.readlink(self.destination / 'selected'), 'input')
        self.assertEqual((self.destination / 'selected').read_text(), 'run 100\n')

    def test_changed_source_link_replaces_the_link_without_writing_its_target(self):
        outside = self.root / 'outside'
        outside.write_text('protected\n')
        (self.destination / 'selected').symlink_to(outside)
        (self.source / 'selected').symlink_to('missing-input')

        copy_lammps_examples(self.source, self.destination, dirs_exist_ok=True)

        self.assertEqual(outside.read_text(), 'protected\n')
        self.assertEqual(os.readlink(self.destination / 'selected'), 'missing-input')

    def test_regular_source_replaces_a_destination_link_without_following_it(self):
        outside = self.root / 'outside'
        outside.write_text('protected\n')
        (self.destination / 'input').symlink_to(outside)
        (self.source / 'input').write_text('run 100\n')

        copy_lammps_examples(self.source, self.destination, dirs_exist_ok=True)

        self.assertEqual(outside.read_text(), 'protected\n')
        self.assertFalse((self.destination / 'input').is_symlink())
        self.assertEqual((self.destination / 'input').read_text(), 'run 100\n')

    def test_destination_directory_link_is_refused_without_following_it(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'retained').write_text('protected\n')
        (self.destination / 'melt').symlink_to(outside, target_is_directory=True)
        (self.source / 'melt').mkdir()
        (self.source / 'melt/in.melt').write_text('run 100\n')

        with self.assertRaisesRegex(EasyBuildError, 'must be a real directory'):
            copy_lammps_examples(self.source, self.destination, dirs_exist_ok=True)

        self.assertTrue((self.destination / 'melt').is_symlink())
        self.assertEqual((outside / 'retained').read_text(), 'protected\n')
        self.assertFalse((outside / 'in.melt').exists())

    def test_linked_destination_root_is_refused_without_following_it(self):
        outside = self.root / 'outside'
        outside.mkdir()
        destination = self.root / 'linked-examples'
        destination.symlink_to(outside, target_is_directory=True)
        (self.source / 'input').write_text('run 100\n')

        with self.assertRaisesRegex(EasyBuildError, 'must be a real directory'):
            copy_lammps_examples(self.source, destination, dirs_exist_ok=True)

        self.assertTrue(destination.is_symlink())
        self.assertEqual(list(outside.iterdir()), [])


def suite(loader):
    return loader.loadTestsFromTestCase(LammpsExampleSymlinks)


if __name__ == '__main__':
    unittest.main()
