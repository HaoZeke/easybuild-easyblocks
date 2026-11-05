from pathlib import Path
import tempfile
import unittest

from easybuild.easyblocks.l.lammps import lammps_example_input
from easybuild.tools.build_log import EasyBuildError


class LammpsExamplePaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def create_input(self, relative, name):
        path = self.root / 'examples' / relative / ('in.' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('run 100\n')
        return str(path)

    def test_flat_layout_keeps_its_installed_input(self):
        expected = self.create_input('melt', 'melt')
        self.assertEqual(lammps_example_input(str(self.root), 'melt'), expected)

    def test_shock_inputs_resolve_in_the_package_layout(self):
        for name in ['hugoniostat', 'msst']:
            with self.subTest(name=name):
                expected = self.create_input('PACKAGES/shock/' + name, name)
                self.assertEqual(lammps_example_input(str(self.root), name), expected)

    def test_explicit_nested_input_keeps_its_basename(self):
        expected = self.create_input('PACKAGES/shock/msst', 'msst')
        self.assertEqual(lammps_example_input(str(self.root), 'PACKAGES/shock/msst'), expected)

    def test_missing_input_is_refused_before_an_mpi_command_is_launched(self):
        with self.assertRaises(EasyBuildError):
            lammps_example_input(str(self.root), 'msst')


if __name__ == '__main__':
    unittest.main()
