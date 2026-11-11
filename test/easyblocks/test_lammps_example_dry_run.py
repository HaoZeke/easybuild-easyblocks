"""
Tests for the dry-run behaviour of lammps_example_input in the LAMMPS easyblock.
"""
import os
import tempfile
import unittest
from pathlib import Path

from easybuild.easyblocks.l.lammps import lammps_example_input
from easybuild.tools.build_log import EasyBuildError


class LammpsExampleDryRun(unittest.TestCase):
    """Tests for lammps_example_input with and without dry_run."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = str(Path(self.temporary.name) / 'not-installed')

    def test_dry_run_reports_the_conventional_path(self):
        """A dry run without installed inputs reports examples/<name>/in.<name>."""
        for name in ['atm', 'hugoniostat', 'msst', 'PACKAGES/shock/msst']:
            with self.subTest(name=name):
                expected = os.path.join(self.root, 'examples', name, 'in.' + os.path.basename(name))
                self.assertEqual(lammps_example_input(self.root, name, dry_run=True), expected)

    def test_dry_run_prefers_an_existing_nested_input(self):
        """A dry run reports an installed nested shock input when it exists."""
        nested = Path(self.temporary.name) / 'examples' / 'PACKAGES' / 'shock' / 'msst' / 'in.msst'
        nested.parent.mkdir(parents=True)
        nested.write_text('run 100\n')
        self.assertEqual(lammps_example_input(self.temporary.name, 'msst', dry_run=True), str(nested))

    def test_missing_input_fails_outside_a_dry_run(self):
        """Outside a dry run a missing input raises EasyBuildError."""
        for kwargs in ({}, {'dry_run': False}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(EasyBuildError):
                    lammps_example_input(self.root, 'msst', **kwargs)


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(LammpsExampleDryRun)


if __name__ == '__main__':
    unittest.main()
