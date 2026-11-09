import unittest

from easybuild.easyblocks.g.gromacs import det_hdf5_configopts, det_no_hdf5_configopts
from easybuild.tools import LooseVersion


class GromacsNoHdf5Configopts(unittest.TestCase):
    hdf5_root = '/software/HDF5/2.1.1-gompi-2026.1'

    def test_undeclared_hdf5_is_disabled_from_2026(self):
        for version in ('2026.0', '2026.3', '2026.4', '2027', LooseVersion('2026.4')):
            with self.subTest(version=version):
                self.assertEqual(det_no_hdf5_configopts(version, None, ''), ['-DGMX_USE_HDF5=OFF'])
                self.assertEqual(det_no_hdf5_configopts(version, '', '-DGMX_TEST_TIMEOUT_FACTOR=3'),
                                 ['-DGMX_USE_HDF5=OFF'])

    def test_declared_hdf5_is_left_to_det_hdf5_configopts(self):
        for version in ('2026.4', '2025.4'):
            with self.subTest(version=version):
                self.assertEqual(det_no_hdf5_configopts(version, self.hdf5_root, ''), [])

    def test_versions_before_2026_are_unchanged(self):
        for version in ('2024.5', '2025.0', '2025.4', LooseVersion('2025.4')):
            with self.subTest(version=version):
                self.assertEqual(det_no_hdf5_configopts(version, None, ''), [])

    def test_explicit_setting_is_kept(self):
        for configopts in ('-DGMX_USE_HDF5=ON', '-DGMX_USE_HDF5=OFF', '-DGMX_USE_HDF5:BOOL=ON',
                           '-DGMX_TEST_TIMEOUT_FACTOR=3 -DGMX_USE_HDF5=ON'):
            with self.subTest(configopts=configopts):
                self.assertEqual(det_no_hdf5_configopts('2026.4', None, configopts), [])

    def test_exactly_one_helper_sets_gmx_use_hdf5(self):
        for version in ('2025.4', '2026.4'):
            for hdf5_root in (None, self.hdf5_root):
                with self.subTest(version=version, hdf5_root=hdf5_root):
                    opts = det_hdf5_configopts(version, hdf5_root, '') + det_no_hdf5_configopts(version, hdf5_root, '')
                    self.assertLessEqual(len([o for o in opts if o.startswith('-DGMX_USE_HDF5')]), 1)


if __name__ == '__main__':
    unittest.main()
