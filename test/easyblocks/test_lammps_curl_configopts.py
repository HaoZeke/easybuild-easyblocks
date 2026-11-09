"""
Tests for det_curl_configopts in the LAMMPS easyblock.
"""
import os
import tempfile
import unittest
from pathlib import Path

from easybuild.easyblocks.l.lammps import det_curl_configopts


class LammpsCurlConfigopts(unittest.TestCase):
    """Tests for the CMake options that bind libcurl to the cURL dependency."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'cURL'

    def install(self, *relative_paths):
        for relative in relative_paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')

    def opts(self, configopts=''):
        return det_curl_configopts(str(self.root), configopts, 'so')

    def test_lib_layout_binds_header_and_shared_library(self):
        """A lib/ installation gives the dependency's header directory and shared library."""
        self.install('include/curl/curl.h', 'lib/libcurl.so', 'lib/libcurl.a')
        self.assertEqual(self.opts(), ['-DCURL_NO_CURL_CMAKE=ON',
                                       '-DCURL_INCLUDE_DIR=%s' % (self.root / 'include'),
                                       '-DCURL_LIBRARY=%s' % (self.root / 'lib' / 'libcurl.so')])

    def test_lib64_layout_and_static_only_library(self):
        """The library is found in lib64/, and a static library is used when no shared one exists."""
        self.install('include/curl/curl.h', 'lib64/libcurl.so')
        self.assertIn('-DCURL_LIBRARY=%s' % (self.root / 'lib64' / 'libcurl.so'), self.opts())
        (self.root / 'lib64' / 'libcurl.so').unlink()
        self.install('lib/libcurl.a')
        self.assertIn('-DCURL_LIBRARY=%s' % (self.root / 'lib' / 'libcurl.a'), self.opts())

    def test_missing_files_are_not_passed(self):
        """Paths that do not exist in the dependency are left out."""
        self.install('lib/libcurl.so')
        self.assertEqual(self.opts(), ['-DCURL_NO_CURL_CMAKE=ON',
                                       '-DCURL_LIBRARY=%s' % (self.root / 'lib' / 'libcurl.so')])
        (self.root / 'lib' / 'libcurl.so').unlink()
        self.install('include/curl/curl.h')
        self.assertEqual(self.opts(), ['-DCURL_NO_CURL_CMAKE=ON', '-DCURL_INCLUDE_DIR=%s' % (self.root / 'include')])

    def test_explicit_settings_are_kept(self):
        """Each CURL_* option set by the caller is kept and not repeated."""
        self.install('include/curl/curl.h', 'lib/libcurl.so')
        opts = self.opts('-DCURL_LIBRARY=/other/libcurl.so')
        self.assertNotIn('-DCURL_LIBRARY=%s' % (self.root / 'lib' / 'libcurl.so'), opts)
        self.assertIn('-DCURL_INCLUDE_DIR=%s' % (self.root / 'include'), opts)
        self.assertNotIn('-DCURL_INCLUDE_DIR=%s' % (self.root / 'include'),
                         self.opts('-DCURL_INCLUDE_DIR:PATH=/other/include'))
        self.assertNotIn('-DCURL_NO_CURL_CMAKE=ON', self.opts('-DCURL_NO_CURL_CMAKE=OFF'))
        for configopts in ('-DCURL_ROOT=/other/cURL', '-DCURL_DIR=/other/cURL/lib/cmake/CURL'):
            with self.subTest(configopts=configopts):
                self.assertEqual(self.opts(configopts), [])

    def test_no_curl_dependency(self):
        """Without a cURL dependency no option is added."""
        for curl_root in (None, ''):
            with self.subTest(curl_root=curl_root):
                self.assertEqual(det_curl_configopts(curl_root, '', 'so'), [])
        self.assertFalse(os.path.exists(self.root))


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(LammpsCurlConfigopts)


if __name__ == '__main__':
    unittest.main()
