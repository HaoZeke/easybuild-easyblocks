##
# Copyright 2026 Ghent University
#
# This file is part of EasyBuild,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# Flemish Research Foundation (FWO) (http://www.fwo.be/en)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# https://github.com/easybuilders/easybuild
#
# EasyBuild is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# EasyBuild is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EasyBuild.  If not, see <http://www.gnu.org/licenses/>.
##
"""
Unit tests for NvidiaBase EasyBuild --sysroot LIBRARY_PATH / DEFSTDOBJDIR helpers.

These tests cover public helper behaviour only. They are not native NVHPC proof:
compile/link success with a real nvc and runtime PT_INTERP/NEEDED checks belong
to the separate native qualification probe.
"""

import os
import tempfile
from unittest import TestCase, TestLoader, TextTestRunner

from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.systemtools import AARCH64, X86_64
from easybuild.easyblocks.generic.nvidiabase import (
    build_sysroot_defstdobjdir_localrc_snippet,
    find_nvhpc_localrc_path,
    prepend_sysroot_to_library_path_env,
    resolve_sysroot_abi_libdir,
    sysroot_link_control_enabled,
    validate_unquoted_localrc_path,
)


def _populate_abi_libdir(root, rel='usr/lib64'):
    """Create a minimal sysroot ABI libdir with crt + libc.so.6."""
    libdir = os.path.join(root, rel)
    os.makedirs(libdir)
    for name in ('crt1.o', 'crti.o', 'libc.so.6'):
        with open(os.path.join(libdir, name), 'w') as handle:
            handle.write('x')
    return os.path.realpath(libdir)


class NvidiaBaseSysrootHelpersTest(TestCase):
    """Tests for sysroot-gated NVHPC link/startup helpers."""

    def test_sysroot_link_control_disabled_when_unset(self):
        """No --sysroot means helpers must stay inactive."""
        self.assertFalse(sysroot_link_control_enabled(None))
        self.assertFalse(sysroot_link_control_enabled(''))

    def test_sysroot_link_control_enabled_when_set(self):
        self.assertTrue(sysroot_link_control_enabled('/cvmfs/example/compat'))

    def test_validate_rejects_relative_and_unsafe_paths(self):
        with self.assertRaises(EasyBuildError):
            validate_unquoted_localrc_path('rel/path', 'label')
        with self.assertRaises(EasyBuildError):
            validate_unquoted_localrc_path('/tmp/has space', 'label')
        with self.assertRaises(EasyBuildError):
            validate_unquoted_localrc_path('/tmp/has;semi', 'label')

    def test_resolve_requires_complete_abi_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            libdir = os.path.join(tmp, 'usr/lib64')
            os.makedirs(libdir)
            open(os.path.join(libdir, 'crt1.o'), 'w').close()
            with self.assertRaises(EasyBuildError):
                resolve_sysroot_abi_libdir(tmp, X86_64)

    def test_resolve_ok_x86_64(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _populate_abi_libdir(tmp, 'usr/lib64')
            self.assertEqual(resolve_sysroot_abi_libdir(tmp, X86_64), expected)

    def test_resolve_ok_aarch64_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _populate_abi_libdir(tmp, 'usr/lib/aarch64-linux-gnu')
            self.assertEqual(resolve_sysroot_abi_libdir(tmp, AARCH64), expected)

    def test_library_path_prepend_does_not_mutate_input(self):
        original = {'LIBRARY_PATH': '/host/lib', 'PATH': '/bin'}
        snapshot = dict(original)
        out = prepend_sysroot_to_library_path_env(original, '/sysroot/lib64')
        self.assertEqual(original, snapshot)
        self.assertEqual(out['LIBRARY_PATH'], '/sysroot/lib64:/host/lib')
        self.assertEqual(out['PATH'], '/bin')

    def test_library_path_prepend_when_absent(self):
        out = prepend_sysroot_to_library_path_env({}, '/sysroot/lib64')
        self.assertEqual(out['LIBRARY_PATH'], '/sysroot/lib64')

    def test_defstdobjdir_snippet_only_sets_vendor_variable(self):
        text = build_sysroot_defstdobjdir_localrc_snippet('/sysroot/usr/lib64')
        self.assertIn('set DEFSTDOBJDIR=/sysroot/usr/lib64;', text)
        # Only DEFSTDOBJDIR may be assigned; do not invent other localrc controls.
        for banned in (
            'set LDLIBARGS',
            'append LDLIBARGS',
            'set DEFLIBDIR',
            'set LDSO',
            'set SITELIB',
            'append SITELIB',
        ):
            self.assertNotIn(banned, text)

    def test_find_localrc_prefers_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, 'bin')
            os.makedirs(bin_dir)
            bin_localrc = os.path.join(bin_dir, 'localrc')
            top_localrc = os.path.join(tmp, 'localrc')
            open(bin_localrc, 'w').close()
            open(top_localrc, 'w').close()
            self.assertEqual(find_nvhpc_localrc_path(tmp), bin_localrc)

    def test_find_localrc_missing_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EasyBuildError):
                find_nvhpc_localrc_path(tmp)


def suite(loader=None):
    """Return all nvidiabase sysroot helper tests."""
    if loader is None:
        loader = TestLoader()
    return loader.loadTestsFromTestCase(NvidiaBaseSysrootHelpersTest)


if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite())
