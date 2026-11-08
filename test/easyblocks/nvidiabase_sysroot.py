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
    rewrite_localrc_sysroot_settings,
    sysroot_dynamic_linker,
    sysroot_localrc_settings,
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

    def test_rewrite_replaces_first_assignment_in_place(self):
        text = (
            "set LFC=-lgfortran;\n"
            "set LDSO=/lib64/ld-linux-x86-64.so.2;\n"
            "set GCCDIR=/gcc/lib/gcc/x86_64-pc-linux-gnu/15.2.0;\n"
            "set DEFLIBDIR=/usr/lib64;\n"
            "set DEFSTDOBJDIR=/usr/lib64;\n"
            "export PGI=$COMPBASE;\n"
        )
        settings = sysroot_localrc_settings('/sysroot/usr/lib64', '/sysroot/usr/lib64/ld-linux-x86-64.so.2')
        out = rewrite_localrc_sysroot_settings(text, settings)
        lines = out.splitlines()
        # The host assignments are gone, each key is assigned once, order is kept.
        self.assertNotIn('set DEFLIBDIR=/usr/lib64;', lines)
        self.assertNotIn('set DEFSTDOBJDIR=/usr/lib64;', lines)
        self.assertNotIn('set LDSO=/lib64/ld-linux-x86-64.so.2;', lines)
        self.assertEqual(lines.index('set LDSO=/sysroot/usr/lib64/ld-linux-x86-64.so.2;'), 1)
        self.assertEqual(lines.index('set DEFLIBDIR=/sysroot/usr/lib64;'), 3)
        self.assertEqual(lines.index('set DEFSTDOBJDIR=/sysroot/usr/lib64;'), 4)
        self.assertEqual(sum(1 for line in lines if line.startswith('set DEFLIBDIR=')), 1)
        self.assertEqual(lines[0], 'set LFC=-lgfortran;')
        self.assertEqual(lines[-2], 'export PGI=$COMPBASE;')
        self.assertEqual(lines[-1], 'append LDLIBARGS=-L/sysroot/usr/lib64 -Wl,-rpath-link,/sysroot/usr/lib64;')
        # Applying the rewrite twice changes nothing more.
        self.assertEqual(rewrite_localrc_sysroot_settings(out, settings), out)

    def test_rewrite_without_sysroot_omits_ld_sysroot(self):
        """Compatibility: callers that omit sysroot must not emit ld --sysroot."""
        text = "set DEFLIBDIR=/usr/lib64;\nset DEFSTDOBJDIR=/usr/lib64;\nset LDSO=/lib64/ld-linux-x86-64.so.2;\n"
        settings = sysroot_localrc_settings('/sysroot/usr/lib64', '/sysroot/usr/lib64/ld-linux-x86-64.so.2')
        out = rewrite_localrc_sysroot_settings(text, settings)
        self.assertNotIn('--sysroot=', out)

    def test_rewrite_propagates_configured_sysroot_to_ld_sysroot(self):
        """
        EasyBuild --sysroot must become official ld --sysroot=PATH in LDLIBARGS.

        Use a multiarch ABI libdir so a dirname-strip of DEFLIBDIR would invent the
        wrong prefix; the configured sysroot argument is authoritative.
        """
        text = (
            "set DEFLIBDIR=/usr/lib/x86_64-linux-gnu;\n"
            "set DEFSTDOBJDIR=/usr/lib/x86_64-linux-gnu;\n"
            "set LDSO=/lib64/ld-linux-x86-64.so.2;\n"
        )
        abi_libdir = '/eessi/compat/usr/lib/x86_64-linux-gnu'
        settings = sysroot_localrc_settings(abi_libdir, abi_libdir + '/ld-linux-x86-64.so.2')
        configured = '/eessi/compat'
        out = rewrite_localrc_sysroot_settings(text, settings, sysroot=configured)
        lines = out.splitlines()
        self.assertIn('append LDLIBARGS=--sysroot=/eessi/compat;', lines)
        self.assertIn(
            'append LDLIBARGS=-L/eessi/compat/usr/lib/x86_64-linux-gnu '
            '-rpath-link=/eessi/compat/usr/lib/x86_64-linux-gnu;',
            lines,
        )
        # Stripping the multiarch ABI libdir must not invent alternate --sysroot values.
        self.assertNotIn('append LDLIBARGS=--sysroot=/eessi/compat/usr/lib;', out)
        self.assertNotIn('append LDLIBARGS=--sysroot=/eessi/compat/usr;', out)
        self.assertNotIn('append LDLIBARGS=--sysroot=/eessi;', out)
        # Idempotent with the same configured sysroot.
        self.assertEqual(
            rewrite_localrc_sysroot_settings(out, settings, sysroot=configured),
            out,
        )

    def test_rewrite_rejects_unsafe_sysroot_for_ld_sysroot(self):
        text = "set DEFLIBDIR=/usr/lib64;\nset DEFSTDOBJDIR=/usr/lib64;\nset LDSO=/lib64/ld-linux-x86-64.so.2;\n"
        settings = sysroot_localrc_settings('/sysroot/usr/lib64', '/sysroot/usr/lib64/ld-linux-x86-64.so.2')
        with self.assertRaises(EasyBuildError):
            rewrite_localrc_sysroot_settings(text, settings, sysroot='/bad path/with space')
        with self.assertRaises(EasyBuildError):
            rewrite_localrc_sysroot_settings(text, settings, sysroot='relative/sysroot')

    def test_rewrite_appends_keys_makelocalrc_did_not_emit(self):
        text = "set LFC=-lgfortran;\nset DEFLIBDIR=/usr/lib64;\nset DEFLIBDIR=/opt/lib64;\n"
        settings = sysroot_localrc_settings('/sysroot/usr/lib64', '/sysroot/usr/lib64/ld-linux-x86-64.so.2')
        out = rewrite_localrc_sysroot_settings(text, settings)
        lines = out.splitlines()
        # A duplicate host assignment collapses into the one rewritten line.
        self.assertEqual(sum(1 for line in lines if line.startswith('set DEFLIBDIR=')), 1)
        self.assertIn('set DEFSTDOBJDIR=/sysroot/usr/lib64;', lines)
        self.assertIn('set LDSO=/sysroot/usr/lib64/ld-linux-x86-64.so.2;', lines)
        self.assertTrue(out.endswith('\n'))

    def test_sysroot_dynamic_linker_prefers_the_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            libdir = os.path.join(tmp, 'usr', 'lib64')
            os.makedirs(libdir)
            real = os.path.join(libdir, 'ld-linux-x86-64.so.2')
            with open(real, 'w') as handle:
                handle.write('')
            os.symlink(real, os.path.join(libdir, 'ld-lsb-x86-64.so.3'))
            self.assertEqual(sysroot_dynamic_linker(tmp, libdir), os.path.realpath(real))

    def test_sysroot_dynamic_linker_missing_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'usr', 'lib64'))
            with self.assertRaises(EasyBuildError):
                sysroot_dynamic_linker(tmp, os.path.join(tmp, 'usr', 'lib64'))

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
