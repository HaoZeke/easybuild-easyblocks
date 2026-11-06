"""Check compiler arguments emitted by the AOCC easyblock."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from easybuild.easyblocks.a.aocc import EB_AOCC


class AoccSysrootConfiguration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'bin').mkdir()
        self.block = SimpleNamespace(
            installdir=str(self.root),
            gcc_prefix='/opt/gcc/lib/gcc/x86_64-linux-gnu/15.2.0',
            log=Mock(),
        )

    def test_clang_configuration_without_sysroot(self):
        with patch('easybuild.easyblocks.a.aocc.build_option', return_value=None):
            EB_AOCC._create_compiler_config_files(self.block, ['clang', 'clang++', 'clang-cpp'])
        for name in ['clang', 'clang++', 'clang-cpp']:
            self.assertEqual(
                (self.root / 'bin' / (name + '.cfg')).read_text(),
                '--gcc-install-dir=/opt/gcc/lib/gcc/x86_64-linux-gnu/15.2.0',
            )

    def test_clang_configuration_selects_sysroot_and_gcc(self):
        for sysroot in ['/opt/compat', '/opt/compat layer']:
            with self.subTest(sysroot=sysroot):
                with patch('easybuild.easyblocks.a.aocc.build_option', return_value=sysroot):
                    EB_AOCC._create_compiler_config_files(self.block, ['clang', 'clang++', 'clang-cpp'])
                for name in ['clang', 'clang++', 'clang-cpp']:
                    self.assertEqual(
                        shlex.split((self.root / 'bin' / (name + '.cfg')).read_text()),
                        ['--gcc-install-dir=/opt/gcc/lib/gcc/x86_64-linux-gnu/15.2.0',
                         '--sysroot=' + sysroot],
                    )

    def test_flang_wrapper_keeps_caller_arguments(self):
        for sysroot in [None, '/opt/compat', '/opt/compat layer']:
            with self.subTest(sysroot=sysroot):
                compiler = self.root / 'bin' / 'flang'
                if compiler.exists():
                    compiler.unlink()
                original = self.root / 'bin' / 'flang.orig'
                if original.exists():
                    original.unlink()
                compiler.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
                compiler.chmod(0o755)
                with patch('easybuild.easyblocks.a.aocc.build_option', return_value=sysroot):
                    EB_AOCC._create_compiler_wrappers(self.block, ['flang'])
                result = subprocess.run(
                    [str(compiler), '-c', 'input file.f90', '--sysroot=/caller'],
                    env=dict(os.environ, EBROOTGCCCORE='/opt/gcc'),
                    check=True, capture_output=True, text=True,
                )
                expected = ['--gcc-toolchain=/opt/gcc']
                if sysroot:
                    expected.append('--sysroot=' + sysroot)
                expected += ['-c', 'input file.f90', '--sysroot=/caller']
                self.assertEqual(result.stdout.splitlines(), expected)
                self.assertTrue(os.access(compiler, os.X_OK))


if __name__ == '__main__':
    unittest.main()
