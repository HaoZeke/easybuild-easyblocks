"""Exercise the Gurobi loader-path helper with original ELF fixtures."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from easybuild.easyblocks.g.gurobi import EB_Gurobi
from easybuild.easyblocks.generic.tarball import Tarball


class GurobiRpathContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ['bin/vendor', 'lib']:
            (self.root / directory).mkdir(parents=True)
        (self.root / 'helper.c').write_text('int helper(void) { return 42; }\n')
        (self.root / 'answer.c').write_text(
            'extern int helper(void); int answer(void) { return helper(); }\n')
        (self.root / 'main.c').write_text(
            '#include <stdio.h>\nextern int answer(void);\n'
            'int main(void) { printf("%d\\n", answer()); return answer() != 42; }\n')
        self.run_command('gcc', '-shared', '-fPIC', str(self.root / 'helper.c'),
                         '-Wl,-soname,libhelper.so', '-o', str(self.root / 'bin/vendor/libhelper.so'))
        self.run_command('gcc', '-shared', '-fPIC', str(self.root / 'answer.c'),
                         '-L' + str(self.root / 'bin/vendor'), '-lhelper',
                         '-Wl,-soname,libanswer.so', '-o', str(self.root / 'lib/libanswer.so'))
        self.run_command('gcc', str(self.root / 'main.c'), '-L' + str(self.root / 'lib'), '-lanswer',
                         '-Wl,-rpath-link,' + str(self.root / 'bin/vendor'),
                         '-Wl,--disable-new-dtags,-rpath,$ORIGIN/vendor:$ORIGIN/vendor-keep',
                         '-o', str(self.root / 'bin/check'))
        self.binary = self.root / 'bin/check'
        self.environment = dict(os.environ)
        self.environment.pop('LD_LIBRARY_PATH', None)
        self.additions = ['$ORIGIN/../lib', str(self.root / 'lib')]

    @staticmethod
    def run_command(*args):
        return subprocess.run(args, check=True, capture_output=True, text=True)

    def repair(self):
        EB_Gurobi._append_permanent_lib_rpath(
            SimpleNamespace(), str(self.binary), self.additions)

    def test_missing_library_is_resolved_and_vendor_paths_survive(self):
        control = subprocess.run([str(self.binary)], env=self.environment,
                                 capture_output=True, text=True)
        self.assertNotEqual(control.returncode, 0)
        self.assertIn('libanswer.so', control.stderr)
        self.repair()
        result = subprocess.run([str(self.binary)], env=self.environment,
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, '42\n')
        paths = self.run_command('patchelf', '--print-rpath', str(self.binary)).stdout.strip().split(':')
        self.assertEqual(paths, ['$ORIGIN/vendor', '$ORIGIN/vendor-keep'] + self.additions)
        self.assertIn('(RPATH)', self.run_command('readelf', '-d', str(self.binary)).stdout)
        digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.repair()
        self.assertEqual(hashlib.sha256(self.binary.read_bytes()).hexdigest(), digest)

    def test_runpath_is_converted_even_when_search_directories_are_present(self):
        self.run_command('patchelf', '--set-rpath',
                         ':'.join(['$ORIGIN/vendor', '$ORIGIN/vendor-keep'] + self.additions),
                         str(self.binary))
        self.assertIn('(RUNPATH)', self.run_command('readelf', '-d', str(self.binary)).stdout)
        self.repair()
        self.assertIn('(RPATH)', self.run_command('readelf', '-d', str(self.binary)).stdout)
        result = subprocess.run([str(self.binary)], env=self.environment,
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, '42\n')

    def build_library(self, name, *flags):
        library = self.root / 'lib' / name
        self.run_command('gcc', '-shared', '-fPIC', str(self.root / 'answer.c'), *flags,
                         '-L' + str(self.root / 'bin/vendor'), '-lhelper',
                         '-Wl,--disable-new-dtags,-rpath,$ORIGIN/../bin/vendor',
                         '-Wl,-soname,' + name, '-o', str(library))
        return library

    def load_segments(self, path):
        segments = []
        for line in self.run_command('readelf', '-lW', str(path)).stdout.splitlines():
            fields = line.split()
            if fields and fields[0] == 'LOAD':
                segments.append((int(fields[1], 16), int(fields[2], 16), int(fields[-1], 16)))
        return segments

    def test_large_load_alignment_survives_rpath_growth(self):
        library = self.build_library('libwide.so', '-Wl,-z,max-page-size=0x200000')
        self.assertEqual({align for _, _, align in self.load_segments(library)}, {0x200000})
        additions = ['$ORIGIN/../lib', os.path.join(str(self.root), *['install'] * 400, 'lib')]
        EB_Gurobi._append_permanent_lib_rpath(SimpleNamespace(), str(library), additions)
        paths = self.run_command('patchelf', '--print-rpath', str(library)).stdout.strip().split(':')
        self.assertEqual(paths, ['$ORIGIN/../bin/vendor'] + additions)
        segments = self.load_segments(library)
        self.assertGreater(len(segments), 4)
        for offset, vaddr, align in segments:
            self.assertEqual(align, 0x200000)
            self.assertEqual((vaddr - offset) % align, 0)
        result = subprocess.run([sys.executable, '-c', 'import ctypes, sys; '
                                 'print(ctypes.CDLL(sys.argv[1]).answer())', str(library)],
                                env=self.environment, check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, '42\n')

    def test_four_kib_library_matches_plain_patchelf(self):
        library = self.build_library('libnarrow.so')
        self.assertEqual({align for _, _, align in self.load_segments(library)}, {0x1000})
        reference = self.root / 'reference.so'
        reference.write_bytes(library.read_bytes())
        additions = ['$ORIGIN/../lib', os.path.join(str(self.root), *['install'] * 400, 'lib')]
        self.run_command('patchelf', '--force-rpath', '--set-rpath',
                         ':'.join(['$ORIGIN/../bin/vendor'] + additions), str(reference))
        EB_Gurobi._append_permanent_lib_rpath(SimpleNamespace(), str(library), additions)
        self.assertEqual(library.read_bytes(), reference.read_bytes())

    def test_non_elf_file_is_left_untouched(self):
        script = self.root / 'bin/launcher.sh'
        script.write_text('#!/bin/sh\nexit 0\n')
        EB_Gurobi._append_permanent_lib_rpath(SimpleNamespace(), str(script), self.additions)
        self.assertEqual(script.read_text(), '#!/bin/sh\nexit 0\n')

    def test_unfiltered_install_does_not_require_patchelf(self):
        block = object.__new__(EB_Gurobi)
        with patch.object(Tarball, 'post_processing_step'):
            with patch('easybuild.easyblocks.g.gurobi.build_option', return_value=[]):
                with patch('easybuild.easyblocks.g.gurobi.which') as which:
                    block.post_processing_step()
        which.assert_not_called()


if __name__ == '__main__':
    unittest.main()

    def test_only_library_directories_are_rewritten(self):
        """The vendor executables under bin/ keep their shipped dynamic section."""
        (self.root / 'lib64').mkdir()
        (self.root / 'matlab').mkdir()
        (self.root / 'lib64/libother.so').write_bytes(b'\x7fELF')
        (self.root / 'matlab/libmex.so').write_bytes(b'\x7fELF')
        (self.root / 'lib/README').write_text('not a library\n')
        os.symlink('libanswer.so', self.root / 'lib/libanswer.so.1')
        selected = EB_Gurobi._library_files(str(self.root))
        relative = sorted(os.path.relpath(path, self.root) for path in selected)
        self.assertEqual(relative, ['lib/README', 'lib/libanswer.so', 'lib64/libother.so', 'matlab/libmex.so'])
        self.assertNotIn(str(self.binary), selected)
        self.assertNotIn(str(self.root / 'bin/vendor/libhelper.so'), selected)
