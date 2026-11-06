"""Exercise the Gurobi loader-path helper with original ELF fixtures."""

import hashlib
import os
from pathlib import Path
import subprocess
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

    def test_unfiltered_install_does_not_require_patchelf(self):
        block = object.__new__(EB_Gurobi)
        with patch.object(Tarball, 'post_processing_step'):
            with patch('easybuild.easyblocks.g.gurobi.build_option', return_value=[]):
                with patch('easybuild.easyblocks.g.gurobi.which') as which:
                    block.post_processing_step()
        which.assert_not_called()


if __name__ == '__main__':
    unittest.main()
