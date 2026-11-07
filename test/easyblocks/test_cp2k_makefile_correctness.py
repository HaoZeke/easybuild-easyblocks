"""Check Makefile compilation and regression-result handling for CP2K."""

import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from easybuild.easyblocks.c.cp2k import EB_CP2K
from easybuild.tools.build_log import EasyBuildError


class Config(dict):
    parallel = 1


class Cp2kMakefileCorrectness(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'src/hfxbase').mkdir(parents=True)
        (self.root / 'src/hfxbase/hfx_compression_core_methods.F').write_text('')

    def configure(self, gcc_version):
        block = SimpleNamespace(
            configure_common=lambda: {'DFLAGS': '', 'FCFLAGSOPT': '', 'FCFLAGSOPT2': ''},
            version='2025.2', cfg={'start_dir': str(self.root)}, make_instructions='',
        )
        with patch('easybuild.easyblocks.c.cp2k.get_software_version', return_value=gcc_version):
            options = EB_CP2K.configure_GCC_based(block)
        return block, options

    def test_gcc15_workaround_is_limited_to_the_packing_object(self):
        for version in ['13.3.0', '14.3.0', '15.0.0', '15.2.0', '16.1.0']:
            with self.subTest(version=version):
                block, options = self.configure(version)
                self.assertNotIn('-fno-tree-loop-vectorize', options['FCFLAGSOPT'])
                self.assertNotIn('-fno-tree-loop-vectorize', options['FCFLAGSOPT2'])
                if version.startswith('15.'):
                    self.assertIn('hfx_compression_core_methods.o: FCFLAGS += -fno-tree-loop-vectorize',
                                  block.make_instructions)
                else:
                    self.assertEqual(block.make_instructions, '')

        block, _ = self.configure('15.2.0')
        generated = EB_CP2K._generate_makefile(block, {'FC': 'gfortran', 'FCFLAGS': '-O2'})
        generated += '\n%.o: %.F\n\t$(FC) $(FCFLAGS) -c $< -o $@\n'
        makefile = self.root / 'Makefile'
        makefile.write_text(generated)
        for name in ['hfx_compression_core_methods.F', 'ordinary.F']:
            (self.root / name).write_text('')
        result = subprocess.run(
            ['make', '-n', '-f', str(makefile), 'hfx_compression_core_methods.o', 'ordinary.o'],
            cwd=self.root, check=True, capture_output=True, text=True,
        )
        commands = result.stdout.splitlines()
        packing = next(line for line in commands if '-o hfx_compression_core_methods.o' in line)
        ordinary = next(line for line in commands if '-o ordinary.o' in line)
        self.assertIn('-fno-tree-loop-vectorize', packing)
        self.assertNotIn('-fno-tree-loop-vectorize', ordinary)

    def test_make_c_command_expands_parallel_and_optional_definitions(self):
        block = SimpleNamespace(
            cfg={'type': 'psmp', 'typeopt': True, 'extracflags': '', 'extradflags': ''},
            toolchain=SimpleNamespace(get_flag=lambda name: '-fopenmp'),
            openmp='', fpic='-fPIC', debug='', log=Mock(), make_instructions='',
        )
        environment = {'CFLAGS': '-O2 -DFROM_TOOLCHAIN', 'CPPFLAGS': '', 'LDFLAGS': '',
                       'FFLAGS': '-O2', 'MPICC': 'mpicc', 'MPIF90': 'mpif90'}

        def root(name):
            return '/modules/OpenMPI' if name == 'OpenMPI' else None

        with patch.dict(os.environ, environment):
            with patch('easybuild.easyblocks.c.cp2k.get_software_root', side_effect=root):
                options = EB_CP2K.configure_common(block)
        options['DFLAGS'] += ' -D__HDF5 -D__SIRIUS'
        generated = EB_CP2K._generate_makefile(block, options)
        generated += '\nfeature.o: feature.c\n\t$(CC) $(CFLAGS) -c $< -o $@\n'
        makefile = self.root / 'Makefile'
        makefile.write_text(generated)
        (self.root / 'feature.c').write_text('int feature;\n')
        result = subprocess.run(
            ['make', '-n', '-f', str(makefile), 'feature.o'], cwd=self.root,
            check=True, capture_output=True, text=True,
        )
        command = result.stdout.strip()
        self.assertIn('-D__parallel', command)
        self.assertIn('-D__HDF5', command)
        self.assertIn('-D__SIRIUS', command)
        self.assertIn('-DFROM_TOOLCHAIN', command)

    def run_summary(self, wrong, failed):
        (self.root / 'tests').mkdir(exist_ok=True)
        (self.root / 'tests/do_regtest.py').write_text('')
        cfg = Config(runtest=True, start_dir=str(self.root), omp_num_threads=1,
                     ignore_regtest_fails=False)
        block = SimpleNamespace(
            cmake_build=False, cfg=cfg, builddir=str(self.root), version='2025.2',
            log=Mock(), postmsg='', py_regtest_cmd=lambda *args: 'regtest-fixture',
        )
        output = ('number of  tests 100\nnumber of failed tests %d\n'
                  'number of wrong tests %d\nnumber of correct tests %d\n') % (
                      failed, wrong, 100 - failed - wrong)
        with patch('easybuild.easyblocks.c.cp2k.setvar'):
            with patch('easybuild.easyblocks.c.cp2k.change_dir'):
                with patch('easybuild.easyblocks.c.cp2k.build_option', return_value=True):
                    with patch('easybuild.easyblocks.c.cp2k.get_avail_core_count', return_value=1):
                        with patch('easybuild.easyblocks.c.cp2k.run_shell_cmd',
                                   return_value=SimpleNamespace(exit_code=0, output=output)):
                            EB_CP2K.test_step(block)

    def test_single_wrong_result_is_rejected_below_ten_percent(self):
        with self.assertRaises(EasyBuildError):
            self.run_summary(wrong=1, failed=0)

    def test_single_failed_result_is_rejected(self):
        with self.assertRaises(EasyBuildError):
            self.run_summary(wrong=0, failed=1)

    def test_all_correct_summary_is_accepted(self):
        self.run_summary(wrong=0, failed=0)


if __name__ == '__main__':
    unittest.main()
