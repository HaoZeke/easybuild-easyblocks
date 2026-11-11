"""CMake compiler flag definitions retain their complete shell argument."""
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from easybuild.tools import LooseVersion
import easybuild.easyblocks.generic.cmakemake as cmakemake
from easybuild.easyblocks.generic.cmakemake import CMakeMake


class _Config(dict):
    @contextmanager
    def disable_templating(self):
        yield self


class CmakeCachedFlags(unittest.TestCase):
    """Capture actual configure commands without executing a compiler or CMake."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        original = os.getcwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, original)
        self.block = object.__new__(CMakeMake)
        self.block.cfg = _Config(
            separate_build_dir=False, start_dir=self.temporary.name, srcdir=None,
            install_target_subdir=None, build_type=None, install_libdir=None,
            generator=None, build_shared_libs=None, abs_path_compilers=False,
            allow_system_boost=True, configopts='', preconfigopts='', configure_cmd='cmake',
        )
        self.block.cfg.toolchain = SimpleNamespace(options={})
        self.block.installdir = '/selected/install'
        self.block.log = mock.Mock()
        self.block._cmake_version = '4.2.1'
        self.block.check_python_paths = mock.Mock()
        self.flags = {
            'CFLAGS': '-O2 -ftree-vectorize -fopenmp -fno-math-errno',
            'CXXFLAGS': '-O3 -fopenmp -march=native -fno-math-errno',
            'FFLAGS': '-O2 -fopenmp -fno-math-errno',
        }

    def configure(self, cache=True, flags=None, configopts=''):
        if cache:
            Path('CMakeCache.txt').write_text('# Existing CMake cache fixture\n')
        self.block.cfg['configopts'] = configopts
        environment = dict(self.flags if flags is None else flags,
                           CC='mpicc', CXX='mpicxx', F90='gfortran')
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(cmakemake, 'setup_cmake_env'), \
                mock.patch.object(cmakemake, 'build_option', return_value=None), \
                mock.patch.object(cmakemake, 'get_software_root', return_value=None), \
                mock.patch.object(cmakemake, 'get_cmake_python_config_dict', return_value={}), \
                mock.patch.object(cmakemake, 'setvar'), \
                mock.patch.object(cmakemake, 'run_shell_cmd', return_value=SimpleNamespace(output='configured')) as run:
            self.assertEqual(self.block.configure_step(srcdir='/selected/source'), 'configured')
        run.assert_called_once()
        arguments = shlex.split(run.call_args[0][0])
        definitions = dict(argument[2:].split('=', 1) for argument in arguments if argument.startswith('-D'))
        return arguments, definitions

    def assert_flag_definitions(self, definitions, flags):
        for variable, definition in (('CFLAGS', 'CMAKE_C_FLAGS'), ('CXXFLAGS', 'CMAKE_CXX_FLAGS'),
                                     ('FFLAGS', 'CMAKE_Fortran_FLAGS')):
            self.assertEqual(definitions[definition], flags[variable])

    def test_cached_configure_retains_every_compiler_flag_in_one_argument(self):
        arguments, definitions = self.configure()
        self.assert_flag_definitions(definitions, self.flags)
        self.assertNotIn('-fno-math-errno', arguments)
        self.assertNotIn('-ftree-vectorize', arguments)

    def test_cached_flags_preserve_internal_quotes_and_spaces(self):
        flags = dict(self.flags, CFLAGS='-O2 -DNAME="two words" -I"/selected/include path"')
        _, definitions = self.configure(flags=flags)
        self.assert_flag_definitions(definitions, flags)

    def test_explicit_caller_flags_keep_precedence_over_cached_environment(self):
        _, definitions = self.configure(configopts='-DCMAKE_C_FLAGS="-O0 -g -DKEEP=1"')
        self.assertEqual(definitions['CMAKE_C_FLAGS'], '-O0 -g -DKEEP=1')
        self.assertEqual(definitions['CMAKE_CXX_FLAGS'], self.flags['CXXFLAGS'])
        self.assertEqual(definitions['CMAKE_Fortran_FLAGS'], self.flags['FFLAGS'])

    def test_empty_environment_flags_remain_explicit_empty_definitions(self):
        flags = dict.fromkeys(self.flags, '')
        _, definitions = self.configure(flags=flags)
        self.assert_flag_definitions(definitions, flags)

    def test_fresh_modern_configure_leaves_flag_initialization_to_cmake(self):
        _, definitions = self.configure(cache=False)
        for definition in ('CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS', 'CMAKE_Fortran_FLAGS'):
            self.assertNotIn(definition, definitions)

    def test_legacy_cmake_flag_definitions_use_the_same_argument_contract(self):
        self.block._cmake_version = '2.6.4'
        _, definitions = self.configure(cache=False)
        self.assert_flag_definitions(definitions, self.flags)


def suite(loader):
    """Return the compiler-flag configure command regressions."""
    return loader.loadTestsFromTestCase(CmakeCachedFlags)


if __name__ == '__main__':
    unittest.main()
