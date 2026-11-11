"""Controls for LAMMPS curl provider selection during linking and loading."""
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import easybuild.easyblocks.l.lammps as lammps
from easybuild.easyblocks.generic.cmakemake import CMakeMake
from easybuild.easyblocks.l.lammps import EB_LAMMPS, det_curl_libdir, det_lammps_exe_linker_flags


class _Config(dict):
    """Expose the toolchain property read by EasyBlock without initializing a build."""


class LammpsCurlLinking(unittest.TestCase):
    """Keep the selected curl ahead of compat while retaining caller and CUDA flags."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.curl = self.root / 'cURL'

    def library(self, relative='lib/libcurl.so'):
        path = self.curl / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic curl library fixture\n')
        return path

    @staticmethod
    def flags(option):
        arguments = shlex.split(option)
        if len(arguments) != 1:
            raise AssertionError('The CMake linker-flags definition must be one shell argument')
        name, value = arguments[0].split('=', 1)
        if name != '-DCMAKE_EXE_LINKER_FLAGS:STRING':
            raise AssertionError('Unexpected CMake option: ' + name)
        return shlex.split(value)

    def block(self, options=None, configopts=''):
        block = object.__new__(EB_LAMMPS)
        block.cfg = _Config(configopts=configopts)
        block.cfg.toolchain = SimpleNamespace(options={'rpath': True} if options is None else options,
                                             use_rpath=False, prepare_rpath_wrappers=mock.Mock())
        block.rpath_wrappers_dir = str(self.root / 'wrappers')
        return block

    def test_shared_library_directory_matches_the_dependency_selection(self):
        first = self.library()
        self.library('lib64/libcurl.so')
        self.assertEqual(det_curl_libdir(str(self.curl), '', 'so'), str(first.parent))

    def test_lib64_shared_layout_is_selected(self):
        library = self.library('lib64/libcurl.so')
        self.assertEqual(det_curl_libdir(str(self.curl), '', 'so'), str(library.parent))

    def test_static_layout_and_missing_dependency_add_no_runtime_priority(self):
        self.library('lib/libcurl.a')
        self.assertIsNone(det_curl_libdir(str(self.curl), '', 'so'))
        self.assertIsNone(det_curl_libdir(None, '', 'so'))

    def test_explicit_library_uses_its_own_directory(self):
        self.library()
        chosen = self.library('custom lib/libcurl.so.4.8.0')
        options = '-DCURL_LIBRARY:FILEPATH=' + shlex.quote(str(chosen))
        self.assertEqual(det_curl_libdir(str(self.curl), options, 'so'), str(chosen.parent))

    def test_config_mode_lookup_is_not_replaced_with_a_dependency_guess(self):
        self.library()
        for option in ('-DCURL_ROOT=/chosen/cURL', '-DCURL_DIR:PATH=/chosen/cURL/lib/cmake/CURL'):
            with self.subTest(option=option):
                self.assertIsNone(det_curl_libdir(str(self.curl), option, 'so'))

    def test_missing_explicit_library_does_not_fall_back_to_the_dependency(self):
        self.library()
        self.assertIsNone(det_curl_libdir(str(self.curl), '-DCURL_LIBRARY=/missing/libcurl.so', 'so'))

    def test_debug_library_configuration_and_last_definition_are_respected(self):
        release = self.library('release/libcurl.so')
        debug = self.library('debug/libcurl.so')
        definitions = '-DCURL_LIBRARY_RELEASE=%s -DCURL_LIBRARY_DEBUG=%s -DCMAKE_BUILD_TYPE=Debug' % (release, debug)
        self.assertEqual(det_curl_libdir(str(self.curl), definitions, 'so'), str(debug.parent))
        definitions = '-DCURL_LIBRARY=%s -D CURL_LIBRARY:FILEPATH=%s' % (release, debug)
        self.assertEqual(det_curl_libdir(str(self.curl), definitions, 'so'), str(debug.parent))

    def test_executable_flags_keep_user_flags_and_both_dependency_lookup_paths(self):
        user = '-DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,-z,now -pthread -Wl,-rpath,/working/lib"'
        option = det_lammps_exe_linker_flags(user, ['/selected/curl/lib', '/selected/CUDA/lib/stubs'])
        self.assertEqual(self.flags(option), ['-Wl,-rpath-link,/selected/curl/lib',
                                             '-Wl,-rpath-link,/selected/CUDA/lib/stubs',
                                             '-Wl,-z,now', '-pthread', '-Wl,-rpath,/working/lib'])

    def test_default_linker_search_paths_are_retained(self):
        defaults = '-L/working/first -L/working/second -pthread'
        option = det_lammps_exe_linker_flags('-DSOME_OTHER_OPTION=ON', ['/selected/curl/lib'], defaults)
        self.assertEqual(self.flags(option), ['-Wl,-rpath-link,/selected/curl/lib',
                                             '-L/working/first', '-L/working/second', '-pthread'])

    def test_explicit_empty_flags_do_not_reintroduce_environment_defaults(self):
        option = det_lammps_exe_linker_flags('-DCMAKE_EXE_LINKER_FLAGS=""', ['/selected/curl/lib'], '-L/unwanted')
        self.assertEqual(self.flags(option), ['-Wl,-rpath-link,/selected/curl/lib'])

    def test_no_lookup_paths_leave_user_options_alone(self):
        self.assertIsNone(det_lammps_exe_linker_flags('-DCMAKE_EXE_LINKER_FLAGS=-pthread', []))

    def test_lookup_paths_with_spaces_remain_single_compiler_arguments(self):
        option = det_lammps_exe_linker_flags('', ['/selected curl/lib', '/selected CUDA/lib/stubs'])
        self.assertEqual(self.flags(option), ['-Wl,-rpath-link,/selected curl/lib',
                                             '-Wl,-rpath-link,/selected CUDA/lib/stubs'])

    def test_lookup_path_merge_is_idempotent(self):
        paths = ['/selected/curl/lib', '/selected/CUDA/lib/stubs', '/selected/curl/lib']
        option = det_lammps_exe_linker_flags('-DCMAKE_EXE_LINKER_FLAGS=-pthread', paths)
        self.assertEqual(self.flags(option).count('-Wl,-rpath-link,/selected/curl/lib'), 1)
        self.assertIsNone(det_lammps_exe_linker_flags(option, paths))

    def test_selected_curl_lookup_precedes_a_retained_compat_hint(self):
        option = det_lammps_exe_linker_flags('-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/compat/lib',
                                             ['/selected/curl/lib'])
        self.assertEqual(self.flags(option), ['-Wl,-rpath-link,/selected/curl/lib', '-Wl,-rpath-link,/compat/lib'])

    def test_wrappers_are_created_once_after_modules_supply_the_selected_library(self):
        library = self.library()
        block = self.block()
        events = []
        old_includes = ['/intentional/user/priority', '$ORIGIN', '$ORIGIN/../lib']
        loaded = []

        def prepare(*args, **kwargs):
            events.append('prepare')
            self.assertFalse(block.toolchain.options['rpath'])
            block.rpath_include_dirs = list(old_includes)
            block.rpath_filter_dirs = ['/temporary/build', '/temporary/tmp']
            loaded.append(True)
            os.environ['PATH'] = '/loaded/toolchain/bin'

        def wrap(**kwargs):
            events.append('wrappers')
            self.assertTrue(block.toolchain.options['rpath'])
            self.assertEqual(kwargs['rpath_include_dirs'], old_includes + [str(library.parent)])
            self.assertEqual(kwargs['rpath_filter_dirs'], ['/temporary/build', '/temporary/tmp'])
            self.assertEqual(kwargs['rpath_wrappers_dir'], block.rpath_wrappers_dir)
            os.environ['PATH'] = '/selected/rpath_wrappers/gxx_wrapper:' + os.environ['PATH']

        def dependency_root(name):
            self.assertTrue(loaded, 'The dependency root must be read after modules are loaded')
            self.assertEqual(name, 'cURL')
            return str(self.curl)

        block.toolchain.prepare_rpath_wrappers.side_effect = wrap
        with mock.patch.dict(os.environ), mock.patch.object(CMakeMake, 'prepare_step', side_effect=prepare) as base, \
                mock.patch.object(lammps, 'build_option', return_value=True), \
                mock.patch.object(lammps, 'get_software_root', side_effect=dependency_root), \
                mock.patch.object(lammps, 'get_shared_lib_ext', return_value='so'), \
                mock.patch.object(lammps.env, 'copy_current_env', side_effect=lambda: dict(os.environ)):
            EB_LAMMPS._prepare_curl_rpath(block, start_dir=False)
            self.assertEqual(block.cached_build_env['PATH'], os.environ['PATH'])
            self.assertTrue(block.cached_build_env['PATH'].startswith('/selected/rpath_wrappers/'))
        base.assert_called_once_with(start_dir=False)
        block.toolchain.prepare_rpath_wrappers.assert_called_once()
        self.assertEqual(events, ['prepare', 'wrappers'])
        self.assertEqual(block.toolchain.options, {'rpath': True})
        self.assertTrue(block.toolchain.use_rpath)

    def test_prepare_failure_restores_the_toolchain_option(self):
        block = self.block()
        with mock.patch.object(CMakeMake, 'prepare_step', side_effect=RuntimeError('module failure')), \
                mock.patch.object(lammps, 'build_option', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'module failure'):
                EB_LAMMPS._prepare_curl_rpath(block)
        self.assertEqual(block.toolchain.options, {'rpath': True})
        block.toolchain.prepare_rpath_wrappers.assert_not_called()

    def test_missing_option_key_is_restored_without_adding_an_override(self):
        block = self.block(options={'usempi': True})

        def prepare():
            self.assertFalse(block.toolchain.options['rpath'])
            block.rpath_include_dirs = []
            block.rpath_filter_dirs = []

        with mock.patch.object(CMakeMake, 'prepare_step', side_effect=prepare), \
                mock.patch.object(lammps, 'build_option', return_value=True), \
                mock.patch.object(lammps, 'get_software_root', return_value=None), \
                mock.patch.object(lammps, 'get_shared_lib_ext', return_value='so'), \
                mock.patch.object(lammps.env, 'copy_current_env', return_value={}):
            EB_LAMMPS._prepare_curl_rpath(block)
        self.assertEqual(block.toolchain.options, {'usempi': True})
        block.toolchain.prepare_rpath_wrappers.assert_called_once()

    def test_explicitly_disabled_rpath_keeps_normal_base_preparation(self):
        for global_rpath, options in ((False, {'rpath': True}), (True, {'rpath': False})):
            with self.subTest(global_rpath=global_rpath, options=options):
                block = self.block(options=dict(options))
                with mock.patch.object(CMakeMake, 'prepare_step') as base, \
                        mock.patch.object(lammps, 'build_option', return_value=global_rpath), \
                        mock.patch.object(lammps, 'get_software_root') as root:
                    EB_LAMMPS._prepare_curl_rpath(block)
                base.assert_called_once_with()
                root.assert_not_called()
                self.assertEqual(block.toolchain.options, options)
                block.toolchain.prepare_rpath_wrappers.assert_not_called()


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(LammpsCurlLinking)


if __name__ == '__main__':
    unittest.main()
