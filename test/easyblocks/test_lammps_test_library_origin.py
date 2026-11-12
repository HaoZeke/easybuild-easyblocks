"""Registered LAMMPS tests resolve their own build library."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from easybuild.easyblocks.l.lammps import normalize_lammps_test_rpaths
from easybuild.tools.build_log import EasyBuildError


class LammpsTestLibraryOrigin(unittest.TestCase):
    """Exercise the loader against real fresh and stale libraries with one SONAME."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.build = self.root / 'build'
        self.stale = self.root / 'installed/lib'
        self.source.mkdir()
        self.stale.mkdir(parents=True)
        (self.source / 'fresh.c').write_text('int value(void) { return 42; }\n')
        (self.source / 'stale.c').write_text('int value(void) { return 13; }\n')
        (self.source / 'probe.c').write_text('int value(void); int main(void) { return value() != 42; }\n')
        subprocess.check_call(['cc', '-shared', '-fPIC', '-Wl,-soname,liblammps.so.0',
                               str(self.source / 'stale.c'), '-o', str(self.stale / 'liblammps.so.0')])
        cmake = """
cmake_minimum_required(VERSION 3.17)
project(library_origin C)
enable_testing()
add_library(lammps SHARED fresh.c)
set_target_properties(lammps PROPERTIES SOVERSION 0)
add_executable(test_probe probe.c)
add_executable(lmp probe.c)
foreach(target test_probe lmp)
  target_link_libraries(${target} PRIVATE lammps)
  set_target_properties(${target} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "${STALE}")
endforeach()
set_target_properties(test_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nested")
add_test(NAME registered COMMAND $<TARGET_FILE:test_probe>)
add_test(NAME installed COMMAND $<TARGET_FILE:lmp>)
install(TARGETS lmp RUNTIME DESTINATION bin)
"""
        (self.source / 'CMakeLists.txt').write_text(cmake)
        subprocess.check_call(['cmake', '-S', str(self.source), '-B', str(self.build),
                               '-DSTALE=' + str(self.stale)])
        subprocess.check_call(['cmake', '--build', str(self.build), '-j2'])
        self.executable = self.build / 'nested/test_probe'
        self.installed_target = self.build / 'lmp'

    @staticmethod
    def sections(path):
        """Read executable and constant sections without changing the ELF."""
        output = subprocess.check_output(['readelf', '-WS', str(path)], text=True)
        result = {}
        for section in ('.text', '.rodata'):
            match = re.search(r'\[\s*\d+\]\s+' + re.escape(section) +
                              r'\s+PROGBITS\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)', output)
            if match is None:
                continue
            with path.open('rb') as stream:
                stream.seek(int(match[1], 16))
                result[section] = hashlib.sha256(stream.read(int(match[2], 16))).hexdigest()
        return result

    def test_fresh_library_precedes_stale_install_without_changing_installed_targets(self):
        self.assertEqual(subprocess.call([str(self.executable)]), 1)
        before = self.sections(self.executable)
        needed = subprocess.check_output(['patchelf', '--print-needed', str(self.executable)])
        installed_bytes = self.installed_target.read_bytes()
        stale_bytes = (self.stale / 'liblammps.so.0').read_bytes()
        changed = normalize_lammps_test_rpaths(str(self.build), str(self.root / 'installed'))
        self.assertEqual(changed, [str(self.executable)])
        self.assertEqual(subprocess.call([str(self.executable)]), 0)
        self.assertEqual(self.sections(self.executable), before)
        self.assertEqual(subprocess.check_output(['patchelf', '--print-needed', str(self.executable)]), needed)
        self.assertEqual(subprocess.check_output(['patchelf', '--print-rpath', str(self.executable)], text=True),
                         '$ORIGIN/..:' + str(self.stale) + '\n')
        self.assertEqual(self.installed_target.read_bytes(), installed_bytes)
        self.assertEqual((self.stale / 'liblammps.so.0').read_bytes(), stale_bytes)
        self.assertEqual(normalize_lammps_test_rpaths(str(self.build), str(self.root / 'installed')), [])


    def test_rpath_tag_and_permanent_entries_are_preserved(self):
        subprocess.check_call(['patchelf', '--force-rpath', '--set-rpath', str(self.stale),
                               str(self.executable)])
        normalize_lammps_test_rpaths(str(self.build), str(self.root / 'installed'))
        dynamic = subprocess.check_output(['readelf', '-d', str(self.executable)], text=True)
        self.assertIn('(RPATH)', dynamic)
        self.assertNotIn('(RUNPATH)', dynamic)
        self.assertEqual(subprocess.call([str(self.executable)]), 0)
        self.assertEqual(subprocess.check_output(['patchelf', '--print-rpath', str(self.executable)], text=True),
                         '$ORIGIN/..:' + str(self.stale) + '\n')

    def test_test_stage_uses_the_configured_object_directory(self):
        from unittest import mock
        from easybuild.easyblocks.generic.cmakemake import CMakeMake
        from easybuild.easyblocks.l.lammps import EB_LAMMPS

        block = object.__new__(EB_LAMMPS)
        block.cfg = {'runtest': True, 'test_cmd': 'ctest'}
        block.builddir = str(self.root / 'control')
        block.separate_build_dir = str(self.build)
        block.installdir = str(self.root / 'installed')
        with mock.patch('easybuild.easyblocks.l.lammps.ensure_lammps_curl_ca_bundle'), \
                mock.patch('easybuild.easyblocks.l.lammps.normalize_lammps_test_rpaths') as normalize, \
                mock.patch.object(CMakeMake, 'test_step') as base:
            block.test_step()
        normalize.assert_called_once_with(str(self.build), str(self.root / 'installed'))
        base.assert_called_once_with()

    def test_shared_inode_is_refused_without_modification(self):
        backup = self.root / 'shared-elf'
        os.link(self.executable, backup)
        original = self.executable.read_bytes()
        with self.assertRaisesRegex(EasyBuildError, 'shared storage'):
            normalize_lammps_test_rpaths(str(self.build), str(self.root / 'installed'))
        self.assertEqual(self.executable.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), original)


def suite(loader):
    """Return loader-origin integration regressions."""
    return loader.loadTestsFromTestCase(LammpsTestLibraryOrigin)


if __name__ == '__main__':
    unittest.main()
