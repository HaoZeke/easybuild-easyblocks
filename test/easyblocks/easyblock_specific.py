##
# Copyright 2019-2026 Ghent University
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
Unit tests for specific easyblocks.

@author: Kenneth Hoste (Ghent University)
"""
import copy
import os
import re
import stat
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import TestLoader, TextTestRunner, mock
from test.easyblocks.module import cleanup

import easybuild.tools.options as eboptions
import easybuild.tools.tomllib as tomllib
import easybuild.easyblocks.c.cp2k as cp2k
import easybuild.easyblocks.generic.pythonpackage as pythonpackage
import easybuild.easyblocks.generic.cargo as cargo
import easybuild.easyblocks.l.lammps as lammps
import easybuild.easyblocks.l.libint as libint
import easybuild.easyblocks.p.python as python
import easybuild.easyblocks.p.pytorch as pytorch
from easybuild.base.testing import TestCase
from easybuild.easyblocks.generic.cmakemake import CMakeMake, det_cmake_version
from easybuild.easyblocks.generic.toolchain import Toolchain
from easybuild.easyblocks.tensorflow import det_binutils_bin_path
from easybuild.framework.easyblock import EasyBlock, get_easyblock_instance
from easybuild.framework.easyconfig.easyconfig import process_easyconfig
from easybuild.tools import config
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.config import GENERAL_CLASS, get_module_syntax
from easybuild.tools.environment import modify_env
from easybuild.tools.filetools import adjust_permissions, mkdir, move_file, remove_dir, symlink, write_file
from easybuild.tools.modules import modules_tool
from easybuild.tools.options import set_tmpdir
from easybuild.tools.run import RunShellCmdResult
from easybuild.tools.systemtools import get_shared_lib_ext
from easybuild.tools.toolchain.toolchain import RPATH_WRAPPERS_SUBDIR


class EasyBlockSpecificTest(TestCase):
    """ Baseclass for easyblock testcases """

    # initialize configuration (required for e.g. default modules_tool setting)
    eb_go = eboptions.parse_options(args=[])
    config.init(eb_go.options, eb_go.get_options_by_section('config'))
    build_options = {
        'suffix_modules_path': GENERAL_CLASS,
        'valid_module_classes': config.module_classes(),
        'valid_stops': [x[0] for x in EasyBlock.get_steps()],
    }
    config.init_build_options(build_options=build_options)
    set_tmpdir()
    del eb_go

    def setUp(self):
        """Test setup."""
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

        self.orig_environ = copy.deepcopy(os.environ)
        self.orig_pythonpackage_run_shell_cmd = pythonpackage.run_shell_cmd

    def tearDown(self):
        """Test cleanup."""
        remove_dir(self.tmpdir)

        pythonpackage.run_shell_cmd = self.orig_pythonpackage_run_shell_cmd

        # restore original environment
        modify_env(os.environ, self.orig_environ, verbose=False)

        super().tearDown()

    def test_toolchain_external_modules(self):
        """Test use of Toolchain easyblock with external modules."""

        external_modules = ['gcc/8.3.0', 'openmpi/4.0.2', 'openblas/0.3.7', 'fftw/3.3.8', 'scalapack/2.0.2']
        external_modules_metadata = {
            # all metadata for gcc/8.3.0
            'gcc/8.3.0': {
                'name': ['GCC'],
                'version': ['8.3.0'],
                'prefix': '/software/gcc/8.3.0',
            },
            # only name/version for openmpi/4.0.2
            'openmpi/4.0.2': {
                'name': ['OpenMPI'],
                'version': ['4.0.2'],
            },
            # only name/prefix for openblas/0.3.7
            'openblas/0.3.7': {
                'name': ['OpenBLAS'],
                'prefix': '/software/openblas/0.3.7',
            },
            # only version/prefix for fftw/3.3.8 (no name)
            'fftw/3.3.8': {
                'version': ['3.3.8'],
                'prefix': '/software/fftw/3.3.8',
            },
            # no metadata for scalapack/2.0.2
        }

        # initialize configuration
        cleanup()
        eb_go = eboptions.parse_options(args=['--installpath=%s' % self.tmpdir])
        config.init(eb_go.options, eb_go.get_options_by_section('config'))
        build_options = {
            'external_modules_metadata': external_modules_metadata,
            'valid_module_classes': config.module_classes(),
        }
        config.init_build_options(build_options=build_options)
        set_tmpdir()
        del eb_go

        modtool = modules_tool()

        # make sure no $EBROOT* or $EBVERSION* environment variables are set in current environment
        for key in os.environ:
            if any(key.startswith(x) for x in ['EBROOT', 'EBVERSION']):
                del os.environ[key]

        # create dummy module file for each of the external modules
        test_mod_path = os.path.join(self.tmpdir, 'modules', 'all')
        for mod in external_modules:
            write_file(os.path.join(test_mod_path, mod), "#%Module")

        modtool.use(test_mod_path)

        # test easyconfig file to install toolchain that uses external modules,
        # and enables set_env_external_modules
        test_ec_path = os.path.join(self.tmpdir, 'test.eb')
        test_ec_txt = '\n'.join([
            "easyblock = 'Toolchain'",
            "name = 'test-toolchain'",
            "version = '1.2.3'",
            "homepage = 'https://example.com'",
            "description = 'just a test'",
            "toolchain = SYSTEM",
            "dependencies = [",
            "   ('gcc/8.3.0', EXTERNAL_MODULE),",
            "   ('openmpi/4.0.2', EXTERNAL_MODULE),",
            "   ('openblas/0.3.7', EXTERNAL_MODULE),",
            "   ('fftw/3.3.8', EXTERNAL_MODULE),",
            "   ('scalapack/2.0.2', EXTERNAL_MODULE),",
            "]",
            "set_env_external_modules = True",
            "moduleclass = 'toolchain'",
        ])
        write_file(test_ec_path, test_ec_txt)
        test_ec = process_easyconfig(test_ec_path)[0]

        # create easyblock & install module via run_all_steps
        tc_inst = get_easyblock_instance(test_ec)
        self.assertTrue(isinstance(tc_inst, Toolchain))
        self.mock_stdout(True)
        tc_inst.run_all_steps(False)
        self.mock_stdout(False)

        # make sure expected module file exists
        test_mod = os.path.join(test_mod_path, 'test-toolchain', '1.2.3')
        if get_module_syntax() == 'Lua':
            test_mod += '.lua'
        self.assertTrue(os.path.exists(test_mod))

        # load test-toolchain/1.2.3 module to get environment variable to check for defined
        modtool.load(['test-toolchain/1.2.3'])

        # check whether expected environment variables are defined
        self.assertEqual(os.environ.pop('EBROOTGCC'), '/software/gcc/8.3.0')
        self.assertEqual(os.environ.pop('EBVERSIONGCC'), '8.3.0')
        self.assertEqual(os.environ.pop('EBVERSIONOPENMPI'), '4.0.2')
        self.assertEqual(os.environ.pop('EBROOTOPENBLAS'), '/software/openblas/0.3.7')
        undefined_env_vars = [
            'EBROOTOPENMPI',  # no prefix in metadata
            'EBVERSIONOPENBLAS'  # no version in metadata
            'EBROOTFFTW', 'EBVERSIONFFTW',  # no name in metadata
            'EBROOTSCALAPACK', 'EBVERSIONSCALAPACK',  # no metadata
        ]
        for env_var in undefined_env_vars:
            self.assertTrue(os.getenv(env_var) is None)

        # make sure no unexpected $EBROOT* or $EBVERSION* environment variables were defined
        del os.environ['EBROOTTESTMINTOOLCHAIN']
        del os.environ['EBVERSIONTESTMINTOOLCHAIN']
        extra_eb_env_vars = []
        for key in os.environ:
            if any(key.startswith(x) for x in ['EBROOT', 'EBVERSION']):
                extra_eb_env_vars.append(key)
        self.assertEqual(extra_eb_env_vars, [])

    def test_det_cmake_version(self):
        """Tests for det_cmake_version function provided along with CMakeMake generic easyblock."""

        # set up fake 'cmake' command
        cmake_cmd = os.path.join(self.tmpdir, 'cmake')
        write_file(cmake_cmd, '#!/bin/bash\nexit 1')
        adjust_permissions(cmake_cmd, stat.S_IXUSR)

        os.environ['PATH'] = '%s:%s' % (self.tmpdir, os.getenv('PATH'))

        self.assertRaisesRegex(EasyBuildError, "Failed to determine CMake version", det_cmake_version)

        # if $EBVERSIONCMAKE is defined (by loaded CMake module), that's picked up
        os.environ['EBVERSIONCMAKE'] = '1.2.3'
        self.assertEqual(det_cmake_version(), '1.2.3')

        del os.environ['EBVERSIONCMAKE']

        # output of 'cmake --version' as produced by CMake 2.x < 2.4.0
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "CMake version 2.3.0"
        """))
        self.assertEqual(det_cmake_version(), '2.3.0')

        # output of 'cmake --version' as produced by CMake 2.x >= 2.4.0
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "cmake version 2.4.1"
        """))
        self.assertEqual(det_cmake_version(), '2.4.1')

        # output of 'cmake --version' as produced by CMake 3.x
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "cmake version 3.15.3"
        echo ""
        echo "CMake suite maintained and supported by Kitware (kitware.com/cmake)."
        """))
        self.assertEqual(det_cmake_version(), '3.15.3')

        # also consider release candidate versions
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "cmake version 1.2.3-rc4"
        """))
        self.assertEqual(det_cmake_version(), '1.2.3-rc4')

    def test_det_installed_python_packages(self):
        """
        Test det_installed_python_packages function providyed by PythonPackage easyblock
        """
        pkg1 = None
        res = python.det_installed_python_packages(python_cmd=sys.executable)
        # we can't make too much assumptions on which installed Python packages are found
        self.assertTrue(isinstance(res, list))
        if res:
            pkg1_name = res[0]
            self.assertTrue(isinstance(pkg1_name, str))

        res_detailed = python.det_installed_python_packages(python_cmd=sys.executable, names_only=False)
        self.assertTrue(isinstance(res_detailed, list))
        if res_detailed:
            pkg1 = res_detailed[0]
            self.assertTrue(isinstance(pkg1, dict))
            self.assertTrue(sorted(pkg1.keys()), ['name', 'version'])
            self.assertEqual(pkg1['name'], pkg1_name)
            regex = re.compile('^[0-9].*')
            ver = pkg1['version']
            self.assertTrue(regex.match(ver), f"Pattern {regex.pattern} matches for pkg version: {ver}")

        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            stderr = None
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}]'
                stderr = "DEPRECATION: Python 2.7 reached the end of its life on January 1st, 2020"
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=stderr, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)
        python.run_shell_cmd = mocked_run_shell_cmd_pip
        res = python.det_installed_python_packages(python_cmd=sys.executable)
        self.assertEqual(res, ['example'])

    def test_det_py_install_scheme(self):
        """Test det_py_install_scheme function provided by PythonPackage easyblock."""
        res = pythonpackage.det_py_install_scheme(sys.executable)
        self.assertTrue(isinstance(res, str))

        # symlink currently used python command to 'python', so we can also test det_py_install_scheme with default;
        # this is required because 'python' command may not be available
        symlink(sys.executable, os.path.join(self.tmpdir, 'python'))
        os.environ['PATH'] = '%s:%s' % (self.tmpdir, os.getenv('PATH'))

        res = pythonpackage.det_py_install_scheme()
        self.assertTrue(isinstance(res, str))

    def test_cargo_get_workspace_members(self):
        """Test get_workspace_members in the Cargo easyblock"""
        # Simple crate
        toml_text = textwrap.dedent("""
            [package]
            name = 'my_crate'
            version = "0.1.0"
            edition = "2021"
            description = 'desc'
            documentation = "url"
            license = "MIT"
        """)
        members = cargo._get_workspace_members(tomllib.loads(toml_text))
        self.assertIsNone(members)

        # Virtual manifest
        toml_text = textwrap.dedent("""
            [workspace]
            members = [
                "reqwest-middleware",
                "reqwest-tracing",
                "reqwest-retry",
            ]
        """)
        members = cargo._get_workspace_members(tomllib.loads(toml_text))
        self.assertEqual(members, ["reqwest-middleware", "reqwest-tracing", "reqwest-retry"])

        # Workspace (root is a package too)
        toml_text = textwrap.dedent("""
            [package]
            name = "nothing-linux-ui"
            version = "0.0.2"
            edition = "2021"
            authors = ["sn99"]

            [workspace]
            members = ["nothing", "src-tauri"]

            [dependencies]
            leptos = { version = "0.6", features = ["csr"] }
        """)
        members = cargo._get_workspace_members(tomllib.loads(toml_text))
        self.assertEqual(members, ["nothing", "src-tauri"])

    def test_cargo_merge_sub_crate(self):
        """Test merge_sub_crate in the Cargo easyblock"""
        crate_dir = Path(tempfile.mkdtemp())
        cargo_toml = crate_dir / 'Cargo.toml'
        ws_parsed = tomllib.loads("""
            [workspace]
            members = ["bar"]

            [workspace.package]
            version = "1.2.3"
            authors = ["Nice Folks"]
            description = "A short description of my package"
            documentation = "https://example.com/bar"

            [workspace.dependencies]
            regex = { version = "1.6.0", default-features = false, features = ["std"] }
            cc = "1.0.73"
            rand = "0.8.5"

            [workspace.lints.rust]
            unsafe_code = "forbid"
        """)
        cargo_toml.write_text("""
            [package]
            name = "bar"
            version.workspace = true
            authors.workspace = true
            description.workspace = true
            documentation.workspace = true

            # Unrelated line that looks like a workspace key
            dummy = "Uses regex=123 and regex = 456 and not foo.workspace = true"

            [dependencies]
            foo = { version = "42" }
            # Overwrite 'features' value
            regex = { workspace = true, features = ["unicode"] }

            [build-dependencies]
            cc.workspace = true

            [dev-dependencies]
            rand = { workspace = true }

            [lints]
            workspace = true
        """)
        cargo._merge_sub_crate(cargo_toml, ws_parsed)
        self.assertEqual(tomllib.loads(cargo_toml.read_text()), tomllib.loads("""
            [package]
            name = "bar"
            version = "1.2.3"
            authors = ["Nice Folks"]
            description = "A short description of my package"
            documentation = "https://example.com/bar"

            dummy = "Uses regex=123 and regex = 456 and not foo.workspace = true"

            [dependencies]
            foo = { version = "42" }
            regex = { version = "1.6.0", default-features = false, features = ["unicode"] }

            [build-dependencies]
            cc = "1.0.73"

            [dev-dependencies]
            rand = "0.8.5"

            [lints.rust]
            unsafe_code = "forbid"
        """))

        # Only dict-style workspace dependency
        cargo_toml.write_text("""
            [package]
            name = "bar"

            [dependencies]
            regex = { workspace = true }
        """)
        cargo._merge_sub_crate(cargo_toml, ws_parsed)
        self.assertEqual(tomllib.loads(cargo_toml.read_text()), tomllib.loads("""
            [package]
            name = "bar"

            [dependencies]
            regex = { version = "1.6.0", default-features = false, features = ["std"] }
        """))

    def test_handle_local_py_install_scheme(self):
        """Test handle_local_py_install_scheme function provided by PythonPackage easyblock."""

        # test with empty dir, should be fine
        pythonpackage.handle_local_py_install_scheme(self.tmpdir)
        self.assertEqual(os.listdir(self.tmpdir), [])

        # create normal structure (no 'local' subdir), shouldn't cause trouble
        bindir = os.path.join(self.tmpdir, 'bin')
        mkdir(bindir)
        write_file(os.path.join(bindir, 'test'), 'test')
        libdir = os.path.join(self.tmpdir, 'lib')
        pyshortver = '.'.join(str(x) for x in sys.version_info[:2])
        pylibdir = os.path.join(libdir, 'python' + pyshortver, 'site-packages')
        mkdir(pylibdir, parents=True)
        write_file(os.path.join(pylibdir, 'test.py'), "import os")

        pythonpackage.handle_local_py_install_scheme(self.tmpdir)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ['bin', 'lib'])

        # move bin + lib into local/, check whether expected symlinks are created
        local_subdir = os.path.join(self.tmpdir, 'local')
        mkdir(local_subdir)
        for subdir in (bindir, libdir):
            move_file(subdir, os.path.join(local_subdir, os.path.basename(subdir)))
        self.assertEqual(os.listdir(self.tmpdir), ['local'])

        pythonpackage.handle_local_py_install_scheme(self.tmpdir)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ['bin', 'lib', 'local'])
        self.assertTrue(os.path.islink(bindir) and os.path.samefile(bindir, os.path.join(local_subdir, 'bin')))
        self.assertTrue(os.path.islink(libdir) and os.path.samefile(libdir, os.path.join(local_subdir, 'lib')))
        self.assertTrue(os.path.exists(os.path.join(bindir, 'test')))
        local_test_py = os.path.join(libdir, 'python' + pyshortver, 'site-packages', 'test.py')
        self.assertTrue(os.path.exists(local_test_py))

    def test_partial_normalize_pip(self):
        """Test partial_normalize_pip function provided by EB_Python easyblock."""

        self.assertEqual(python.partial_normalize_pip('friendly-bard'), 'friendly-bard')  # normalized form
        self.assertEqual(python.partial_normalize_pip('Friendly-Bard'), 'friendly-bard')  # uppercase -> lowercase
        self.assertEqual(python.partial_normalize_pip('friendly_bard'), 'friendly-bard')  # underscore -> hyphen
        self.assertEqual(python.partial_normalize_pip('friendly.bard'), 'friendly.bard')  # dots are not normalized
        # multiple consecutive hyphens are merged into a single hyphen
        self.assertEqual(python.partial_normalize_pip('friendly--bard'), 'friendly-bard')
        # multiple consecutive underscores are merged and converted into a single hyphen
        self.assertEqual(python.partial_normalize_pip('friendly__bard'), 'friendly-bard')
        # multiple consecutive underscores and hyphens are merged and converted into a single hyphen
        self.assertEqual(python.partial_normalize_pip('FrIeNdLy--__--bArD'), 'friendly-bard')

    def test_run_pip_check(self):
        """Test run_pip_check function provided by EB_Python easyblock."""

        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip check" in cmd:
                output = "No broken requirements found."
            elif "pip --version" in cmd:
                output = "pip 20.0"
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        with self.mocked_stdout_stderr():
            python.run_pip_check(python_cmd=sys.executable)

        # inject all possible errors
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip check" in cmd:
                output = "foo-1.2.3 requires bar-4.5.6, which is not installed."
                exit_code = 1
            elif "pip --version" in cmd:
                output = "pip 20.0"
                exit_code = 0
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=exit_code, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = '\n'.join([
            "pip check.*failed.*",
            "foo.*requires.*bar.*not installed.*",
        ])
        with self.mocked_stdout_stderr():
            self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_check,
                                   python_cmd=sys.executable)

        # invalid pip version
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            return RunShellCmdResult(cmd=cmd, exit_code=0, output="1.2.3", stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = "Failed to determine pip version!"
        self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_check, python_cmd=sys.executable)

    def test_run_pip_list(self):
        """Test run_pip_list function provided by EB_Python easyblock."""

        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}]'
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        with self.mocked_stdout_stderr():
            python.run_pip_list([], python_cmd=sys.executable)

        # test ignored unversioned Python packages
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "zero", "version": "0.0.0"}, {"name": "example-pkg", "version": "1.2.3"}]'
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        with self.mocked_stdout_stderr():
            python.run_pip_list([('example_pkg', '1.2.3')], python_cmd=sys.executable, unversioned_packages=('zero', ))

        with self.mocked_stdout_stderr():
            python.run_pip_list([('example.pkg', '1.2.3')], python_cmd=sys.executable, unversioned_packages={'zero'})

        # inject all possible errors with unversioned packages
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}, {"name": "wrong", "version": "0.0.0"}]'
                exit_code = 0
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=exit_code, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = '\n'.join([
            r"Package 'example'.*version of 1\.2\.3 which is valid.*",
            "Package 'nosuchpkg' in unversioned_packages was not found in the installed packages.*",
            r".*not installed correctly.*version of '0\.0\.0':",
            "wrong",
        ])
        with self.mocked_stdout_stderr():
            self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_list, [],
                                   python_cmd=sys.executable, unversioned_packages=['example', 'nosuchpkg'])

        # inject errors with mismatched packages name or version
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}, {"name": "wrong-version", "version": "1.1.1"}]'
                exit_code = 0
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=exit_code, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = '\n'.join([
            r"The following Python packages were likely specified with a wrong name because they are missing.*",
            r"wrong-name.*",
            r"The following Python packages were likely specified with a wrong version.*",
            r"wrong-version 5.6.7.*",
        ])
        with self.mocked_stdout_stderr():
            self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_list,
                                   [('wrong_name', '1.2.3'), ('wrong_version', '5.6.7')],
                                   python_cmd=sys.executable, strict_check=True)

    def test_symlink_dist_site_packages(self):
        """Test symlink_dist_site_packages provided by PythonPackage easyblock."""
        pyshortver = '.'.join(str(x) for x in sys.version_info[:2])
        pylibdir_lib_dist = os.path.join('lib', 'python' + pyshortver, 'dist-packages')
        pylibdir_lib64_site = os.path.join('lib64', 'python' + pyshortver, 'site-packages')
        pylibdirs = [pylibdir_lib_dist, pylibdir_lib64_site]

        # first test on empty dir
        pythonpackage.symlink_dist_site_packages(self.tmpdir, pylibdirs)
        self.assertEqual(os.listdir(self.tmpdir), [])

        # check intended usage: dist-packages exists, site-packages doesn't => symlink created
        lib64_site_path = os.path.join(self.tmpdir, pylibdir_lib_dist)
        mkdir(os.path.join(self.tmpdir, pylibdir_lib_dist), parents=True)
        # also create (empty) site-packages directory, which should get replaced by a symlink to dist-packages
        mkdir(os.path.join(self.tmpdir, os.path.dirname(pylibdir_lib_dist), 'site-packages'), parents=True)

        lib64_site_path = os.path.join(self.tmpdir, pylibdir_lib64_site)
        mkdir(lib64_site_path, parents=True)
        # populate site-packages under lib64, because it'll get removed if empty
        write_file(os.path.join(lib64_site_path, 'test.py'), "import os")

        pythonpackage.symlink_dist_site_packages(self.tmpdir, pylibdirs)

        # check for expected directory structure
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ['lib', 'lib64'])
        path = os.path.join(self.tmpdir, 'lib')
        self.assertEqual(os.listdir(path), ['python' + pyshortver])
        path = os.path.join(path, 'python' + pyshortver)
        self.assertEqual(sorted(os.listdir(path)), ['dist-packages', 'site-packages'])

        # check for site-packages -> dist-packages symlink
        dist_pkgs = os.path.join(path, 'dist-packages')
        self.assertTrue(os.path.isdir(dist_pkgs))
        self.assertFalse(os.path.islink(dist_pkgs))
        site_pkgs = os.path.join(path, 'site-packages')
        self.assertTrue(os.path.isdir(site_pkgs))
        self.assertTrue(os.path.islink(site_pkgs))

        # if (non-empty) site-packages dir was there, no changes were made
        lib64_path = os.path.dirname(lib64_site_path)
        self.assertEqual(sorted(os.listdir(lib64_path)), ['site-packages'])
        self.assertTrue(os.path.isdir(lib64_site_path))
        self.assertFalse(os.path.islink(lib64_site_path))

    def test_det_binutils_bin_path(self):
        """Test det_binutils_bin_path provided by TensorFlow easyblock."""
        binutils_bin_path = det_binutils_bin_path()
        self.assertTrue(os.path.join(binutils_bin_path, 'ld'))
        self.assertFalse(RPATH_WRAPPERS_SUBDIR in binutils_bin_path)

        wrappers_dir = os.path.join(self.tmpdir, 'fake_wrappers', RPATH_WRAPPERS_SUBDIR)

        # put fake wrappers in place for a couple of binutils command
        binutils_cmds = ('as', 'ld', 'nm', 'objdump')
        for cmd in binutils_cmds:
            wrapper_dir = os.path.join(wrappers_dir, '%s.wrapper' % cmd)
            os.environ['PATH'] = wrapper_dir + ':' + os.getenv('PATH')
            fake_wrapper = os.path.join(wrapper_dir, cmd)
            # fake contents for RPATH wrapper script, enough to fool Toolchain.is_rpath_wrapper
            fake_wrapper_txt = '"$RPATH_ARGS_PY" "$CMD"'
            write_file(fake_wrapper, fake_wrapper_txt)
            adjust_permissions(fake_wrapper, stat.S_IXUSR)

        # if $EBROOTBINUTILS is set, binutils commands to consider is determined by contents of $EBROOTBINUTILS/bin
        binutils_root = os.path.join(self.tmpdir, 'binutils_root')
        for cmd in binutils_cmds[:2]:
            cmd_path = os.path.join(binutils_root, 'bin', cmd)
            write_file(cmd_path, '#!/bin/bash\necho %s' % cmd)
            adjust_permissions(cmd_path, stat.S_IXUSR)
        os.environ['EBROOTBINUTILS'] = binutils_root

        binutils_bin_path = det_binutils_bin_path()
        self.assertEqual(os.path.basename(binutils_bin_path), RPATH_WRAPPERS_SUBDIR)
        self.assertEqual(sorted(os.listdir(binutils_bin_path)), ['as', 'ld'])
        for cmd in binutils_cmds[:2]:
            cmd_path = os.path.join(binutils_bin_path, cmd)
            self.assertTrue(os.path.islink(cmd_path))
            expected_target = os.path.join(wrappers_dir, '%s.wrapper' % cmd, cmd)
            self.assertEqual(os.path.realpath(cmd_path), os.path.realpath(expected_target))

        del os.environ['EBROOTBINUTILS']

        # if $EBROOTBINUTILS is not set, a pre-defined list of known binutils commands is used (KNOWN_BINUTILS constant)
        binutils_bin_path = det_binutils_bin_path()
        self.assertEqual(os.path.basename(binutils_bin_path), RPATH_WRAPPERS_SUBDIR)
        found_cmds = os.listdir(binutils_bin_path)
        self.assertTrue(all(x in found_cmds for x in binutils_cmds))
        for cmd in binutils_cmds:
            cmd_path = os.path.join(binutils_bin_path, cmd)
            self.assertTrue(os.path.islink(cmd_path))
            expected_target = os.path.join(wrappers_dir, '%s.wrapper' % cmd, cmd)
            self.assertEqual(os.path.realpath(cmd_path), os.path.realpath(expected_target))

    def test_translate_lammps_version(self):
        """Test translate_lammps_version function from LAMMPS easyblock"""
        lammps_versions = {
            '23Jun2022': '2022.06.23',
            '2Aug2023_update2': '2023.08.02.2',
            '29Aug2024': '2024.08.29',
            '29Aug2024_update2': '2024.08.29.2',
            '28Oct2024': '2024.10.28',
        }
        for key, expected_version in lammps_versions.items():
            self.assertEqual(lammps.translate_lammps_version(key), expected_version)

        version_file = os.path.join(self.tmpdir, 'src', 'version.h')
        version_txt = '\n'.join([
            '#define LAMMPS_VERSION "2 Apr 2025"',
            '#define LAMMPS_UPDATE "Development"',
        ])
        write_file(version_file, version_txt)

        self.assertEqual(lammps.translate_lammps_version('d3adb33f', path=self.tmpdir), '2025.04.02')
        self.assertEqual(lammps.translate_lammps_version('devel', path=self.tmpdir), '2025.04.02')

        version_file = os.path.join(self.tmpdir, 'src', 'version.h')
        version_txt = '\n'.join([
            '#define LAMMPS_VERSION "2 Apr 2025"',
            '#define LAMMPS_UPDATE "Update 3"',
        ])
        write_file(version_file, version_txt)

        self.assertEqual(lammps.translate_lammps_version('d3adb33f', path=self.tmpdir), '2025.04.02.3')

    def test_cp2k_py_regtest_cmd(self):
        """Test construction of the command for the Python regression test driver of CP2K."""
        from types import SimpleNamespace

        startdir = os.path.join(self.tmpdir, 'cp2k-2025.2')
        mkdir(os.path.join(startdir, 'tests'), parents=True)

        def mpi_cmd_for(cmd, nr_ranks):
            return 'mpirun -n %s %s' % (nr_ranks, cmd)

        eb = SimpleNamespace(
            cfg={'start_dir': startdir, 'maxtasks': 4, 'omp_num_threads': 2, 'type': 'psmp'},
            typearch='Linux-foss',
            toolchain=SimpleNamespace(mpi_cmd_for=mpi_cmd_for),
        )

        # CP2K 2025.1 and later: executable directory and mpiexec template with {N}
        script = os.path.join(startdir, 'tests', 'do_regtest.py')
        write_file(script, 'add_argument("binary_dir")\nadd_argument("--mpiexec", default="mpiexec -n {N}")\n')
        cmd = cp2k.EB_CP2K.py_regtest_cmd(eb, script, 4)
        exe_dir = os.path.join(startdir, 'exe', 'Linux-foss')
        expected = "%s %s --mpiranks 4 --maxtasks 4 --ompthreads 2 --mpiexec 'mpirun -n {N}' %s psmp" % (
            sys.executable, script, exe_dir)
        self.assertEqual(cmd, expected)

        # CP2K 2024.1: arch as argument, driver adds '-n <ranks>' itself
        write_file(script, 'parser.add_argument("arch")\nparser.add_argument("--mpiexec", default="mpiexec")\n')
        cmd = cp2k.EB_CP2K.py_regtest_cmd(eb, script, 4)
        expected = "%s %s --mpiranks 4 --maxtasks 4 --ompthreads 2 --mpiexec mpirun Linux-foss psmp" % (
            sys.executable, script)
        self.assertEqual(cmd, expected)

        # without OpenMP threads configured
        eb.cfg['omp_num_threads'] = None
        self.assertNotIn('--ompthreads', cp2k.EB_CP2K.py_regtest_cmd(eb, script, 4))

    def test_cp2k_helpers(self):
        """Test helper functions of the CP2K easyblock"""
        configopts = ' '.join([
            '-DCP2K_USE_MPI=OFF',
            '-DCP2K_USE_MPI_F08:BOOL=ON',
            "-DCP2K_USE_LIBXC='YES'",
            '-D CP2K_USE_ELPA=on',
            '-DCP2K_USE_SPGLIB=ON -DCP2K_USE_SPGLIB=0',
            '-DCMAKE_BUILD_TYPE="Debug"',
            '-DCP2K_USE_VORI=',
        ])
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_MPI'), 'OFF')
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_MPI_F08'), 'ON')
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_LIBXC'), 'YES')
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_ELPA'), 'on')
        # the last value wins, as for CMake
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_SPGLIB'), '0')
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CMAKE_BUILD_TYPE'), 'Debug')
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_VORI'), '')
        self.assertEqual(cp2k.get_cmake_option(configopts, 'CP2K_USE_PLUMED'), None)
        # only the complete option name matches
        self.assertEqual(cp2k.get_cmake_option('-DXCP2K_USE_PLUMED=ON -DCP2K_USE_PLUMEDX=ON', 'CP2K_USE_PLUMED'), None)

        for value in ['ON', 'on', 'Yes', 'TRUE', 'y', '1', '2', 'FlexiBLAS']:
            self.assertTrue(cp2k.is_cmake_true(value), value)
        for value in ['OFF', 'off', 'No', 'FALSE', 'n', '0', '', 'IGNORE', 'NOTFOUND', 'ELPA-NOTFOUND']:
            self.assertFalse(cp2k.is_cmake_true(value), value)

        # output in the format printed by tests/do_regtest.py
        errors = '\n'.join([
            'x' * 100,
            '/tmp/regtesting/TEST-2026-11-03_10-00-00/QS/regtest-kp-hfx-ri/Si_kp_hfx.inp.out',
            '(last lines of the output)',
            '',
            'Runtime failure with code 1.',
        ])
        output = '\n'.join([
            "Launched 2 test directories and 2 worker...",
            '',
            '------------------------------- Errors ---------------------------------',
            errors,
            '',
            '------------------------------- Timings --------------------------------',
            'Plot: name="timings", title="Timing Distribution", ylabel="time [s]"',
            '',
            '------------------------------- Summary --------------------------------',
            'Number of FAILED  tests 1',
            'Number of WRONG   tests 1',
            'Number of CORRECT tests 2',
            'Total number of   tests 5',
            '',
            'Summary: correct: 2 / 5; wrong: 1; failed: 1; n/a: 1; 3min',
            'Status: FAILED',
            '',
            '*************************** Testing ended ******************************',
        ])
        expected = {'failed': 1, 'wrong': 1, 'correct': 2, 'total': 5, 'na': 1, 'status': 'FAILED'}
        self.assertEqual(cp2k.parse_regtest_summary(output), expected)
        self.assertEqual(cp2k.get_regtest_errors(output), errors)

        output = output.replace('Number of FAILED  tests 1', 'Number of FAILED  tests 0')
        output = output.replace('Number of WRONG   tests 1', 'Number of WRONG   tests 0')
        output = output.replace('Total number of   tests 5', 'Total number of   tests 2')
        output = output.replace('Status: FAILED', 'Status: OK')
        expected = {'failed': 0, 'wrong': 0, 'correct': 2, 'total': 2, 'na': 0, 'status': 'OK'}
        self.assertEqual(cp2k.parse_regtest_summary(output), expected)

        # no summary if the driver stopped before running tests
        output = "\nNo test directories selected, check --restrictdir filter.\n"
        self.assertEqual(cp2k.parse_regtest_summary(output), None)
        self.assertEqual(cp2k.get_regtest_errors(output), '')

    def test_cp2k_cmake(self):
        """Test the support of the CP2K easyblock for building with CMake"""

        def get_cp2k_instance(version, toolchain, deps):
            # a new path for each easyconfig, since process_easyconfig caches parsed easyconfigs by path
            ec_path = os.path.join(tempfile.mkdtemp(dir=self.tmpdir), 'CP2K-%s.eb' % version)
            write_file(ec_path, '\n'.join([
                "easyblock = 'EB_CP2K'",
                "name = 'CP2K'",
                "version = '%s'" % version,
                "homepage = 'https://www.cp2k.org'",
                "description = 'CP2K'",
                "toolchain = %s" % toolchain,
                "builddependencies = [('Python', '3.14.2')]",
                "dependencies = %s" % deps,
                "moduleclass = 'chem'",
            ]))
            # no easyconfig for the toolchain is available to determine its hierarchy, which prints warnings
            self.mock_stderr(True)
            self.mock_stdout(True)
            inst = get_easyblock_instance(process_easyconfig(ec_path)[0])
            self.mock_stderr(False)
            self.mock_stdout(False)
            # sets the parallelism (cfg.parallel), as done before running any step
            inst.post_init()
            return inst

        self.assertFalse(get_cp2k_instance('2025.2', 'SYSTEM', []).cmake_build)

        deps = [('DBCSR', '2.10.0'), ('ELPA', '2026.02.002'), ('Libint', '2.13.1', '-lmax-6-cp2k'),
                ('libvori', '220621'), ('libxc', '7.0.0'), ('PLUMED', '2.10.0'), ('spglib', '2.7.0')]
        cp2k_inst = get_cp2k_instance('2026.2', "{'name': 'foss', 'version': '2026.1'}", deps)
        self.assertTrue(cp2k_inst.cmake_build)
        self.assertEqual(cp2k_inst.build_type, 'Generic')

        features = cp2k_inst.cmake_features()
        expected_options = ['CP2K_USE_ELPA', 'CP2K_USE_FFTW3', 'CP2K_USE_LIBINT2', 'CP2K_USE_LIBXC', 'CP2K_USE_MPI',
                            'CP2K_USE_MPI_F08', 'CP2K_USE_PLUMED', 'CP2K_USE_SPGLIB', 'CP2K_USE_VORI']
        self.assertEqual(features['options'], dict.fromkeys(expected_options, 'ON'))
        self.assertEqual(features['ext'], 'psmp')
        expected_flags = ['omp', 'parallel', 'scalapack', 'mpi_f08', 'fftw3', 'elpa', 'libint', 'libxc', 'plumed2',
                          'spglib', 'libvori', 'libbqb']
        self.assertEqual(features['flags'], expected_flags)

        with mock.patch.object(CMakeMake, 'configure_step') as cmake_configure_step:
            cp2k_inst.cfg['configopts'] = '-DCP2K_USE_SPGLIB=OFF'
            cp2k_inst.configure_step()
        cmake_configure_step.assert_called_once()
        configopts = cp2k_inst.cfg['configopts'].split()
        self.assertIn('-DCP2K_BLAS_VENDOR=FlexiBLAS', configopts)
        for option in expected_options:
            if option != 'CP2K_USE_SPGLIB':
                self.assertIn('-D%s=ON' % option, configopts)
        # options in configopts take precedence
        self.assertIn('-DCP2K_USE_SPGLIB=OFF', configopts)
        self.assertNotIn('-DCP2K_USE_SPGLIB=ON', configopts)
        self.assertNotIn('spglib', cp2k_inst.cmake_features()['flags'])

        cp2k_inst.cfg['configopts'] = '-D CP2K_USE_MPI_F08:BOOL=NO -DCMAKE_BUILD_TYPE=Debug'
        features = cp2k_inst.cmake_features()
        self.assertEqual(features['ext'], 'pdbg')
        self.assertNotIn('mpi_f08', features['flags'])
        cp2k_inst.cfg['configopts'] = ''

        cp2k_inst.cfg['plumed'] = False
        features = cp2k_inst.cmake_features()
        self.assertNotIn('CP2K_USE_PLUMED', features['options'])
        self.assertNotIn('plumed2', features['flags'])

        # oversubscription of the MPI ranks started by the test driver, for Open MPI 4.x and 5.x
        regtest_cmd = 'python3 -B /src/tests/do_regtest.py --maxtasks %s --smoketest /inst/bin psmp'
        regtest_cmd = regtest_cmd % cp2k_inst.cfg.parallel
        os.environ.pop('EBVERSIONOPENMPI', None)
        prte_env = 'export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe && '
        self.assertEqual(cp2k_inst.regtest_cmd('/src', '/inst/bin', 'psmp', ['--smoketest']), prte_env + regtest_cmd)
        os.environ['EBVERSIONOPENMPI'] = '4.1.6'
        ompi_env = 'export OMPI_MCA_rmaps_base_oversubscribe=1 && '
        self.assertEqual(cp2k_inst.regtest_cmd('/src', '/inst/bin', 'psmp', ['--smoketest']), ompi_env + regtest_cmd)
        os.environ['EBVERSIONOPENMPI'] = '5.0.8'
        cp2k_inst.cfg['maxtasks'] = 6
        cp2k_inst.cfg['omp_num_threads'] = 3
        regtest_cmd = 'python3 -B /src/tests/do_regtest.py --maxtasks 6 --ompthreads 3 /inst/bin psmp'
        self.assertEqual(cp2k_inst.regtest_cmd('/src', '/inst/bin', 'psmp', []), prte_env + regtest_cmd)

        with mock.patch.object(EasyBlock, 'sanity_check_step') as sanity_check_step:
            cp2k_inst.sanity_check_step()
        kwargs = sanity_check_step.call_args.kwargs
        for path in ['bin/cp2k.psmp', 'bin/cp2k.popt', 'lib/libcp2k.%s' % get_shared_lib_ext(),
                     'lib/cmake/cp2k/cp2kConfig.cmake', 'include/cp2k/libcp2k.h', 'share/cp2k/tests/do_regtest.py']:
            self.assertIn(path, kwargs['custom_paths']['files'])
        self.assertEqual(kwargs['custom_paths']['dirs'], ['share/cp2k/data', 'share/cp2k/src/grid/sample_tasks'])
        commands = kwargs['custom_commands']
        flag_cmd = "cp2k.psmp --version | grep 'cp2kflags:' | grep -w %s"
        for flag in expected_flags:
            if flag == 'plumed2':
                self.assertNotIn(flag_cmd % flag, commands)
            else:
                self.assertIn(flag_cmd % flag, commands)
        share_dir = os.path.join(cp2k_inst.installdir, 'share', 'cp2k')
        regtest = commands[-1]
        self.assertTrue(regtest.startswith('cp -R %s/tests %s/src . && chmod -R u+w tests src && ' %
                                           (share_dir, share_dir)), regtest)
        self.assertTrue(regtest.endswith('./tests/do_regtest.py --maxtasks 6 --ompthreads 3 --smoketest '
                                         '--cp2kdatadir %s/data %s/bin psmp' % (share_dir, cp2k_inst.installdir)),
                        regtest)
        self.assertEqual(kwargs['extra_modules'], ['Python/3.14.2-foss-2026.1'])

        cp2k_inst.cfg['sanity_regtest'] = False
        cp2k_inst.cfg['configopts'] = '-DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_LIBDIR=lib64'
        with mock.patch.object(EasyBlock, 'sanity_check_step') as sanity_check_step:
            cp2k_inst.sanity_check_step()
        kwargs = sanity_check_step.call_args.kwargs
        self.assertIn('lib64/libcp2k.a', kwargs['custom_paths']['files'])
        self.assertFalse(any('do_regtest.py' in cmd for cmd in kwargs['custom_commands']))
        self.assertEqual(kwargs['extra_modules'], None)
        cp2k_inst.cfg['configopts'] = ''

        # test step: regression tests with the binaries in the build directory, judged by the driver's summary
        cp2k_inst.cfg['start_dir'] = '/src/cp2k-2026.2'
        cp2k_inst.cfg['pretestopts'] = 'ulimit -s unlimited && '
        cp2k_inst.builddir = '/build'
        cp2k_inst.separate_build_dir = '/build/easybuild_obj'
        # the easyblock class is loaded as easybuild.easyblocks.cp2k, which is not the module imported above
        cp2k_module = sys.modules[type(cp2k_inst).__module__]

        def run_test_step(output, exit_code):
            res = RunShellCmdResult(cmd='', exit_code=exit_code, output=output, stderr=None, work_dir='/build',
                                    out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)
            with mock.patch.object(cp2k_module, 'run_shell_cmd', return_value=res) as run_shell_cmd:
                self.mock_stderr(True)
                try:
                    cp2k_inst.test_step()
                finally:
                    stderr = self.get_stderr()
                    self.mock_stderr(False)
            return run_shell_cmd.call_args.args[0], stderr

        summary = '\n'.join(['Number of FAILED  tests %d', 'Number of WRONG   tests %d',
                             'Number of CORRECT tests %d', 'Total number of   tests %d', 'Status: %s'])
        cmd, stderr = run_test_step(summary % (0, 0, 2, 2, 'OK'), 0)
        self.assertEqual(cmd, 'ulimit -s unlimited &&  ' + prte_env + 'python3 -B /src/cp2k-2026.2/tests/do_regtest.py '
                         '--maxtasks 6 --ompthreads 3 --workbasedir /build/regtesting '
                         '--cp2kdatadir /src/cp2k-2026.2/data /build/easybuild_obj/bin psmp')
        self.assertEqual(stderr, '')

        failed = summary % (1, 1, 2, 5, 'FAILED')
        error_regex = "CP2K regression tests: 2 of 5 correct, 1 wrong, 1 failed, 1 without result"
        self.assertRaisesRegex(EasyBuildError, error_regex, run_test_step, failed, 1)
        self.assertRaisesRegex(EasyBuildError, r"No summary .* \(exit code 1\)", run_test_step, '', 1)

        cp2k_inst.cfg['ignore_regtest_fails'] = True
        stderr = run_test_step(failed, 1)[1]
        self.assertIn(error_regex + '; ignoring this, as requested', stderr)
        cp2k_inst.cfg['ignore_regtest_fails'] = False

        # failures are reported as test failures, which --ignore-test-failure turns into a warning
        orig_ignore_test_failure = config.update_build_option('ignore_test_failure', True)
        try:
            stderr = run_test_step(failed, 1)[1]
        finally:
            config.update_build_option('ignore_test_failure', orig_ignore_test_failure)
        self.assertIn('Test failure ignored: ' + error_regex, stderr)

        cp2k_inst.cfg['runtest'] = False
        with mock.patch.object(cp2k_module, 'run_shell_cmd') as run_shell_cmd:
            cp2k_inst.test_step()
        run_shell_cmd.assert_not_called()

        # without MPI
        cp2k_inst = get_cp2k_instance('2026.2', 'SYSTEM', [('libxc', '7.0.0')])
        features = cp2k_inst.cmake_features()
        self.assertEqual(features['options'], {'CP2K_USE_LIBXC': 'ON'})
        self.assertEqual((features['ext'], features['flags']), ('ssmp', ['omp', 'libxc']))
        cp2k_inst.cfg['configopts'] = '-DCP2K_USE_MPI_F08=ON'
        self.assertNotIn('mpi_f08', cp2k_inst.cmake_features()['flags'])
        cp2k_inst.cfg['configopts'] = '-DCP2K_USE_ELPA=ON'
        self.assertRaisesRegex(EasyBuildError, "only supports CP2K_USE_ELPA with MPI", cp2k_inst.cmake_features)
        cp2k_inst.cfg['configopts'] = ''
        cp2k_inst.cfg['plumed'] = True
        self.assertRaisesRegex(EasyBuildError, "PLUMED is not a dependency", cp2k_inst.cmake_features)

        cp2k_inst = get_cp2k_instance('2026.2', 'SYSTEM', [('CUDA', '13.0.1')])
        self.assertRaisesRegex(EasyBuildError, "GPU support is not enabled", cp2k_inst.cmake_features)
        cp2k_inst.cfg['configopts'] = '-DCP2K_USE_ACCEL=CUDA -DCP2K_WITH_GPU=H100'
        self.assertEqual(cp2k_inst.cmake_features()['ext'], 'ssmp')

    def test_pytorch_test_log_parsing(self):
        """Verify parsing of XML files produced by PyTorch tests."""
        TestState = pytorch.TestState

        test_log_dir = Path(__file__).parent.parent / 'pytorch_test_logs'

        results = pytorch.get_test_results(test_log_dir / 'test-reports')
        results2 = pytorch.get_test_results(test_log_dir)
        self.assertEqual(results.keys(), results2.keys())
        for name, suite in results.items():
            self.assertEqual((name, suite.summary), (name, results2[name].summary))
        del results2

        self.assertEqual(len(results), 16)

        # 2 small test suites used as a smoke test using a most features
        self.assertIn('backends/xeon/test_launch', results)
        suite = results['backends/xeon/test_launch']
        self.assertEqual((suite.errors, suite.failures, suite.num_tests, suite.skipped), (1, 2, 8, 3))
        # Failure in one file, success in the other --> Success
        self.assertEqual(suite['TestTorchrun.test_cpu_info'].state, TestState.SUCCESS)
        # New in 2nd file
        self.assertEqual(suite['TestTorchrun.test_multi_threads'].state, TestState.SUCCESS)
        self.assertEqual(suite['TestTorchrun.test_reshape_cpu_float64'].state, TestState.FAILURE)
        self.assertEqual(suite['TestTorchrun.test_foo'].state, TestState.SKIPPED)
        self.assertEqual(suite['TestTorchrun.test_bar'].state, TestState.ERROR)
        self.assertEqual(suite.get_errored_tests(), ['TestTorchrun.test_bar'])
        self.assertEqual(suite.get_failed_tests(), ['TestTorchrun.test_reshape_cpu_float64', 'TestTorchrun.test_baz'])
        self.assertIn('test_autoload', results)
        suite = results['test_autoload']
        self.assertEqual((suite.errors, suite.failures, suite.num_tests, suite.skipped), (0, 0, 2, 1))
        self.assertEqual(suite['TestBackendAutoload.test_autoload'].state, TestState.SUCCESS)
        self.assertEqual(suite['TestBackendAutoload.test_unload'].state, TestState.SKIPPED)

        # Verify summaries which should be enough to catch most issues
        report = '\n'.join(sorted(f'{suite.name}: {suite.summary}' for suite in results.values()))
        self.assertEqual(report, textwrap.dedent("""
            backends/xeon/test_launch: 2 failed, 2 passed, 3 skipped, 1 errors
            dist-gloo-init-env/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist-gloo-init-file/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist-nccl-init-env/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist-nccl-init-file/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist/foo/bar: 0 failed, 4 passed, 0 skipped, 0 errors
            distributed/_composable/test_composability/test_pp_composability: 0 failed, 2 passed, 0 skipped, 0 errors
            distributed/tensor/test_dtensor_ops: 0 failed, 2 passed, 2 skipped, 0 errors
            dynamo/test_dynamic_shapes: 3 failed, 14 passed, 0 skipped, 0 errors
            dynamo/test_misc: 3 failed, 11 passed, 0 skipped, 0 errors
            inductor/test_aot_inductor_arrayref: 2 failed, 0 passed, 0 skipped, 0 errors
            inductor/test_cudagraph_trees: 1 failed, 0 passed, 0 skipped, 0 errors
            jit/test_builtins: 0 failed, 1 passed, 0 skipped, 0 errors
            test_autoload: 0 failed, 1 passed, 1 skipped, 0 errors
            test_nestedtensor: 3 failed, 2 passed, 3 skipped, 1 errors
            test_quantization: 0 failed, 12 passed, 5 skipped, 0 errors
        """).strip())
        tests = '\n'.join(sorted(f'{test.name}: {test.state.value}'
                                 for suite in results.values()
                                 for test in suite.get_tests()))
        self.assertEqual(tests, textwrap.dedent("""
            AOTInductorTestABICompatibleCpuWithStackAllocation.test_fail_and_skip: failure
            AOTInductorTestABICompatibleCpuWithStackAllocation.test_skip_and_fail: failure
            ComposabilityTest.test_pass_on_rerun_different_classname: success
            ComposabilityTest.test_pp_and_dcp: success
            CudaGraphTreeTests.test_workspace_allocation_error: failure
            DistQuantizationTests.test_all_gather_fp16: success
            DistQuantizationTests.test_all_gather_fp16: success
            DistQuantizationTests.test_all_gather_fp16: success
            DistQuantizationTests.test_all_gather_fp16: success
            DynamicShapesCtxManagerTests.test_autograd_profiler_dynamic_shapes: success
            DynamicShapesCtxManagerTests.test_generic_context_manager_with_graph_break_dynamic_shapes: success
            DynamicShapesCtxManagerTests.test_generic_ctx_manager_with_graph_break_dynamic_shapes: success
            DynamicShapesMiscTests.test_outside_linear_module_free_dynamic_shapes: failure
            DynamicShapesMiscTests.test_packaging_version_parse_dynamic_shapes: success
            DynamicShapesMiscTests.test_pair_dynamic_shapes: success
            DynamicShapesMiscTests.test_param_shape_binops_dynamic_shapes: success
            DynamicShapesMiscTests.test_parameter_free_dynamic_shapes: failure
            DynamicShapesMiscTests.test_patched_builtin_functions_dynamic_shapes: success
            DynamicShapesMiscTests.test_proxy_frozen_dataclass_dynamic_shapes: success
            DynamicShapesMiscTests.test_pt2_compliant_ops_are_allowed_dynamic_shapes: success
            DynamicShapesMiscTests.test_pt2_compliant_overload_dynamic_shapes: success
            DynamicShapesMiscTests.test_pure_python_accumulate_dynamic_shapes: success
            DynamicShapesMiscTests.test_py_guards_mark_dynamic_dynamic_shapes: success
            DynamicShapesMiscTests.test_python_slice_dynamic_shapes: success
            DynamicShapesMiscTests.test_pytree_tree_flatten_unflatten_dynamic_shapes: success
            DynamicShapesMiscTests.test_pytree_tree_leaves_dynamic_shapes: failure
            MiscTests.test_fail_then_skip2: failure
            MiscTests.test_fail_then_skip: failure
            MiscTests.test_fail_then_xfail2: success
            MiscTests.test_fail_then_xfail: success
            MiscTests.test_packaging_version_parse: success
            MiscTests.test_pair: success
            MiscTests.test_param_shape_binops: success
            MiscTests.test_parameter_free: failure
            MiscTests.test_pytree_tree_map: success
            MiscTests.test_shape_env_no_recording: success
            MiscTests.test_shape_env_recorded_function_fallback: success
            MiscTests.test_yield_from_in_a_loop: success
            TestBackendAutoload.test_autoload: success
            TestBackendAutoload.test_unload: skipped
            TestBuiltins.test_name: success
            TestCustomFunction.test_autograd_function_with_matmul_folding_at_output: success
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_float16: success
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_float32: success
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_float64: skipped
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_int8: skipped
            TestDynamicQuantizedOps.test_qrnncell: success
            TestFakeQuantizeOps.test_backward_per_channel: skipped
            TestFakeQuantizeOps.test_backward_per_channel_cachemask_cpu: success
            TestFakeQuantizeOps.test_backward_per_channel_cachemask_cuda: success
            TestName.test_bar: success
            TestNestedTensor.test_bmm_cuda_gpu_float16: failure
            TestNestedTensor.test_bmm_cuda_gpu_float32: failure
            TestNestedTensor.test_bmm_cuda_gpu_float64: error
            TestNestedTensor.test_cat: success
            TestNestedTensor.test_copy_: success
            TestNestedTensor.test_reshape_cpu_float16: skipped
            TestNestedTensor.test_reshape_cpu_float32: skipped
            TestNestedTensor.test_reshape_cpu_float64: failure
            TestNestedTensorSubclassCPU.test_linear_backward_memory_usage_cpu_float32: skipped
            TestNumericDebugger.test_quantize_pt2e_preserve_handle: success
            TestNumericDebugger.test_re_export_preserve_handle: success
            TestPadding.test_reflection_pad1d: success
            TestQuantizedConv.test_conv_reorder_issue_onednn: success
            TestQuantizedConv.test_conv_transpose_reorder_issue_onednn: success
            TestQuantizedFunctionalOps.test_relu_api: success
            TestQuantizedLinear.test_qlinear_cudnn: skipped
            TestQuantizedLinear.test_qlinear_gelu_pt2e: success
            TestQuantizedOps.test_adaptive_avg_pool2d_nhwc: success
            TestQuantizedOps.test_adaptive_avg_pool: skipped
            TestQuantizedOps.test_qadd_relu_cudnn: skipped
            TestQuantizedOps.test_qadd_relu_cudnn_nhwc: skipped
            TestQuantizedOps.test_qadd_relu_different_qparams: success
            TestTorchrun.test_bar: error
            TestTorchrun.test_baz: failure
            TestTorchrun.test_cpu_info: success
            TestTorchrun.test_foo2: skipped
            TestTorchrun.test_foo3: skipped
            TestTorchrun.test_foo: skipped
            TestTorchrun.test_multi_threads: success
            TestTorchrun.test_reshape_cpu_float64: failure
            TestTracer.test_jit_save: success
            test_2.test_func3: success
            test_foo.TestBar.test_func2: success
            test_foo.TestName.test_func1: success
        """).strip())

        #  Some error cases
        error_log_dir = test_log_dir / 'faulty-reports'

        self.assertRaisesRegex(ValueError, "<testsuites> or <testsuite>",
                               pytorch.get_test_results, error_log_dir / 'root')
        self.assertRaisesRegex(ValueError, "Failed to parse",
                               pytorch.get_test_results, error_log_dir / 'invalid_xml')
        self.assertRaisesRegex(ValueError, "multiple reported files",
                               pytorch.get_test_results, error_log_dir / 'multi_file')
        self.assertRaisesRegex(ValueError, "Path from folder and filename should be equal",
                               pytorch.get_test_results, error_log_dir / 'different_file_name')
        self.assertRaisesRegex(ValueError, "Unexpected file attribute",
                               pytorch.get_test_results, error_log_dir / 'file_attribute')
        self.assertRaisesRegex(ValueError, "Invalid state",
                               pytorch.get_test_results, error_log_dir / 'skip_and_failed')
        self.assertRaisesRegex(ValueError, "no test",
                               pytorch.get_test_results, error_log_dir / 'no_tests')
        self.assertRaisesRegex(ValueError, "Invalid test count",
                               pytorch.get_test_results, error_log_dir / 'consistency')
        self.assertRaisesRegex(ValueError, "Duplicate test",
                               pytorch.get_test_results, error_log_dir / 'duplicate')

    def test_libint_compiler_configure_skips_cmake_flags(self):
        """Autoconf ./configure must not receive CMake -D options from configopts."""
        cmd = libint.libint_compiler_configure_cmd('', '--enable-eri=1 --with-max-am=6')
        self.assertEqual(cmd, './configure --enable-eri=1 --with-max-am=6')
        self.assertNotIn('-DLIBINT2_ENABLE_FORTRAN=ON', cmd)

        cmd = libint.libint_compiler_configure_cmd('export FOO=1 &&', '--disable-unrolling')
        self.assertEqual(cmd, 'export FOO=1 && ./configure --disable-unrolling')

        self.assertEqual(libint.libint_fortran_cmake_flag('2.11.2'), '-DENABLE_FORTRAN=ON')
        self.assertEqual(libint.libint_fortran_cmake_flag('2.12.0'), '-DLIBINT2_ENABLE_FORTRAN=ON')

    def test_cp2k_cmake_tblite(self):
        """A declared tblite dependency enables its feature without overriding configopts."""
        from types import SimpleNamespace

        class Config(dict):
            def dependencies(self):
                return [{'name': 'tblite'}]

        cfg = Config(configopts='', plumed=False)
        instance = SimpleNamespace(cfg=cfg, build_type='Generic',
                                   toolchain=SimpleNamespace(mpi_family=lambda: None))
        features = cp2k.EB_CP2K.cmake_features(instance)
        self.assertEqual(features['options'], {'CP2K_USE_TBLITE': 'ON'})
        self.assertTrue(features['enabled']['CP2K_USE_TBLITE'])
        self.assertIn('tblite', features['flags'])

        cfg['configopts'] = '-DCP2K_USE_TBLITE=OFF'
        features = cp2k.EB_CP2K.cmake_features(instance)
        self.assertFalse(features['enabled']['CP2K_USE_TBLITE'])
        self.assertNotIn('tblite', features['flags'])

    def test_cp2k_cmake_tblite_dispersion_flags(self):
        """The sanity check requires every cp2kflag that CP2K_USE_TBLITE adds and tests/TEST_DIRS selects on."""

        def tblite_sanity_commands(version, configopts=''):
            ec_path = os.path.join(tempfile.mkdtemp(dir=self.tmpdir), 'CP2K-%s.eb' % version)
            write_file(ec_path, '\n'.join([
                "easyblock = 'EB_CP2K'",
                "name = 'CP2K'",
                "version = '%s'" % version,
                "homepage = 'https://www.cp2k.org'",
                "description = 'CP2K'",
                "toolchain = SYSTEM",
                "builddependencies = [('Python', '3.14.2')]",
                "dependencies = [('tblite', '0.6.0', '-cp2k')]",
                "configopts = '%s'" % configopts,
                "moduleclass = 'chem'",
            ]))
            self.mock_stderr(True)
            self.mock_stdout(True)
            inst = get_easyblock_instance(process_easyconfig(ec_path)[0])
            self.mock_stderr(False)
            self.mock_stdout(False)
            inst.post_init()
            with mock.patch.object(EasyBlock, 'sanity_check_step') as sanity_check_step:
                inst.sanity_check_step()
            return sanity_check_step.call_args.kwargs['custom_commands']

        flag_cmd = "cp2k.ssmp --version | grep 'cp2kflags:' | grep -w %s"

        # CP2K 2026.2: __TBLITE, __DFTD4 and __S_DFTD3
        commands = tblite_sanity_commands('2026.2')
        for flag in ['tblite', 'mctc-lib', 'libdftd4', 's_dftd3']:
            self.assertIn(flag_cmd % flag, commands)

        # CP2K 2026.1 defines __TBLITE and __DFTD4, but not __S_DFTD3
        commands = tblite_sanity_commands('2026.1')
        for flag in ['tblite', 'mctc-lib', 'libdftd4']:
            self.assertIn(flag_cmd % flag, commands)
        self.assertNotIn(flag_cmd % 's_dftd3', commands)

        commands = tblite_sanity_commands('2026.2', configopts='-DCP2K_USE_TBLITE=OFF')
        for flag in ['tblite', 'mctc-lib', 'libdftd4', 's_dftd3']:
            self.assertNotIn(flag_cmd % flag, commands)

    def test_cp2k_makefile_hdf5_sirius(self):
        """The Makefile build of CP2K < 2026.1 enables HDF5 and SIRIUS when they are dependencies."""

        def options():
            return {'DFLAGS': ' -D__parallel', 'FCFLAGSOPT': '', 'LIBS': ''}

        res = cp2k.makefile_hdf5_sirius_options(options(), '/hdf5', '/sirius')
        self.assertEqual(res['DFLAGS'].split(), ['-D__parallel', '-D__HDF5', '-D__SIRIUS'])
        self.assertEqual(res['FCFLAGSOPT'].split(), ['-I/hdf5/include', '-I/sirius/include/sirius'])
        self.assertEqual(res['LIBS'].split(), ['-lhdf5_fortran', '-lhdf5', '-lsirius'])

        res = cp2k.makefile_hdf5_sirius_options(options(), '/hdf5', None)
        self.assertEqual(res['DFLAGS'].split(), ['-D__parallel', '-D__HDF5'])
        self.assertEqual(res['LIBS'].split(), ['-lhdf5_fortran', '-lhdf5'])

        self.assertEqual(cp2k.makefile_hdf5_sirius_options(options(), None, None), options())

        self.assertRaisesRegex(EasyBuildError, "SIRIUS support in CP2K requires HDF5",
                               cp2k.makefile_hdf5_sirius_options, options(), None, '/sirius')


def suite(loader):
    """Return all easyblock-specific tests."""
    return loader.loadTestsFromTestCase(EasyBlockSpecificTest)


if __name__ == '__main__':
    res = TextTestRunner(verbosity=1).run(suite(TestLoader()))
    sys.exit(len(res.failures))
