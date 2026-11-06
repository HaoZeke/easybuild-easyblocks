# -*- coding: utf-8 -*-
##
# Copyright 2009-2026 Ghent University
#
# This file is part of EasyBuild,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://vscentrum.be/nl/en),
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
EasyBuild support for installing Gurobi, implemented as an easyblock

@author: Bob Dröge (University of Groningen)
@author: Samuel Moors (Vrije Universiteit Brussel)
@author: Sven Hansen (RWTH Aachen University)
"""
import os
import shlex

from easybuild.easyblocks.generic.tarball import Tarball
from easybuild.framework.easyconfig import CUSTOM
from easybuild.tools import LooseVersion
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.config import build_option
from easybuild.tools.filetools import copy_file, which
from easybuild.tools.modules import get_software_root
from easybuild.tools.run import run_shell_cmd


class EB_Gurobi(Tarball):
    """Support for installing linux64 version of Gurobi."""

    @staticmethod
    def extra_options(extra_vars=None):
        """Define extra options for Gurobi"""
        extra = {
            'copy_license_file': [True, "Copy license_file to installdir", CUSTOM],
        }
        return Tarball.extra_options(extra_vars=extra)

    def __init__(self, *args, **kwargs):
        """Easyblock constructor, define custom class variables specific to Gurobi."""
        super().__init__(*args, **kwargs)

        # make sure license file is available
        self.orig_license_file = self.cfg['license_file']
        if self.orig_license_file is None:
            self.orig_license_file = os.getenv('EB_GUROBI_LICENSE_FILE', None)

        if self.cfg['copy_license_file']:
            self.license_file = os.path.join(self.installdir, 'gurobi.lic')
        else:
            self.license_file = self.orig_license_file

    def install_step(self):
        """Install Gurobi and license file."""
        super().install_step()

        if self.cfg['copy_license_file']:
            if self.orig_license_file is None or not os.path.exists(self.orig_license_file):
                raise EasyBuildError("No existing license file specified: %s", self.orig_license_file)

            copy_file(self.orig_license_file, self.license_file)

        if get_software_root('Python') and LooseVersion(self.version) < LooseVersion('11'):
            run_shell_cmd("python setup.py install --prefix=%s" % self.installdir)

        # Build C++ ABI with the chosen compiler
        libbuilddir = os.path.join(self.installdir, "src", "build")
        run_shell_cmd("make", work_dir=libbuilddir)
        copy_file(os.path.join(libbuilddir, "libgurobi_c++.a"), os.path.join(self.installdir, "lib"))
        run_shell_cmd("make clean", work_dir=libbuilddir)

    def post_processing_step(self):
        """Keep shipped binaries runnable when the module omits LD_LIBRARY_PATH."""
        super().post_processing_step()

        if 'LD_LIBRARY_PATH' not in (build_option('filter_env_vars') or []):
            return
        if which('patchelf') is None:
            raise EasyBuildError("patchelf is required when LD_LIBRARY_PATH is filtered")

        additions = ['$ORIGIN/../lib', os.path.join(self.installdir, 'lib')]
        for subdir in ('bin', 'lib', 'lib64', 'matlab'):
            root = os.path.join(self.installdir, subdir)
            for directory, _, filenames in os.walk(root):
                for filename in filenames:
                    path = os.path.join(directory, filename)
                    if not os.path.islink(path) and os.path.isfile(path):
                        self._append_permanent_lib_rpath(path, additions)

    def _append_permanent_lib_rpath(self, path, additions):
        """Record the install library directory without discarding vendor paths."""
        with open(path, 'rb') as stream:
            if stream.read(4) != b'\x7fELF':
                return
        quoted_path = shlex.quote(path)
        info = run_shell_cmd("file -b %s" % quoted_path, hidden=True).output.lower()
        if 'dynamically linked' not in info:
            return

        current = run_shell_cmd("patchelf --print-rpath %s" % quoted_path, hidden=True).output.strip()
        original = [entry for entry in current.split(':') if entry]
        entries = list(original)
        for addition in additions:
            if addition not in entries:
                entries.append(addition)
        dynamic = run_shell_cmd("readelf -d %s" % quoted_path, hidden=True).output
        if entries == original and '(RPATH)' in dynamic:
            return

        run_shell_cmd("patchelf --force-rpath --set-rpath %s %s" %
                      (shlex.quote(':'.join(entries)), quoted_path), hidden=True)
        recorded = run_shell_cmd("patchelf --print-rpath %s" % quoted_path, hidden=True).output.strip()
        if recorded.split(':') != entries:
            raise EasyBuildError("RPATH entries were not preserved in %s: %s", path, recorded)
        dynamic = run_shell_cmd("readelf -d %s" % quoted_path, hidden=True).output
        if '(RPATH)' not in dynamic:
            raise EasyBuildError("DT_RPATH was not recorded for %s", path)

    def sanity_check_step(self):
        """Custom sanity check for Gurobi."""
        files = ['grbprobe', 'grbtune', 'gurobi_cl']
        if LooseVersion(self.version) < LooseVersion('13'):
            files.append('gurobi.sh')
        custom_paths = {
            'files': ['bin/%s' % f for f in files],
            'dirs': ['matlab'],
        }

        custom_commands = [
            "gurobi_cl --help",
            'test -f $GRB_LICENSE_FILE',
        ]

        if get_software_root('Python'):
            custom_commands.append("python -c 'import gurobipy'")

        super().sanity_check_step(custom_commands=custom_commands, custom_paths=custom_paths)

    def make_module_extra(self):
        """Custom extra module file entries for Gurobi."""
        txt = super().make_module_extra()
        txt += self.module_generator.set_environment('GUROBI_HOME', self.installdir)
        txt += self.module_generator.set_environment('GRB_LICENSE_FILE', self.license_file)
        txt += self.module_generator.prepend_paths('MATLABPATH', 'matlab')

        return txt
