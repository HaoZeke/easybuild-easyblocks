#!/usr/bin/python
##
# Copyright 2012-2026 Ghent University
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
"""Check interpreter version parsing with diagnostic launcher output."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import easybuild.easyblocks.generic.pythonpackage as pythonpackage


class PythonVersionWrapperOutput(unittest.TestCase):
    def test_launcher_diagnostics_leave_the_interpreter_version(self):
        output = "launcher: selecting target interpreter\nlauncher: library paths configured\n3.14.2\n"
        with patch.object(pythonpackage, 'run_shell_cmd', return_value=SimpleNamespace(output=output)) as run:
            self.assertEqual(pythonpackage.det_python_version('/managed/bin/python'), '3.14.2')
        run.assert_called_once()
        self.assertTrue(run.call_args.args[0].startswith('/managed/bin/python -c '))
        self.assertTrue(run.call_args.kwargs['hidden'])

    def test_interpreter_version_without_diagnostics(self):
        with patch.object(pythonpackage, 'run_shell_cmd', return_value=SimpleNamespace(output='3.13.7\n')):
            self.assertEqual(pythonpackage.det_python_version('python3'), '3.13.7')
