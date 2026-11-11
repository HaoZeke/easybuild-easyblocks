##
# Copyright 2009-2026 Ghent University
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
EasyBuild support for building and installing CP2K, implemented as an easyblock

@author: Stijn De Weirdt (Ghent University)
@author: Dries Verdegem (Ghent University)
@author: Kenneth Hoste (Ghent University)
@author: Pieter De Baets (Ghent University)
@author: Jens Timmerman (Ghent University)
@author: Ward Poelmans (Ghent University)
@author: Luca Marsella (CSCS)
@author: Damian Alvarez (Forschungszentrum Juelich GmbH)
@author: Alan O'Cais (Forschungszentrum Juelich GmbH)
@author: Balazs Hajgato (Free University Brussels (VUB))
"""

import fileinput
import glob
import re
import os
import sys
from shlex import quote
from easybuild.tools import LooseVersion

import easybuild.tools.toolchain as toolchain
from easybuild.easyblocks.generic.cmakemake import CMakeMake
from easybuild.framework.easyconfig import CUSTOM
from easybuild.tools.build_log import EasyBuildError, print_warning
from easybuild.tools.environment import setvar
from easybuild.tools.filetools import change_dir, copy_dir, copy_file, mkdir, write_file
from easybuild.tools.config import build_option
from easybuild.tools.modules import get_software_root, get_software_version
from easybuild.tools.run import run_shell_cmd
from easybuild.tools.systemtools import get_avail_core_count, get_shared_lib_ext

# CP2K 2026.1 removed the Makefile-based build (https://github.com/cp2k/cp2k/releases/tag/v2026.1)
CMAKE_ONLY_VERSION = '2026.1'

# maxtasks easyconfig default for the Makefile build before it became None.
# CP2K < 2026.1 keeps running this many instances (the previous parameter default).
MAKEFILE_REGTEST_MAXTASKS = 4

# CMake option enabled for each dependency, and the cp2kflags that 'cp2k --version' reports for it
# (compile definitions in src/CMakeLists.txt, flags in src/cp2k_info.F)
CMAKE_DEP_OPTIONS = {
    'ELPA': ('CP2K_USE_ELPA', ['elpa']),
    'Libint': ('CP2K_USE_LIBINT2', ['libint']),
    'libvori': ('CP2K_USE_VORI', ['libvori', 'libbqb']),
    'libxc': ('CP2K_USE_LIBXC', ['libxc']),
    'PLUMED': ('CP2K_USE_PLUMED', ['plumed2']),
    'spglib': ('CP2K_USE_SPGLIB', ['spglib']),
    # CP2K_USE_TBLITE defines __DFTD4; tests/TEST_DIRS selects dispersion cases using these flags.
    'tblite': ('CP2K_USE_TBLITE', ['libdftd4', 'mctc-lib', 'tblite']),
}
# CP2K 2026.2 and later define __S_DFTD3 with CP2K_USE_TBLITE for the s_dftd3 test directories.
TBLITE_S_DFTD3_VERSION = '2026.2'
# options of dependencies that CP2K only honours together with CP2K_USE_MPI (cmake_dependent_option in CMakeLists.txt)
CMAKE_MPI_ONLY_OPTIONS = ['CP2K_USE_ELPA', 'CP2K_USE_PLUMED']

# per-input results in the summary printed by tests/do_regtest.py
REGTEST_SUMMARY_PATTERNS = {
    'failed': r'^Number of FAILED\s+tests\s+(\d+)\s*$',
    'wrong': r'^Number of WRONG\s+tests\s+(\d+)\s*$',
    'correct': r'^Number of CORRECT\s+tests\s+(\d+)\s*$',
    'total': r'^Total number of\s+tests\s+(\d+)\s*$',
}


def get_cmake_option(configopts, name):
    """Return the value given to CMake option 'name' by the last -D[ ]<name>[:<type>]=<value> in configopts, or None"""
    regex = re.compile(r"(?:^|\s)-D\s*%s(?::\w+)?=(\"[^\"]*\"|'[^']*'|\S*)" % re.escape(name))
    values = regex.findall(configopts)
    if values:
        return values[-1].strip('"\'')
    return None


def is_cmake_true(value):
    """
    Interpret the value of a CMake option as if(<variable>) does: true unless it is a false constant
    (0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, empty or ending in -NOTFOUND, in any case),
    see https://cmake.org/cmake/help/latest/command/if.html
    """
    value = value.upper()
    return not (value in ('', '0', 'OFF', 'NO', 'FALSE', 'N', 'IGNORE', 'NOTFOUND') or value.endswith('-NOTFOUND'))


def parse_regtest_summary(output):
    """
    Parse the summary that CP2K's tests/do_regtest.py prints at the end of a run

    :return: dict with the number of failed, wrong, correct and total tests, the number of tests without a
             result ('na', for which the comparison with the reference failed), and the 'status' ('OK' or 'FAILED'),
             or None if the output has no (complete) summary
    """
    summary = {}
    for key, pattern in REGTEST_SUMMARY_PATTERNS.items():
        counts = re.findall(pattern, output, re.M)
        if not counts:
            return None
        summary[key] = int(counts[-1])
    summary['na'] = summary['total'] - summary['correct'] - summary['wrong'] - summary['failed']

    status = re.findall(r'^Status:\s+(\S+)\s*$', output, re.M)
    if not status:
        return None
    summary['status'] = status[-1]

    return summary


def get_regtest_errors(output):
    """Return the 'Errors' section of the output of CP2K's tests/do_regtest.py"""
    res = re.search(r'^-+ Errors -+$\n(.*?)^-+ Timings -+$', output, re.M | re.S)
    return res.group(1).strip() if res else ''


def do_regtest_py_cmd(script, arguments, python_cmd=None, env_prefix=''):
    """Assemble the shell command that runs CP2K's tests/do_regtest.py.

    The Makefile path (py_regtest_cmd, CP2K < 2026.1) and the CMake path
    (regtest_cmd, CP2K >= 2026.1) both call this. arguments are shell tokens
    placed after the script. python_cmd defaults to this interpreter; the
    CMake path passes 'python3 -B' so a copied tests/ tree gains no __pycache__.
    """
    if python_cmd is None:
        python_cmd = sys.executable
    return env_prefix + ' '.join([python_cmd, script] + list(arguments))


def makefile_hdf5_sirius_options(options, hdf5_root, sirius_root):
    """Add HDF5 and SIRIUS support to the options of the Makefile build (CP2K < 2026.1).

    CP2K's INSTALL.md: -D__HDF5 enables HDF5, a hard dependency of SIRIUS; -D__SIRIUS enables SIRIUS.
    SIRIUS installs its Fortran module in include/sirius and links its own dependencies into libsirius.
    """
    if sirius_root and not hdf5_root:
        raise EasyBuildError("SIRIUS support in CP2K requires HDF5, but HDF5 is not a dependency")
    if hdf5_root:
        options['DFLAGS'] += ' -D__HDF5'
        options['FCFLAGSOPT'] += ' -I%s ' % os.path.join(hdf5_root, 'include')
        options['LIBS'] += ' -lhdf5_fortran -lhdf5'
    if sirius_root:
        options['DFLAGS'] += ' -D__SIRIUS'
        options['FCFLAGSOPT'] += ' -I%s ' % os.path.join(sirius_root, 'include', 'sirius')
        options['LIBS'] += ' -lsirius'
    return options


class EB_CP2K(CMakeMake):
    """
    Support for building CP2K
    - CP2K 2026.1 and newer: configure with CMake, building the CP2K_USE_* features that match the dependencies
    - older versions:
      - prepare module include files if required
      - generate custom config file in 'arch' directory
      - build CP2K
    - run regression test if desired
    - install by copying binary executables
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cmake_build = LooseVersion(self.version) >= LooseVersion(CMAKE_ONLY_VERSION)

        self.typearch = None

        # this should be set to False for old versions of GCC (e.g. v4.1)
        self.compilerISO_C_BINDING = True

        # compiler options that need to be set in Makefile
        self.debug = ''
        self.fpic = ''

        # used for both libsmm and libxsmm
        self.libsmm = ''
        self.modincpath = ''
        self.openmp = ''

        self.make_instructions = ''

    @staticmethod
    def extra_options():
        # update after CMakeMake, since it defines 'runtest' with another meaning and default
        extra_vars = CMakeMake.extra_options()
        extra_vars.update({
            'extracflags': ['', "Extra CFLAGS to be added (CP2K < %s)" % CMAKE_ONLY_VERSION, CUSTOM],
            'extradflags': ['', "Extra DFLAGS to be added (CP2K < %s)" % CMAKE_ONLY_VERSION, CUSTOM],
            'ignore_regtest_fails': [False, ("Ignore failures in regression test "
                                             "(should be used with care)"), CUSTOM],
            'library': [False, "Also build CP2K as a library (CP2K < %s, which always builds it)" % CMAKE_ONLY_VERSION,
                        CUSTOM],
            'maxtasks': [None, ("Maximum number of CP2K instances run at the same time during testing; "
                                "for CP2K >= %s the number of MPI ranks times OpenMP threads, "
                                "default: 4 for older versions, else the number of cores EasyBuild uses "
                                "(parallel)" % CMAKE_ONLY_VERSION), CUSTOM],
            'modinc': [[], ("List of modinc's to use (*.f90], or 'True' to use "
                            "all found at given prefix"), CUSTOM],
            'modincprefix': ['', "Intel MKL prefix for modinc include dir", CUSTOM],
            'runtest': [True, "Build and run CP2K tests", CUSTOM],
            'omp_num_threads': [None, "Value to set $OMP_NUM_THREADS to during testing", CUSTOM],
            'plumed': [None, "Enable PLUMED support", CUSTOM],
            'sanity_regtest': [True, ("Run the first input of each regression test directory with the installed "
                                      "binaries in the sanity check (CP2K >= %s)" % CMAKE_ONLY_VERSION), CUSTOM],
            'type': ['popt', "Type of build ('popt' or 'psmp'; CP2K < %s)" % CMAKE_ONLY_VERSION, CUSTOM],
            'typeopt': [True, "Enable optimization (CP2K < %s)" % CMAKE_ONLY_VERSION, CUSTOM],
        })
        return extra_vars

    @property
    def build_type(self):
        """
        CMake build type: CP2K's 'Release' adds -march=native (GNU) or -xHOST (Intel) after the compiler flags
        of EasyBuild, so use 'Generic' (as CP2K's generic toolchain build does), which keeps --optarch in effect
        """
        build_type = super().build_type
        if self.cfg.get('build_type') is None and build_type == 'Release':
            build_type = 'Generic'
        return build_type

    def _generate_makefile(self, options):
        """Generate Makefile based on options dictionary and optional make instructions"""

        text = "# Makefile generated by CP2K easyblock in EasyBuild\n"
        for key, value in sorted(options.items()):
            text += "%s = %s\n" % (key, value)
        return text + self.make_instructions

    def cmake_features(self):
        """
        Determine the CP2K features of a CMake build from the toolchain, the dependencies and configopts

        :return: dict with the CMake options to pass by default ('options'), the effective state of each
                 CP2K_USE_* option that the easyblock knows about ('enabled'), the cp2kflags reported by
                 'cp2k --version' for the enabled ones ('flags'), and the suffix of the binaries ('ext')
        """
        configopts = self.cfg['configopts']
        dep_names = [dep['name'] for dep in self.cfg.dependencies()]

        if 'CUDA' in dep_names and get_cmake_option(configopts, 'CP2K_USE_ACCEL') is None:
            raise EasyBuildError("CUDA is a dependency, but GPU support is not enabled: "
                                 "set -DCP2K_USE_ACCEL and -DCP2K_WITH_GPU in configopts")

        if self.cfg['plumed'] and 'PLUMED' not in dep_names:
            raise EasyBuildError("PLUMED support is enabled (plumed = True), but PLUMED is not a dependency")

        options = {}
        if self.toolchain.mpi_family():
            options['CP2K_USE_MPI'] = 'ON'
            options['CP2K_USE_MPI_F08'] = 'ON'
        if getattr(self.toolchain, 'FFT_MODULE_NAME', None) or 'FFTW' in dep_names:
            options['CP2K_USE_FFTW3'] = 'ON'
        for dep_name, (option, _) in sorted(CMAKE_DEP_OPTIONS.items()):
            if dep_name in dep_names and not (dep_name == 'PLUMED' and self.cfg['plumed'] is False):
                options[option] = 'ON'

        known_options = ['CP2K_USE_MPI', 'CP2K_USE_MPI_F08', 'CP2K_USE_FFTW3']
        known_options += [option for option, _ in CMAKE_DEP_OPTIONS.values()]
        enabled = {}
        for option in known_options:
            value = get_cmake_option(configopts, option)
            if value is None:
                value = options.get(option, 'OFF')
            enabled[option] = is_cmake_true(value)

        if not enabled['CP2K_USE_MPI']:
            mpi_only = [opt for opt in CMAKE_MPI_ONLY_OPTIONS if enabled[opt]]
            if mpi_only:
                raise EasyBuildError("CP2K only supports %s with MPI (CP2K_USE_MPI)", ', '.join(mpi_only))
            # CMakeLists.txt also turns CP2K_USE_MPI_F08 off without MPI
            enabled['CP2K_USE_MPI_F08'] = False

        # CP2K always requires OpenMP (find_package(OpenMP REQUIRED ...) in CMakeLists.txt)
        flags = ['omp']
        if enabled['CP2K_USE_MPI']:
            flags.extend(['parallel', 'scalapack'])
        if enabled['CP2K_USE_MPI_F08']:
            flags.append('mpi_f08')
        if enabled['CP2K_USE_FFTW3']:
            flags.append('fftw3')
        for option, option_flags in sorted(CMAKE_DEP_OPTIONS.values()):
            if enabled[option]:
                flags.extend(option_flags)

        # suffix of the binaries, see __cp2k_ext in CMakeLists.txt
        build_type = get_cmake_option(configopts, 'CMAKE_BUILD_TYPE') or self.build_type
        ext = ('p' if enabled['CP2K_USE_MPI'] else 's') + ('dbg' if build_type == 'Debug' else 'smp')

        return {'options': options, 'enabled': enabled, 'flags': flags, 'ext': ext}

    def configure_step_cmake(self):
        """Configure CP2K with CMake, enabling the features that match the toolchain and the dependencies"""

        options = self.cmake_features()['options']

        blas_vendors = {
            toolchain.FLEXIBLAS: 'FlexiBLAS',
            toolchain.INTELMKL: 'MKL',
            toolchain.OPENBLAS: 'OpenBLAS',
        }
        blas_vendor = blas_vendors.get(self.toolchain.blas_family())
        if blas_vendor:
            options['CP2K_BLAS_VENDOR'] = blas_vendor

        self.prepend_config_opts(options)

        return super().configure_step()

    def configure_step(self):
        """Configure build
        - build Libint wrapper
        - generate Makefile
        """

        if self.cmake_build:
            return self.configure_step_cmake()

        known_types = ['popt', 'psmp']
        if self.cfg['type'] not in known_types:
            raise EasyBuildError("Unknown build type specified: '%s', known types are %s",
                                 self.cfg['type'], known_types)

        # correct start dir, if needed
        # recent CP2K versions have a 'cp2k' dir in the unpacked 'cp2k' dir
        cp2k_path = os.path.join(self.cfg['start_dir'], 'cp2k')
        if os.path.exists(cp2k_path):
            self.cfg['start_dir'] = cp2k_path
            self.log.info("Corrected start_dir to %s" % self.cfg['start_dir'])

        # set compilers options according to toolchain config
        # full debug: -g -traceback -check all -fp-stack-check
        # -g links to mpi debug libs
        if self.toolchain.options['debug']:
            self.debug = '-g'
            self.log.info("Debug build")
        if self.toolchain.options['pic']:
            self.fpic = "-fPIC"
            self.log.info("Using fPIC")

        # report on extra flags being used
        if self.cfg['extracflags']:
            self.log.info("Using extra CFLAGS: %s" % self.cfg['extracflags'])
        if self.cfg['extradflags']:
            self.log.info("Using extra DFLAGS: %s" % self.cfg['extradflags'])

        # lib(x)smm support
        libsmm = get_software_root('libsmm')
        libxsmm = get_software_root('libxsmm')
        if libxsmm:
            self.cfg.update('extradflags', '-D__LIBXSMM')
            self.libsmm = '-lxsmm -lxsmmf'
            self.log.debug('Using libxsmm %s' % libxsmm)
        elif libsmm:
            libsmms = glob.glob(os.path.join(libsmm, 'lib', 'libsmm_*nn.a'))
            dfs = [os.path.basename(os.path.splitext(x)[0]).replace('lib', '-D__HAS_') for x in libsmms]
            moredflags = ' ' + ' '.join(dfs)
            self.cfg.update('extradflags', moredflags)
            self.libsmm = ' '.join(libsmms)
            self.log.debug('Using libsmm %s (extradflags %s)' % (self.libsmm, moredflags))

        # obtain list of modinc's to use
        if self.cfg["modinc"]:
            self.modincpath = self.prepmodinc()

        # set typearch
        self.typearch = "Linux-x86-64-%s" % self.toolchain.name

        # extra make instructions
        self.make_instructions = ''  # "graphcon.o: graphcon.F\n\t$(FC) -c $(FCFLAGS2) $<\n"

        # compiler toolchain specific configuration
        comp_fam = self.toolchain.comp_family()
        if comp_fam == toolchain.INTELCOMP:
            options = self.configure_intel_based()
        elif comp_fam == toolchain.GCC:
            options = self.configure_GCC_based()
        else:
            raise EasyBuildError("Don't know how to tweak configuration for compiler family %s" % comp_fam)

        # BLAS/LAPACK/FFTW
        if get_software_root('imkl'):
            options = self.configure_MKL(options)
        else:
            # BLAS
            if get_software_root('ACML'):
                options = self.configure_ACML(options)
            else:
                options = self.configure_BLAS_lib(options)

            # FFTW (no MKL involved)
            if 'fftw3' in os.getenv('LIBFFT', ''):
                options = self.configure_FFTW3(options)

            # LAPACK
            if os.getenv('LIBLAPACK_MT', None) is not None:
                options = self.configure_LAPACK(options)

            if os.getenv('LIBSCALAPACK', None) is not None:
                options = self.configure_ScaLAPACK(options)

        # PLUMED
        plumed = get_software_root('PLUMED')
        if self.cfg['plumed'] and not plumed:
            raise EasyBuildError("The PLUMED module needs to be loaded to build CP2K with PLUMED support")

        # enable PLUMED support if PLUMED is listed as a dependency
        # and PLUMED support is either explicitly enabled (plumed = True) or unspecified ('plumed' not defined)
        if plumed and (self.cfg['plumed'] or self.cfg['plumed'] is None):
            options['LIBS'] += ' -lplumed'
            options['DFLAGS'] += ' -D__PLUMED2'

        # ELPA
        elpa = get_software_root('ELPA')
        if elpa:
            options['LIBS'] += ' -lelpa'
            elpa_inc_dir = os.path.join(elpa, 'include', 'elpa-%s' % get_software_version('ELPA'), 'modules')
            options['FCFLAGSOPT'] += ' -I%s ' % elpa_inc_dir
            if LooseVersion(self.version) >= LooseVersion('6.1'):
                elpa_ver = ''.join(get_software_version('ELPA').split('.')[:2])
                options['DFLAGS'] += ' -D__ELPA=%s' % elpa_ver
                elpa_inc_dir = os.path.join(elpa, 'include', 'elpa-%s' % get_software_version('ELPA'), 'elpa')
                options['FCFLAGSOPT'] += ' -I%s ' % elpa_inc_dir
            else:
                options['DFLAGS'] += ' -D__ELPA3'

        options = makefile_hdf5_sirius_options(options, get_software_root('HDF5'), get_software_root('SIRIUS'))

        # CUDA
        cuda = get_software_root('CUDA')
        if cuda:
            options['DFLAGS'] += ' -D__ACC -D__DBCSR_ACC'
            options['LIBS'] += ' -lcudart -lcublas -lcufft -lrt'
            options['NVCC'] = ' nvcc'

        # avoid group nesting
        options['LIBS'] = options['LIBS'].replace('-Wl,--start-group', '').replace('-Wl,--end-group', '')

        options['LIBS'] = "-Wl,--start-group %s -Wl,--end-group" % options['LIBS']

        # specify correct location for 'data' directory in final installation
        options['DATA_DIR'] = os.path.join(self.installdir, 'data')

        # create arch file using options set
        archfile = os.path.join(self.cfg['start_dir'], 'arch', '%s.%s' % (self.typearch, self.cfg['type']))
        txt = self._generate_makefile(options)
        write_file(archfile, txt)
        self.log.info("Content of makefile (%s):\n%s" % (archfile, txt))

    def prepmodinc(self):
        """Prepare list of module files"""

        self.log.debug("Preparing module files")

        imkl = get_software_root('imkl')

        if imkl:

            # prepare modinc target path
            modincpath = os.path.join(os.path.dirname(os.path.normpath(self.cfg['start_dir'])), 'modinc')
            self.log.debug("Preparing module files in %s" % modincpath)

            mkdir(modincpath, parents=True)

            # get list of modinc source files
            modincdir = os.path.join(os.getenv('MKLROOT'), self.cfg["modincprefix"], 'include')

            if isinstance(self.cfg["modinc"], list):
                modfiles = [os.path.join(modincdir, x) for x in self.cfg["modinc"]]

            elif isinstance(self.cfg["modinc"], bool) and self.cfg["modinc"]:
                modfiles = glob.glob(os.path.join(modincdir, '*.f90'))

            else:
                raise EasyBuildError("prepmodinc: Please specify either a boolean value or a list of files in modinc "
                                     "(found: %s).", self.cfg["modinc"])

            f77 = os.getenv('F77')
            if not f77:
                raise EasyBuildError("F77 environment variable not set, can't continue.")

            # create modinc files
            for f in modfiles:
                if f77.endswith('ifort'):
                    cmd = "%s -module %s -c %s" % (f77, modincpath, f)
                elif f77 in ['gfortran', 'mpif77']:
                    cmd = "%s -J%s -c %s" % (f77, modincpath, f)
                else:
                    raise EasyBuildError("prepmodinc: Unknown value specified for F77 (%s)", f77)

                run_shell_cmd(cmd)

            return modincpath
        else:
            raise EasyBuildError("Don't know how to prepare modinc, imkl not found")

    def configure_common(self):
        """Common configuration for all toolchains"""

        # openmp introduces 2 major differences
        # -automatic is default: -noautomatic -auto-scalar
        # some mem-bandwidth optimisation
        if self.cfg['type'] == 'psmp':
            self.openmp = self.toolchain.get_flag('openmp')

        # determine which opt flags to use
        if self.cfg['typeopt']:
            optflags = 'OPT'
            regflags = 'OPT2'
        else:
            optflags = 'NOOPT'
            regflags = 'NOOPT'

        # make sure a MPI-2 able MPI lib is used
        mpi2 = False
        if hasattr(self.toolchain, 'MPI_FAMILY') and self.toolchain.MPI_FAMILY is not None:
            known_mpi2_fams = [toolchain.MPICH, toolchain.MPICH2, toolchain.MVAPICH2, toolchain.OPENMPI,
                               toolchain.INTELMPI]
            mpi_fam = self.toolchain.mpi_family()
            if mpi_fam in known_mpi2_fams:
                mpi2 = True
                self.log.debug("Determined MPI2 compatibility based on MPI toolchain component: %s" % mpi_fam)
            else:
                self.log.debug("Cannot determine MPI2 compatibility based on MPI toolchain component: %s" % mpi_fam)
        else:
            # can't use toolchain.mpi_family, because of system toolchain
            mpi2libs = ['impi', 'MVAPICH2', 'OpenMPI', 'MPICH2', 'MPICH']
            for mpi2lib in mpi2libs:
                if get_software_root(mpi2lib):
                    mpi2 = True
                    self.log.debug("Determined MPI2 compatibility based on loaded MPI module: %s")
                else:
                    self.log.debug("MPI-2 supporting MPI library %s not loaded.")

        if not mpi2:
            raise EasyBuildError("CP2K needs MPI-2, no known MPI-2 supporting library loaded?")

        cppflags = os.getenv('CPPFLAGS')
        ldflags = os.getenv('LDFLAGS')
        cflags = os.getenv('CFLAGS')
        fflags = os.getenv('FFLAGS')
        fflags_lowopt = re.sub('-O[0-9]', '-O1', fflags)
        options = {
            'CC': os.getenv('MPICC'),
            'CPP': '',
            'FC': '%s %s' % (os.getenv('MPIF90'), self.openmp),
            'LD': '%s %s' % (os.getenv('MPIF90'), self.openmp),
            'AR': 'ar -r',
            'CPPFLAGS': '',

            'FPIC': self.fpic,
            'DEBUG': self.debug,

            'FCFLAGS': '$(FCFLAGS%s)' % optflags,
            'FCFLAGS2': '$(FCFLAGS%s)' % regflags,

            'CFLAGS': ' $(DFLAGS) %s %s %s $(FPIC) $(DEBUG) %s ' % (
                cflags, cppflags, ldflags, self.cfg['extracflags']),
            'DFLAGS': ' -D__parallel -D__BLACS -D__SCALAPACK -D__FFTSG %s' % self.cfg['extradflags'],

            'LIBS': os.getenv('LIBS', ''),

            'FCFLAGSNOOPT': '$(DFLAGS) $(CFLAGS) -O0  $(FREE) $(FPIC) $(DEBUG)',
            'FCFLAGSOPT': '%s $(FREE) $(SAFE) $(FPIC) $(DEBUG)' % fflags,
            'FCFLAGSOPT2': '%s $(FREE) $(SAFE) $(FPIC) $(DEBUG)' % fflags_lowopt,
        }

        libint = get_software_root('LibInt')
        if libint:
            options['DFLAGS'] += ' -D__LIBINT'

            libintcompiler = "%s %s" % (os.getenv('CC'), os.getenv('CFLAGS'))

            # Build libint-wrapper, if required
            libint_wrapper = ''

            # required for old versions of GCC
            if not self.compilerISO_C_BINDING:
                options['DFLAGS'] += ' -D__HAS_NO_ISO_C_BINDING'

                # determine path for libint_tools dir
                libinttools_paths = ['libint_tools', 'tools/hfx_tools/libint_tools']
                libinttools_path = None
                for path in libinttools_paths:
                    path = os.path.join(self.cfg['start_dir'], path)
                    if os.path.isdir(path):
                        libinttools_path = path
                        change_dir(libinttools_path)
                if not libinttools_path:
                    raise EasyBuildError("No libinttools dir found")

                # build libint wrapper
                cmd = "%s -c libint_cpp_wrapper.cpp -I%s/include" % (libintcompiler, libint)
                res = run_shell_cmd(cmd, fail_on_error=False)
                if res.exit_code:
                    raise EasyBuildError("Building the libint wrapper failed")
                libint_wrapper = '%s/libint_cpp_wrapper.o' % libinttools_path

            # determine Libint libraries based on major version number
            libint_maj_ver = get_software_version('Libint').split('.')[0]
            if libint_maj_ver == '1':
                libint_libs = "$(LIBINTLIB)/libderiv.a $(LIBINTLIB)/libint.a $(LIBINTLIB)/libr12.a"
            elif libint_maj_ver == '2':
                libint_libs = "$(LIBINTLIB)/libint2.a"
            else:
                raise EasyBuildError("Don't know how to handle libint version %s", libint_maj_ver)
            self.log.info("Using Libint version %s" % (libint_maj_ver))

            options['LIBINTLIB'] = '%s/lib' % libint
            options['LIBS'] += ' %s -lstdc++ %s' % (libint_libs, libint_wrapper)

            # add Libint include dir to $FCFLAGS
            options['FCFLAGS'] += ' -I' + os.path.join(libint, 'include')

        else:
            # throw a warning, since CP2K without Libint doesn't make much sense
            self.log.warning("Libint module not loaded, so building without Libint support")

        libxc = get_software_root('libxc')
        if libxc:
            cur_libxc_version = get_software_version('libxc')
            if LooseVersion(self.version) >= LooseVersion('6.1'):
                libxc_min_version = '4.0.3'
                options['DFLAGS'] += ' -D__LIBXC'
            else:
                libxc_min_version = '2.0.1'
                options['DFLAGS'] += ' -D__LIBXC2'

            if LooseVersion(cur_libxc_version) < LooseVersion(libxc_min_version):
                raise EasyBuildError("This version of CP2K is not compatible with libxc < %s" % libxc_min_version)

            if LooseVersion(cur_libxc_version) >= LooseVersion('4.0.3'):
                # cfr. https://www.cp2k.org/howto:compile#k_libxc_optional_wider_choice_of_xc_functionals
                options['LIBS'] += ' -L%s/lib -lxcf03 -lxc' % libxc
            elif LooseVersion(cur_libxc_version) >= LooseVersion('2.2'):
                options['LIBS'] += ' -L%s/lib -lxcf90 -lxc' % libxc
            else:
                options['LIBS'] += ' -L%s/lib -lxc' % libxc
            self.log.info("Using Libxc-%s" % cur_libxc_version)
        else:
            self.log.info("libxc module not loaded, so building without libxc support")

        libvori = get_software_root('libvori')
        if libvori:
            if LooseVersion(self.version) >= LooseVersion('8.1'):
                options['LIBS'] += ' -lvori'
                options['DFLAGS'] += ' -D__LIBVORI'
            else:
                raise EasyBuildError("This version of CP2K does not suppport libvori")
        else:
            self.log.info("libvori module not loaded, so building without support for Voronoi integration")

        return options

    def configure_intel_based(self):
        """Configure for Intel based toolchains"""

        # based on guidelines available at
        # http://software.intel.com/en-us/articles/build-cp2k-using-intel-fortran-compiler-professional-edition/
        intelurl = ''.join(["http://software.intel.com/en-us/articles/",
                            "build-cp2k-using-intel-fortran-compiler-professional-edition/"])

        options = self.configure_common()

        extrainc = ''
        if self.modincpath:
            extrainc = '-I%s' % self.modincpath

        options.update({
            # -Vaxlib : older options
            'FREE': '-fpp -free',

            # SAFE = -assume protect_parens -fp-model precise -ftz  # causes problems, so don't use this
            'SAFE': '-assume protect_parens -no-unroll-aggressive',

            'INCFLAGS': '$(DFLAGS) -I$(INTEL_INC) -I$(INTEL_INCF) %s' % extrainc,

            'LDFLAGS': '$(INCFLAGS) ',
            'OBJECTS_ARCHITECTURE': 'machine_intel.o',
        })

        options['DFLAGS'] += ' -D__INTEL'

        options['FCFLAGSOPT'] += ' $(INCFLAGS) -heap-arrays 64'
        options['FCFLAGSOPT2'] += ' $(INCFLAGS) -heap-arrays 64'

        # for recent intel toolchains (>= intel/2021a), intel-compilers is the toolchain component
        ifortver = get_software_version('intel-compilers')
        if ifortver is None:
            # fall back to trying to determining Intel Fortran compiler version using 'ifort' as software name
            ifortver = get_software_version('ifort')

        if ifortver:
            ifortver = LooseVersion(ifortver)
        else:
            raise EasyBuildError("Failed to determine Intel Fortran compiler version!")

        # Required due to memory leak that occurs if high optimizations are used (from CP2K 7.1 intel-popt-makefile)
        if ifortver >= LooseVersion("2018.5"):
            self.make_instructions += "mp2_optimize_ri_basis.o: mp2_optimize_ri_basis.F\n" \
                "\t$(FC) -c $(subst O2,O0,$(FCFLAGSOPT)) $<\n"
            self.log.info("Optimization level of mp2_optimize_ri_basis.F was decreased to '-O0'")

        # RHEL8 intel/2020a lots of CPASSERT failed (due to high optimization in cholesky decomposition)
        if ifortver >= LooseVersion("2019"):
            self.make_instructions += "cp_fm_cholesky.o: cp_fm_cholesky.F\n\t$(FC) -c $(FCFLAGS2) $<\n"
            self.log.info("Optimization flags for cp_fm_cholesky.F is set to '%s'", options['FCFLAGSOPT2'])

        # -i-static has been deprecated prior to 2013, but was still usable. From 2015 it is not.
        if ifortver < LooseVersion("2013"):
            options['LDFLAGS'] += ' -i-static '
        else:
            options['LDFLAGS'] += ' -static-intel '

        # Otherwise it fails on linking, since there are 2 definitions of main
        if LooseVersion(self.version) >= LooseVersion('4.1'):
            options['LDFLAGS'] += ' -nofor-main '

        failmsg = "CP2K won't build correctly with the Intel %%s compilers prior to %%s, see %s" % intelurl

        if ifortver >= LooseVersion("2011") and ifortver < LooseVersion("2012"):

            # don't allow using Intel compiler 2011 prior to release 8, because of known issue (see Intel URL)
            if ifortver >= LooseVersion("2011.8"):
                # add additional make instructions to Makefile
                self.make_instructions += "et_coupling.o: et_coupling.F\n\t$(FC) -c $(FCFLAGS2) $<\n"
                self.make_instructions += "qs_vxc_atom.o: qs_vxc_atom.F\n\t$(FC) -c $(FCFLAGS2) $<\n"

            else:
                raise EasyBuildError(failmsg, "v12", "v2011.8")

        elif ifortver >= LooseVersion("11"):
            if ifortver >= LooseVersion("11.1.072"):
                self.make_instructions += "qs_vxc_atom.o: qs_vxc_atom.F\n\t$(FC) -c $(FCFLAGS2) $<\n"

            else:
                raise EasyBuildError(failmsg, "v11", "v11.1.072")

        else:
            raise EasyBuildError("Intel compilers version %s not supported yet.", ifortver)

        return options

    def configure_GCC_based(self):
        """Configure for GCC based toolchains"""
        options = self.configure_common()

        options.update({
            # need this to prevent "Unterminated character constant beginning" errors
            'FREE': '-ffree-form -ffree-line-length-none',

            'LDFLAGS': '$(FCFLAGS)',
            'OBJECTS_ARCHITECTURE': 'machine_gfortran.o',
        })

        options['DFLAGS'] += ' -D__GFORTRAN'

        options['FCFLAGSOPT'] += ' $(DFLAGS) $(CFLAGS)'
        if not self.toolchain.options.get('openmp', None):
            # an explicit -fmax-stack-var-size overrides the -frecursive implied by -fopenmp and puts local
            # arrays above the limit in static memory shared by all threads, so it is only set without OpenMP
            options['FCFLAGSOPT'] += ' -fmax-stack-var-size=32768'
        options['FCFLAGSOPT2'] += ' $(DFLAGS) $(CFLAGS)'

        gcc_version = get_software_version('GCCcore') or get_software_version('GCC')
        if LooseVersion(gcc_version) >= LooseVersion('10.0') and LooseVersion(self.version) <= LooseVersion('7.1'):
            # -fallow-argument-mismatch is required for CP2K 7.1 (and older) when compiling with GCC 10.x & more recent,
            # see https://github.com/cp2k/cp2k/issues/1157, https://github.com/cp2k/dbcsr/issues/351,
            # https://github.com/cp2k/dbcsr/commit/58ee9709545deda8524cab804bf1f88a61a864ac and
            # https://gcc.gnu.org/legacy-ml/gcc-patches/2019-10/msg01861.html
            options['FCFLAGSOPT'] += ' -fallow-argument-mismatch'
            options['FCFLAGSOPT2'] += ' -fallow-argument-mismatch'

        # GCC 15 loop vectorization corrupts integer packing on AVX2 and AVX-512 targets.
        # The target-specific flag preserves optimization for the other objects.
        compression_source = os.path.join(self.cfg['start_dir'], 'src', 'hfxbase',
                                          'hfx_compression_core_methods.F')
        if (LooseVersion('15.0') <= LooseVersion(gcc_version) < LooseVersion('16.0')
                and os.path.exists(compression_source)):
            self.make_instructions += 'hfx_compression_core_methods.o: FCFLAGS += -fno-tree-loop-vectorize\n'

        return options

    def configure_ACML(self, options):
        """Configure for AMD Math Core Library (ACML)"""

        openmp_suffix = ''
        if self.openmp:
            openmp_suffix = '_mp'

        options['ACML_INC'] = '%s/gfortran64%s/include' % (get_software_root('ACML'), openmp_suffix)
        options['CFLAGS'] += ' -I$(ACML_INC) -I$(FFTW_INC)'
        options['DFLAGS'] += ' -D__FFTACML'

        blas = os.getenv('LIBBLAS', '')
        blas = blas.replace('gfortran64', 'gfortran64%s' % openmp_suffix)
        options['LIBS'] += ' %s %s %s' % (self.libsmm, os.getenv('LIBSCALAPACK', ''), blas)

        return options

    def configure_BLAS_lib(self, options):
        """Configure for BLAS library."""
        options['LIBS'] += ' %s %s' % (self.libsmm, os.getenv('LIBBLAS', ''))
        return options

    def configure_MKL(self, options):
        """Configure for Intel Math Kernel Library (MKL)"""

        options['INTEL_INC'] = '$(MKLROOT)/include'
        options['DFLAGS'] += ' -D__FFTW3'

        extra = ''
        if self.modincpath:
            extra = '-I%s' % self.modincpath
        options['CFLAGS'] += ' -I$(INTEL_INC) %s $(FPIC) $(DEBUG)' % extra

        options['LIBS'] += ' %s %s' % (self.libsmm, os.getenv('LIBSCALAPACK', ''))

        fftw_root = get_software_root('FFTW')
        if fftw_root:
            libfft = '-lfftw3'
            if self.cfg['type'] == 'psmp':
                libfft += ' -lfftw3_omp'

            options['CFLAGS'] += ' -I$(INTEL_INCF)'
            options['INTEL_INCF'] = os.path.join(fftw_root, 'include')
            options['LIBS'] += ' -L%s %s' % (os.path.join(fftw_root, 'lib'), libfft)

        else:
            # only use Intel FFTW wrappers if FFTW is not loaded
            options['CFLAGS'] += ' -I$(INTEL_INCF)'
            if LooseVersion(self.version) > LooseVersion('2.3'):
                options['DFLAGS'] += ' -D__MKL'
            else:
                options['DFLAGS'] += ' -D__FFTMKL'
            options['INTEL_INCF'] = '$(INTEL_INC)/fftw'
            options['LIBS'] = '%s %s' % (os.getenv('LIBFFT', ''), options['LIBS'])

        return options

    def configure_FFTW3(self, options):
        """Configure for FFTW3"""

        options.update({
            'FFTW_INC': os.getenv('FFT_INC_DIR', ''),  # GCC
            'FFTW3INC': os.getenv('FFT_INC_DIR', ''),  # Intel
            'FFTW3LIB': os.getenv('FFT_LIB_DIR', ''),  # Intel
        })

        options['DFLAGS'] += ' -D__FFTW3'
        if self.cfg['type'] == 'psmp':
            libfft = os.getenv('LIBFFT_MT', '')
        else:
            libfft = os.getenv('LIBFFT', '')
        options['LIBS'] += ' -L%s %s' % (os.getenv('FFT_LIB_DIR', '.'), libfft)

        return options

    def configure_LAPACK(self, options):
        """Configure for LAPACK library"""
        options['LIBS'] += ' %s' % os.getenv('LIBLAPACK_MT', '')
        return options

    def configure_ScaLAPACK(self, options):
        """Configure for ScaLAPACK library"""

        options['LIBS'] += ' %s' % os.getenv('LIBSCALAPACK', '')

        return options

    def build_step(self):
        """Start the actual build
        - go into makefiles dir
        - patch Makefile
        -build_and_install
        """

        if self.cmake_build:
            return super().build_step()

        if LooseVersion(self.version) < LooseVersion('7.0'):
            makefiles = os.path.join(self.cfg['start_dir'], 'makefiles')
            change_dir(makefiles)

            # modify makefile for parallel build
            parallel = self.cfg.parallel
            if parallel > 1:
                try:
                    for line in fileinput.input('Makefile', inplace=1, backup='.orig.patchictce'):
                        line = re.sub(r"^PMAKE\s*=.*$", f"PMAKE\t= $(SMAKE) -j {parallel}", line)
                        sys.stdout.write(line)
                except IOError as err:
                    raise EasyBuildError("Can't modify/write Makefile in %s: %s", makefiles, err)

        # update make options with MAKE
        self.cfg.update('buildopts', f'MAKE="make -j {self.cfg.parallel}"')

        # update make options with ARCH and VERSION
        self.cfg.update('buildopts', 'ARCH=%s VERSION=%s' % (self.typearch, self.cfg['type']))

        cmd = "make %s" % self.cfg['buildopts']

        # clean first
        run_shell_cmd(cmd + " clean")

        # build and install
        # compile regularly first with the default make target
        # and only then build the library
        run_shell_cmd(cmd + ' all')

        # build as a library
        if self.cfg['library']:
            run_shell_cmd(cmd + 'libcp2k')

    def py_regtest_cmd(self, script, num_ranks):
        """Construct command to run the Python regression test driver of CP2K (do_regtest.py)."""

        with open(script, 'r', encoding='utf-8', errors='replace') as fh:
            script_txt = fh.read()

        # CP2K 2025.1 and later take the directory with the executables, earlier versions take the arch
        if 'binary_dir' in script_txt:
            target = os.path.join(self.cfg['start_dir'], 'exe', self.typearch)
        else:
            target = self.typearch

        # None is the easyconfig default; the Makefile path still uses its historical count
        maxtasks = self.cfg['maxtasks']
        if maxtasks is None:
            maxtasks = MAKEFILE_REGTEST_MAXTASKS
        cmd = ['--mpiranks', str(num_ranks), '--maxtasks', str(maxtasks)]
        if self.cfg['omp_num_threads']:
            cmd.extend(['--ompthreads', str(self.cfg['omp_num_threads'])])

        # --mpiexec is a template with {N} for the number of ranks since CP2K 2024.2, earlier versions
        # append '-n <ranks>' themselves
        placeholder = '@RANKS@'
        mpiexec = self.toolchain.mpi_cmd_for('', placeholder).strip()
        if '{N}' in script_txt:
            mpiexec = mpiexec.replace(placeholder, '{N}')
        else:
            mpiexec = re.sub(r'\s+(-n|-np|--np)\s+' + placeholder, '', mpiexec)
        if placeholder in mpiexec:
            raise EasyBuildError("Failed to construct MPI launcher for the regression test from '%s'", mpiexec)
        cmd.extend(['--mpiexec', quote(mpiexec)])
        cmd.extend([quote(target), self.cfg['type']])

        return do_regtest_py_cmd(script, cmd)

    def regtest_env_cmd(self):
        """Command that lets mpiexec start more MPI ranks than there are slots, as the regression tests may"""
        if self.toolchain.mpi_family() != toolchain.OPENMPI:
            return ''
        ompi_ver = get_software_version('OpenMPI')
        if ompi_ver and LooseVersion(ompi_ver) < LooseVersion('5.0'):
            return 'export OMPI_MCA_rmaps_base_oversubscribe=1 && '
        # also when the version is unknown since OpenMPI is not loaded yet (--sanity-check-only): Open MPI 4
        # ignores this, and EasyBuild sets $OMPI_MCA_rmaps_base_oversubscribe for sanity check commands
        return 'export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe && '

    def regtest_cmd(self, cp2k_root, bin_dir, ext, opts):
        """
        Command to run CP2K's regression test driver

        :param cp2k_root: directory with tests/do_regtest.py, and with src/grid/sample_tasks for grid_unittest
        :param bin_dir: directory with the CP2K binaries to test
        :param ext: suffix of the CP2K binaries (psmp, ssmp, ...)
        :param opts: extra options for the test driver
        """
        # the driver runs at most maxtasks / (MPI ranks * OpenMP threads) inputs at the same time
        regtest_opts = ['--maxtasks %s' % (self.cfg['maxtasks'] or self.cfg.parallel)]
        if self.cfg['omp_num_threads']:
            regtest_opts.append('--ompthreads %s' % self.cfg['omp_num_threads'])
        regtest_opts.extend(opt for opt in opts if opt)

        # -B: no __pycache__ next to the driver, which may be copied into (or be part of) the installation
        driver = os.path.join(cp2k_root, 'tests', 'do_regtest.py')
        return do_regtest_py_cmd(driver, regtest_opts + [bin_dir, ext], python_cmd='python3 -B',
                                 env_prefix=self.regtest_env_cmd())

    def test_step_cmake(self):
        """Run CP2K's regression tests with the binaries in the build directory"""

        if not self.cfg['runtest']:
            return

        features = self.cmake_features()
        if features['enabled']['CP2K_USE_MPI'] and not build_option('mpi_tests'):
            self.log.info("Skipping testing of CP2K since MPI testing is disabled")
            return

        start_dir = self.cfg['start_dir']
        opts = [
            '--workbasedir %s' % os.path.join(self.builddir, 'regtesting'),
            '--cp2kdatadir %s' % os.path.join(start_dir, 'data'),
            self.cfg['testopts'],
        ]
        bin_dir = os.path.join(self.separate_build_dir or start_dir, 'bin')
        cmd = ' '.join([self.cfg['pretestopts'], self.regtest_cmd(start_dir, bin_dir, features['ext'], opts)])
        res = run_shell_cmd(cmd, fail_on_error=False)

        summary = parse_regtest_summary(res.output)
        if summary is None:
            self.report_test_failure("No summary in the output of CP2K's regression tests (exit code %s)" %
                                     res.exit_code)
            return

        msg = ("CP2K regression tests: %(correct)d of %(total)d correct, %(wrong)d wrong, %(failed)d failed, "
               "%(na)d without result" % summary)
        # the driver reports status OK only if every test is correct
        if res.exit_code == 0 and summary['status'] == 'OK':
            self.log.info(msg)
        else:
            self.log.warning("Errors reported by CP2K's regression tests:\n%s", get_regtest_errors(res.output))
            if self.cfg['ignore_regtest_fails']:
                print_warning("%s; ignoring this, as requested (ignore_regtest_fails)", msg)
            else:
                self.report_test_failure(msg)

    def test_step(self):
        """Run regression test."""

        if self.cmake_build:
            return self.test_step_cmake()

        if self.cfg['runtest']:

            # we need to specify location of 'data' directory in *build* dir,
            # since we've configured CP2K to look into the installation directory
            # (where 'data' will be copied to in install step)
            setvar('CP2K_DATA_DIR', os.path.join(self.cfg['start_dir'], 'data'))

            if not build_option('mpi_tests'):
                self.log.info("Skipping testing of CP2K since MPI testing is disabled")
                return

            if self.cfg['omp_num_threads']:
                setvar('OMP_NUM_THREADS', self.cfg['omp_num_threads'])

            # change to root of build dir
            change_dir(self.builddir)

            # use regression test reference output if available
            # try and find an unpacked directory that starts with 'LAST-'
            regtest_refdir = None
            for d in os.listdir(self.builddir):
                if d.startswith("LAST-"):
                    regtest_refdir = d
                    break

            # location of do_regtest script
            cfg_fn = 'cp2k_regtest.cfg'

            # CP2K 2023.2 and later ship a Python driver (tools/regtesting/do_regtest.py, moved to
            # tests/do_regtest.py in CP2K 2024.1) instead of the shell script do_regtest
            py_regtest_script = None
            regtest_script = os.path.join(self.cfg['start_dir'], 'tools', 'regtesting', 'do_regtest')
            regtest_cmd = [regtest_script, '-nobuild', '-config', cfg_fn]
            if LooseVersion(self.version) < LooseVersion('7.1'):
                # -nosvn option was removed in CP2K 7.1
                regtest_cmd.insert(1, '-nosvn')

            if not os.path.exists(regtest_script):
                # older version of CP2K
                old_regtest_script = os.path.join(self.cfg['start_dir'], 'tools', 'do_regtest')
                if os.path.exists(old_regtest_script):
                    regtest_script = old_regtest_script
                    regtest_cmd = [regtest_script, '-nocvs', '-quick', '-nocompile', '-config', cfg_fn]
                else:
                    for subdir in (('tests',), ('tools', 'regtesting')):
                        candidate = os.path.join(self.cfg['start_dir'], *(subdir + ('do_regtest.py',)))
                        if os.path.exists(candidate):
                            py_regtest_script = candidate
                            break

            # patch do_regtest so that reference output is used (shell script only)
            if py_regtest_script:
                self.log.info("Using Python regression test driver %s", py_regtest_script)
            elif regtest_refdir:
                self.log.info("Using reference output available in %s" % regtest_refdir)
                try:
                    for line in fileinput.input(regtest_script, inplace=1, backup='.orig.refout'):
                        line = re.sub(r"^(dir_last\s*=\${dir_base})/.*$", r"\1/%s" % regtest_refdir, line)
                        sys.stdout.write(line)
                except IOError as err:
                    raise EasyBuildError("Failed to modify '%s': %s", regtest_script, err)

            else:
                self.log.info("No reference output found for regression test, just continuing without it...")

            # prefer using 4 cores, since some tests require/prefer square (n^2) numbers or powers of 2 (2^n)
            test_core_cnt = min(self.cfg.parallel, 4)
            if get_avail_core_count() < test_core_cnt:
                raise EasyBuildError("Cannot run MPI tests as not enough cores (< %s) are available", test_core_cnt)
            else:
                self.log.info("Using %s cores for the MPI tests" % test_core_cnt)

            if py_regtest_script:
                regtest_cmd = self.py_regtest_cmd(py_regtest_script, test_core_cnt)
                self.log.debug("Command to run regression test: %s", regtest_cmd)
            else:
                # configure regression test
                cfg_txt = '\n'.join([
                    'FORT_C_NAME="%(f90)s"',
                    'dir_base=%(base)s',
                    'cp2k_version=%(cp2k_version)s',
                    'dir_triplet=%(triplet)s',
                    'export ARCH=${dir_triplet}',
                    'cp2k_dir=%(cp2k_dir)s',
                    'leakcheck="YES"',
                    'maxtasks=%(maxtasks)s',
                    'cp2k_run_prefix="%(mpicmd_prefix)s"',
                ]) % {
                    'f90': os.getenv('F90'),
                    'base': os.path.dirname(os.path.normpath(self.cfg['start_dir'])),
                    'cp2k_version': self.cfg['type'],
                    'triplet': self.typearch,
                    'cp2k_dir': os.path.basename(os.path.normpath(self.cfg['start_dir'])),
                    'maxtasks': (self.cfg['maxtasks'] if self.cfg['maxtasks'] is not None
                                 else MAKEFILE_REGTEST_MAXTASKS),
                    'mpicmd_prefix': self.toolchain.mpi_cmd_for('', test_core_cnt),
                }

                write_file(cfg_fn, cfg_txt)
                self.log.debug("Contents of %s: %s" % (cfg_fn, cfg_txt))
                regtest_cmd = ' '.join(regtest_cmd)

            # run regression test
            regtest = run_shell_cmd(regtest_cmd, fail_on_error=False)

            if regtest.exit_code == 0:
                self.log.info("Regression test output:\n%s" % regtest.output)
            elif py_regtest_script and re.search(r"number\s+of\s+failed\s+tests\s+[0-9]+", regtest.output, re.I):
                # the Python driver exits with a non-zero code if any test did not pass (including tests that
                # are not applicable), the summary is evaluated below to decide if that is an error
                self.log.info("Regression test exited with code %s, output:\n%s", regtest.exit_code, regtest.output)
            else:
                raise EasyBuildError("Regression test failed (non-zero exit code): %s", regtest.output)

            # pattern to search for regression test summary
            re_pattern = r"number\s+of\s+%s\s+tests\s+(?P<cnt>[0-9]+)"

            # find total number of tests
            regexp = re.compile(re_pattern % "", re.M | re.I)
            res = regexp.search(regtest.output)
            tot_cnt = None
            if res:
                tot_cnt = int(res.group('cnt'))
            else:
                raise EasyBuildError("Finding total number of tests in regression test summary failed")

            # function to report on regtest results
            def test_report(test_result):
                """Report on tests with given result."""

                postmsg = ''

                test_result = test_result.upper()
                regexp = re.compile(re_pattern % test_result, re.M | re.I)

                cnt = None
                res = regexp.search(regtest.output)
                if not res:
                    raise EasyBuildError("Finding number of %s tests in regression test summary failed",
                                         test_result.lower())
                else:
                    cnt = int(res.group('cnt'))

                logmsg = "Regression test reported %s / %s %s tests"
                logmsg_values = (cnt, tot_cnt, test_result.lower())

                # A numerically wrong result is an installation failure at any count.
                if test_result in ("FAILED", "WRONG") and cnt > 0:
                    if self.cfg['ignore_regtest_fails']:
                        self.log.warning(logmsg, *logmsg_values)
                        self.log.info("Ignoring failures in regression test, as requested.")
                    else:
                        raise EasyBuildError(logmsg, *logmsg_values)
                elif test_result == "CORRECT" or cnt == 0:
                    self.log.info(logmsg, *logmsg_values)
                else:
                    self.log.warning(logmsg, *logmsg_values)

                return postmsg

            # number of failed/wrong tests, will report error if count is positive
            self.postmsg += test_report("FAILED")
            self.postmsg += test_report("WRONG")

            # there are no more 'new' tests from CP2K 8.1 onwards
            if LooseVersion(self.version) < LooseVersion('8.0'):
                # number of new tests, will be high if a non-suitable regtest reference was used
                # will report error if count is positive (is that what we want?)
                self.postmsg += test_report("NEW")

            # number of correct tests: just report
            test_report("CORRECT")

    def install_step_cmake(self):
        """Install CP2K with CMake, and add what CP2K's regression test driver needs to run on the installation"""

        super().install_step()

        share_dir = os.path.join(self.installdir, 'share', 'cp2k')
        for subdir in ['tests', os.path.join('src', 'grid', 'sample_tasks')]:
            copy_dir(os.path.join(self.cfg['start_dir'], subdir), os.path.join(share_dir, subdir))

    def install_step(self):
        """Install built CP2K
        - copy from exe to bin
        - copy data dir (if exists)
        - copy tests
        """

        if self.cmake_build:
            return self.install_step_cmake()

        # copy executables
        exedir = os.path.join(self.cfg['start_dir'], 'exe', self.typearch)
        targetdir = os.path.join(self.installdir, 'bin')
        copy_dir(exedir, targetdir)

        # copy libraries and include files, not sure what is strictly required so we take everything
        if self.cfg['library']:
            libdir = os.path.join(self.cfg['start_dir'], 'lib', self.typearch, self.cfg['type'])
            targetdir = os.path.join(self.installdir, 'lib')
            copy_dir(libdir, targetdir)
            # Also need to populate the include directory
            targetdir = os.path.join(self.installdir, 'include')
            libcp2k_header = os.path.join(self.cfg['start_dir'], 'src', 'start', 'libcp2k.h')
            target_header = os.path.join(targetdir, os.path.basename(libcp2k_header))
            copy_file(libcp2k_header, target_header)
            # include all .mod files for fortran users (don't know the exact list so take everything)
            mod_path = os.path.join(self.cfg['start_dir'], 'obj', self.typearch, self.cfg['type'])
            for mod_file in glob.glob(os.path.join(mod_path, '*.mod')):
                target_mod = os.path.join(targetdir, os.path.basename(mod_file))
                copy_file(mod_file, target_mod)

        # copy data dir
        datadir = os.path.join(self.cfg['start_dir'], 'data')
        targetdir = os.path.join(self.installdir, 'data')
        if os.path.exists(targetdir):
            self.log.info("Won't copy data dir. Destination directory %s already exists" % targetdir)
        elif os.path.exists(datadir):
            copy_dir(datadir, targetdir)
        else:
            self.log.info("Won't copy data dir. Source directory %s does not exist" % datadir)

        # copy tests
        srctests = os.path.join(self.cfg['start_dir'], 'tests')
        targetdir = os.path.join(self.installdir, 'tests')
        if os.path.exists(targetdir):
            self.log.info("Won't copy tests. Destination directory %s already exists" % targetdir)
        else:
            copy_dir(srctests, targetdir)

        # copy regression test results
        if self.cfg['runtest']:
            try:
                testdir = os.path.dirname(os.path.normpath(self.cfg['start_dir']))
                for d in os.listdir(testdir):
                    if d.startswith('TEST-%s-%s' % (self.typearch, self.cfg['type'])):
                        path = os.path.join(testdir, d)
                        target = os.path.join(self.installdir, d)
                        copy_dir(path, target)
                        self.log.info("Regression test results dir %s copied to %s" % (d, self.installdir))
                        break
            except (OSError, IOError) as err:
                raise EasyBuildError("Failed to copy regression test results dir: %s", err)

    def sanity_check_step_cmake(self):
        """Custom sanity check for CP2K built with CMake"""

        features = self.cmake_features()
        ext = features['ext']
        # same precedence as in CMakeMake: CMAKE_INSTALL_LIBDIR in configopts, then install_libdir
        libdir = (get_cmake_option(self.cfg['configopts'], 'CMAKE_INSTALL_LIBDIR') or self.cfg['install_libdir']
                  or 'lib')
        shared_libs = get_cmake_option(self.cfg['configopts'], 'BUILD_SHARED_LIBS')
        if shared_libs is None:
            shared_libs = self.cfg['build_shared_libs'] is not False
        else:
            shared_libs = is_cmake_true(shared_libs)
        lib_ext = get_shared_lib_ext() if shared_libs else 'a'

        share_dir = os.path.join('share', 'cp2k')
        custom_paths = {
            'files': [os.path.join('bin', 'cp2k.%s' % ext), os.path.join(libdir, 'libcp2k.%s' % lib_ext),
                      os.path.join(libdir, 'cmake', 'cp2k', 'cp2kConfig.cmake'),
                      os.path.join('include', 'cp2k', 'libcp2k.h'),
                      os.path.join(share_dir, 'tests', 'do_regtest.py')],
            'dirs': [os.path.join(share_dir, 'data'), os.path.join(share_dir, 'src', 'grid', 'sample_tasks')],
        }
        # CP2K installs cp2k.popt and cp2k.sopt as symlinks to cp2k.psmp and cp2k.ssmp
        alias = {'psmp': 'popt', 'ssmp': 'sopt'}.get(ext)
        if alias:
            custom_paths['files'].append(os.path.join('bin', 'cp2k.%s' % alias))

        flags = features['flags']
        if features['enabled']['CP2K_USE_TBLITE'] and LooseVersion(self.version) >= TBLITE_S_DFTD3_VERSION:
            flags = flags + ['s_dftd3']
        custom_commands = ["cp2k.%s --version | grep 'cp2kflags:' | grep -w %s" % (ext, flag) for flag in flags]

        extra_modules = None
        if not self.cfg['sanity_regtest']:
            self.log.info("Not running CP2K's regression tests in the sanity check, as requested")
        elif features['enabled']['CP2K_USE_MPI'] and not build_option('mpi_tests'):
            self.log.info("Not running CP2K's regression tests in the sanity check since MPI testing is disabled")
        else:
            # The driver copies tests/ with its permissions and writes into the copy, so it fails on a read-only
            # installation: run it from a writable copy of tests/ and src/grid/sample_tasks, in the
            # sanity check's temporary working directory
            install_share_dir = os.path.join(self.installdir, share_dir)
            opts = ['--smoketest', '--cp2kdatadir %s' % os.path.join(install_share_dir, 'data')]
            custom_commands.append(' && '.join([
                'cp -R %s %s .' % (os.path.join(install_share_dir, 'tests'), os.path.join(install_share_dir, 'src')),
                'chmod -R u+w tests src',
                self.regtest_cmd('.', os.path.join(self.installdir, 'bin'), ext, opts),
            ]))
            # the driver needs Python, which may only be a build dependency
            extra_modules = [dep['short_mod_name'] for dep in self.cfg.dependencies(build_only=True)
                             if dep['name'] == 'Python'] or None

        return super().sanity_check_step(custom_paths=custom_paths, custom_commands=custom_commands,
                                         extra_modules=extra_modules)

    def sanity_check_step(self):
        """Custom sanity check for CP2K"""

        if self.cmake_build:
            return self.sanity_check_step_cmake()

        cp2k_type = self.cfg['type']
        custom_paths = {
            'files': ["bin/%s.%s" % (x, cp2k_type) for x in ["cp2k", "cp2k_shell"]],
            'dirs': ["tests"]
        }
        if self.cfg['library']:
            custom_paths['files'].append(os.path.join('lib', 'libcp2k.a'))
            custom_paths['files'].append(os.path.join('include', 'libcp2k.h'))
            custom_paths['files'].append(os.path.join('include', 'libcp2k.mod'))
        super().sanity_check_step(custom_paths=custom_paths)

    def make_module_extra(self):
        """Set up a CP2K_DATA_DIR environment variable to find CP2K provided basis sets"""

        txt = super().make_module_extra()

        # also define $CP2K_DATA_DIR in module,
        # even though CP2K was already configured to pick up 'data' from install dir
        # this could be useful for users to access the 'data' dir in a documented way (and it doesn't hurt)
        if self.cmake_build:
            datadir = os.path.join(self.installdir, 'share', 'cp2k', 'data')
        else:
            datadir = os.path.join(self.installdir, 'data')
        if os.path.exists(datadir):
            txt += self.module_generator.set_environment('CP2K_DATA_DIR', datadir)

        return txt
