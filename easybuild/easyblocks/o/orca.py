##
# Copyright 2021-2026 Vrije Universiteit Brussel
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
EasyBuild support for ORCA, implemented as an easyblock

@author: Alex Domingo (Vrije Universiteit Brussel)
"""
import glob
import os
import re
import shlex
import struct
import tempfile

from easybuild.tools import LooseVersion
from easybuild.easyblocks.generic.makecp import MakeCp
from easybuild.easyblocks.generic.packedbinary import PackedBinary
from easybuild.framework.easyconfig import CUSTOM
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.config import build_option
from easybuild.tools.filetools import symlink, which, write_file
from easybuild.tools.run import run_shell_cmd
from easybuild.tools.systemtools import X86_64, get_cpu_architecture


ORCA_PT_LOAD = 1
ORCA_ET_EXEC = 2
ORCA_MIN_LOAD_VADDR = 0x10000
ORCA_MAX_LOAD_ALIGNMENT = 0x200000


def orca_elf_load_info(path):
    """Return the ELF type and (offset, virtual address, alignment) of its load segments."""
    try:
        with open(path, 'rb') as stream:
            ident = stream.read(16)
            if (len(ident) != 16 or ident[:4] != b'\x7fELF' or ident[4] not in (1, 2)
                    or ident[5] not in (1, 2)):
                raise EasyBuildError("Unsupported ELF header in %s", path)
            endian = '<' if ident[5] == 1 else '>'
            if ident[4] == 2:
                ehdr, phdr, offset_idx, vaddr_idx = endian + 'HHIQQQIHHH', endian + 'IIQQQQQQ', 2, 3
            else:
                ehdr, phdr, offset_idx, vaddr_idx = endian + 'HHIIIIIHHH', endian + 'IIIIIIII', 1, 2
            header = struct.unpack(ehdr, stream.read(struct.calcsize(ehdr)))
            phoff, phentsize, phnum = header[4], header[8], header[9]
            if phentsize < struct.calcsize(phdr):
                raise EasyBuildError("Unsupported ELF program header size in %s", path)
            segments = []
            for index in range(phnum):
                stream.seek(phoff + index * phentsize)
                entry = struct.unpack(phdr, stream.read(struct.calcsize(phdr)))
                if entry[0] == ORCA_PT_LOAD:
                    segments.append((entry[offset_idx], entry[vaddr_idx], entry[7]))
            return header[0], segments
    except (OSError, struct.error) as err:
        raise EasyBuildError("Failed to read ELF load segments in %s: %s", path, err)


def orca_permanent_rpaths(rpath, excluded_roots):
    """Keep unique absolute toolchain paths outside the build and temporary directories."""
    roots = [os.path.realpath(root) for root in excluded_roots if root]
    entries = []
    for entry in rpath.split(':'):
        if not os.path.isabs(entry):
            continue
        resolved = os.path.realpath(entry)
        if any(os.path.commonpath([resolved, root]) == root for root in roots):
            continue
        if entry not in entries:
            entries.append(entry)
    return entries


class EB_ORCA(PackedBinary, MakeCp):
    """
    ORCA installation files are extracted and placed in standard locations using 'files_to_copy' from MakeCp.
    Sanity checks on files are automatically generated based on the contents of 'files_to_copy' by gathering
    the target files in the build directory and checking their presence in the installation directory.
    Sanity checks also include a quick test calculating the HF energy of a water molecule.
    """

    @staticmethod
    def extra_options(extra_vars=None):
        """Extra easyconfig parameters for ORCA."""
        extra_vars = MakeCp.extra_options()
        extra_vars.update(PackedBinary.extra_options())

        # files_to_copy is not mandatory here, since we set it by default in install_step
        extra_vars['files_to_copy'][2] = CUSTOM

        return extra_vars

    def __init__(self, *args, **kwargs):
        """Init and validate easyconfig parameters and system architecture"""
        super().__init__(*args, **kwargs)

        # If user overwrites 'files_to_copy', custom 'sanity_check_paths' must be present
        if self.cfg['files_to_copy'] and not self.cfg['sanity_check_paths']:
            raise EasyBuildError("Found 'files_to_copy' option in easyconfig without 'sanity_check_paths'")

        # Add orcaarch template for supported architectures
        myarch = get_cpu_architecture()
        if myarch == X86_64:
            orcaarch = 'x86-64'
        else:
            raise EasyBuildError("Architecture %s is not supported by ORCA on EasyBuild", myarch)

        self.cfg.template_values['orcaarch'] = orcaarch
        self.cfg.generate_template_values()

    def install_step(self):
        """Install ORCA with MakeCp easyblock"""

        if not self.cfg['files_to_copy']:
            # Put installation files in standard locations
            files_to_copy = [
                (['auto*', 'orca*', 'otool*'], 'bin'),
                (['*.pdf'], 'share'),
            ]

            # Version 6 extra files
            if LooseVersion(self.version) >= LooseVersion('6.1.0'):
                files_to_copy.extend(['datasets', 'lib', (['openCOSMORS'], 'bin')])
            elif LooseVersion(self.version) >= LooseVersion('6.0.0'):
                files_to_copy.extend(['datasets', 'lib', (['CompoundScripts', 'openCOSMORS'], 'bin')])
            # Version 5 extra files
            elif LooseVersion(self.version) >= LooseVersion('5.0.0'):
                compoundmethods = (['ORCACompoundMethods'], 'bin')
                files_to_copy.append(compoundmethods)

            # Shared builds have additional libraries
            libs_to_copy = (['liborca*'], 'lib')
            if all(glob.glob(p) for p in libs_to_copy[0]):
                files_to_copy.append(libs_to_copy)

            self.cfg['files_to_copy'] = files_to_copy

        MakeCp.install_step(self)

        # Shared executables use $ORIGIN/lib for the vendor runtime libraries.
        libdir = os.path.join(self.installdir, 'lib')
        binlib = os.path.join(self.installdir, 'bin', 'lib')
        if (os.path.isdir(os.path.dirname(binlib)) and glob.glob(os.path.join(libdir, 'liborca*'))
                and not os.path.lexists(binlib)):
            symlink(os.path.join(os.pardir, 'lib'), binlib, use_abspath_source=False)

    def _mpi_consumers(self):
        """Find installed ELF files with a direct dependency on an MPI runtime library."""
        files = set()
        for directory, subdirs, filenames in os.walk(self.installdir, followlinks=False):
            if directory == self.installdir:
                subdirs[:] = [name for name in subdirs if name in ('bin', 'lib', 'lib64')]
            for filename in filenames:
                path = os.path.join(directory, filename)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                with open(path, 'rb') as stream:
                    if stream.read(4) != b'\x7fELF':
                        continue
                dynamic = run_shell_cmd("readelf -d %s" % shlex.quote(path), hidden=True).output
                if re.search(r'\(NEEDED\).*\[libmpi[^]]*\]', dynamic):
                    files.add(path)
        return sorted(files)

    def _toolchain_mpi_rpath(self):
        """Link an MPI probe with the active wrappers without executing it."""
        compiler = os.environ.get('MPICC')
        if not compiler:
            raise EasyBuildError("MPICC is required to determine the active MPI runtime RPATH")
        source = os.path.join(self.builddir, 'eb_orca_mpi_rpath.c')
        executable = os.path.join(self.builddir, 'eb_orca_mpi_rpath')
        write_file(source, '#include <mpi.h>\n'
                   'int main(int argc, char **argv) {\n'
                   '    MPI_Init(&argc, &argv);\n'
                   '    return MPI_Finalize();\n'
                   '}\n')
        command = ' '.join([compiler, os.environ.get('CPPFLAGS', ''), os.environ.get('CFLAGS', ''),
                            shlex.quote(source), '-o', shlex.quote(executable), os.environ.get('LDFLAGS', '')])
        run_shell_cmd(command, work_dir=self.builddir, hidden=True)
        rpath = run_shell_cmd("patchelf --print-rpath %s" % shlex.quote(executable), hidden=True).output.strip()
        excluded = [self.builddir, self.cfg.get('start_dir'), os.environ.get('TMPDIR'),
                    tempfile.gettempdir(), '/tmp', '/var/tmp']
        entries = orca_permanent_rpaths(rpath, excluded)
        if not entries:
            raise EasyBuildError("The active MPI compiler did not provide permanent RPATH entries")
        return entries

    def _append_mpi_rpath(self, path, additions):
        """Preserve vendor search paths and ELF load safety while adding the selected MPI paths."""
        quoted_path = shlex.quote(path)
        current = run_shell_cmd("patchelf --print-rpath %s" % quoted_path, hidden=True).output.strip()
        original = [entry for entry in current.split(':') if entry]
        entries = list(original)
        for addition in additions:
            if addition not in entries:
                entries.append(addition)
        dynamic = run_shell_cmd("readelf -d %s" % quoted_path, hidden=True).output
        if entries == original and '(RPATH)' in dynamic:
            return

        elf_type, segments = orca_elf_load_info(path)
        alignments = [align for _, _, align in segments]
        if not alignments or any(align > ORCA_MAX_LOAD_ALIGNMENT or align & (align - 1) for align in alignments):
            raise EasyBuildError("Unsupported PT_LOAD alignment in %s: %s", path, alignments)
        alignment = max(alignments + [1])
        fixed_address = elf_type == ORCA_ET_EXEC
        # A large page size can place a fixed-address image's grown string table on page zero.
        page_size = '--page-size %d ' % alignment if alignment > 0x1000 and not fixed_address else ''
        run_shell_cmd("patchelf %s--force-rpath --set-rpath %s %s" %
                      (page_size, shlex.quote(':'.join(entries)), quoted_path), hidden=True)
        recorded = run_shell_cmd("patchelf --print-rpath %s" % quoted_path, hidden=True).output.strip()
        if recorded.split(':') != entries:
            raise EasyBuildError("RPATH entries were not preserved in %s: %s", path, recorded)
        dynamic = run_shell_cmd("readelf -d %s" % quoted_path, hidden=True).output
        if '(RPATH)' not in dynamic:
            raise EasyBuildError("DT_RPATH was not recorded for %s", path)
        patched_type, patched_segments = orca_elf_load_info(path)
        if patched_type != elf_type or not patched_segments:
            raise EasyBuildError("ELF load structure was not preserved in %s", path)
        for offset, vaddr, align in patched_segments:
            if align > 1 and (vaddr - offset) % align:
                raise EasyBuildError("PT_LOAD alignment was not preserved in %s", path)
            if fixed_address and vaddr < ORCA_MIN_LOAD_VADDR:
                raise EasyBuildError("RPATH rewrite placed a load segment below %#x in %s", ORCA_MIN_LOAD_VADDR, path)
            if not fixed_address and align < min(alignments):
                raise EasyBuildError("PT_LOAD alignment was not preserved in %s", path)

    def post_processing_step(self):
        """Give precompiled MPI consumers the permanent RPATH of the selected toolchain."""
        super().post_processing_step()
        if not build_option('rpath') and 'LD_LIBRARY_PATH' not in (build_option('filter_env_vars') or []):
            return
        consumers = self._mpi_consumers()
        if not consumers:
            return
        if which('patchelf') is None:
            raise EasyBuildError("patchelf is required to record the active ORCA MPI runtime paths")
        additions = self._toolchain_mpi_rpath()
        for path in consumers:
            self._append_mpi_rpath(path, additions)

    def sanity_check_step(self):
        """Custom sanity check for ORCA"""
        custom_paths = None

        if not self.cfg['sanity_check_paths']:
            custom_paths = {'files': [], 'dirs': []}

            if self.cfg['files_to_copy']:
                # Convert 'files_to_copy' to list of files in build directory
                for spec in self.cfg['files_to_copy']:
                    if isinstance(spec, tuple):
                        file_pattern = spec[0]
                        dest_dir = spec[1]
                    elif isinstance(spec, str):
                        file_pattern = spec
                        dest_dir = ''
                    else:
                        raise EasyBuildError(
                            "Found neither string nor tuple as file to copy: '%s' (type %s)", spec, type(spec)
                        )

                    if isinstance(file_pattern, str):
                        file_pattern = [file_pattern]

                    source_files = []
                    for pattern in file_pattern:
                        source_files.extend(glob.glob(pattern))

                    # Add files to custom sanity checks
                    for source in source_files:
                        if os.path.isfile(source):
                            custom_paths['files'].append(os.path.join(dest_dir, source))
                        else:
                            custom_paths['dirs'].append(os.path.join(dest_dir, source))
            else:
                # Minimal check of files (needed by --module-only)
                custom_paths['files'] = ['bin/orca']

        # Simple test: HF energy of water molecule
        test_input_content = """
!HF DEF2-SVP
%%PAL NPROCS %(nprocs)s END
* xyz 0 1
O   0.0000   0.0000   0.0626
H  -0.7920   0.0000  -0.4973
H   0.7920   0.0000  -0.4973
*
"""
        nprocs = self.cfg.parallel
        test_input_content = test_input_content % {'nprocs': nprocs}
        test_input_path = os.path.join(self.builddir, 'eb_test_hf_water.inp')
        write_file(test_input_path, test_input_content)

        # Reference total energy
        test_output_energy = '-75.95934031'
        test_output_regex = 'FINAL SINGLE POINT ENERGY[ \t]*%s' % test_output_energy

        # Instruct openmpi to treat hardware threads as slot
        test_ompi_env = 'env OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1'

        # ORCA has to be executed using its full path to run in parallel
        if os.path.isdir(os.path.join(self.installdir, 'bin')):
            orca_bin = '$EBROOTORCA/bin/orca'
        else:
            orca_bin = '$(which orca)'

        test_orca_cmd = "%s %s %s" % (test_ompi_env, orca_bin, test_input_path)

        custom_commands = [
            # Execute test in ORCA
            test_orca_cmd,
            # Repeat test and check total energy
            "%s | grep -c '%s'" % (test_orca_cmd, test_output_regex),
        ]

        super().sanity_check_step(custom_paths=custom_paths, custom_commands=custom_commands)
