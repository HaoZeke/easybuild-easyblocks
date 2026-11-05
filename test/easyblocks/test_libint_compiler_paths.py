import shlex
import unittest

from easybuild.easyblocks.l.libint import libint_compiler_configure_cmd


class LibintCompilerPaths(unittest.TestCase):
    def test_managed_boost_root_is_an_explicit_configure_argument(self):
        command = libint_compiler_configure_cmd('', '--enable-eri=1', '/managed/boost')
        self.assertIn('--with-boost=/managed/boost', shlex.split(command))

    def test_managed_root_with_spaces_stays_one_argument(self):
        command = libint_compiler_configure_cmd('', '--enable-eri=1', '/managed software/boost')
        self.assertIn('--with-boost=/managed software/boost', shlex.split(command))

    def test_explicit_configure_root_keeps_its_precedence(self):
        command = libint_compiler_configure_cmd('', '--with-boost=/chosen/boost', '/managed/boost')
        args = shlex.split(command)
        self.assertEqual([arg for arg in args if arg.startswith('--with-boost=')], ['--with-boost=/chosen/boost'])

    def test_split_explicit_configure_root_keeps_its_precedence(self):
        command = libint_compiler_configure_cmd('', '--with-boost /chosen/boost', '/managed/boost')
        self.assertEqual(shlex.split(command), ['./configure', '--with-boost', '/chosen/boost'])


if __name__ == '__main__':
    unittest.main()
