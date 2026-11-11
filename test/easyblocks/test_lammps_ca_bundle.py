"""
Additive controls for LAMMPS CURL_CA_BUNDLE host CA discovery and module pushenv text.
"""
import os
import unittest
from unittest import mock

from easybuild.easyblocks.l.lammps import (
    HOST_CA_BUNDLE_CANDIDATES,
    det_host_ca_bundle,
    ensure_lammps_curl_ca_bundle,
    lammps_curl_ca_bundle_module_statements,
)
from easybuild.easyblocks.openssl import GENERIC_SSL_CERTS_DIR


class _FakeModuleGenerator:
    """Minimal stand-in exposing SYNTAX like ModuleGeneratorLua/Tcl."""

    def __init__(self, syntax):
        self.SYNTAX = syntax
        self.log = mock.Mock()


class LammpsCaBundle(unittest.TestCase):
    """Focused CURL_CA_BUNDLE resolution and module statement controls."""

    def test_candidates_reuse_openssl_generic_certs_dir(self):
        """Debian/Ubuntu CAfile is under OpenSSL easyblock GENERIC_SSL_CERTS_DIR."""
        self.assertEqual(
            HOST_CA_BUNDLE_CANDIDATES[0],
            os.path.join(GENERIC_SSL_CERTS_DIR, 'ca-certificates.crt'),
        )

    def test_empty_env_uses_first_existing_host_path(self):
        """With no explicit setting, the first existing public host CAfile is chosen."""
        existing = '/etc/pki/tls/certs/ca-bundle.crt'
        isfile = lambda path: path == existing
        self.assertEqual(det_host_ca_bundle({}, isfile=isfile), existing)
        self.assertIsNone(det_host_ca_bundle({}, isfile=lambda path: False))

    def test_default_environ_prefers_real_host_bundle_when_present(self):
        """On this host, empty CURL/SSL env resolves to an existing public CAfile if any."""
        environ = {}
        resolved = det_host_ca_bundle(environ)
        if resolved is not None:
            self.assertTrue(os.path.isfile(resolved))
            self.assertIn(resolved, HOST_CA_BUNDLE_CANDIDATES)

    def test_valid_custom_curl_ca_bundle_is_preserved(self):
        """A non-empty CURL_CA_BUNDLE wins even when SSL_CERT_FILE and host paths exist."""
        custom = '/custom/valid-ca-bundle.pem'
        environ = {
            'CURL_CA_BUNDLE': custom,
            'SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
        }
        self.assertEqual(det_host_ca_bundle(environ, isfile=lambda path: True), custom)

    def test_invalid_custom_curl_ca_bundle_is_preserved(self):
        """Invalid explicit CURL_CA_BUNDLE must not fall back to SSL_CERT_FILE or host paths."""
        invalid = '/no/such/ca-bundle-%s.pem' % os.getpid()
        environ = {
            'CURL_CA_BUNDLE': invalid,
            'SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
        }
        self.assertFalse(os.path.isfile(invalid))
        self.assertEqual(det_host_ca_bundle(environ, isfile=lambda path: True), invalid)

    def test_ssl_cert_file_fallback_when_curl_unset(self):
        """SSL_CERT_FILE is used when CURL_CA_BUNDLE is unset, including invalid paths."""
        ssl = '/custom/ssl-cert-file.pem'
        environ = {'SSL_CERT_FILE': ssl}
        self.assertEqual(det_host_ca_bundle(environ, isfile=lambda path: True), ssl)
        environ_empty_curl = {'CURL_CA_BUNDLE': '', 'SSL_CERT_FILE': ssl}
        self.assertEqual(det_host_ca_bundle(environ_empty_curl, isfile=lambda path: True), ssl)

    def test_host_path_missing_returns_none(self):
        """When no explicit setting and no host CAfile exists, resolution is None."""
        self.assertIsNone(det_host_ca_bundle({}, isfile=lambda path: False))

    def test_ensure_sets_only_when_unset(self):
        """ensure_lammps_curl_ca_bundle writes CURL_CA_BUNDLE only when unset/empty."""
        host = '/etc/ssl/certs/ca-certificates.crt'
        with mock.patch('easybuild.easyblocks.l.lammps.det_host_ca_bundle', return_value=host):
            environ = {}
            self.assertEqual(ensure_lammps_curl_ca_bundle(environ), host)
            self.assertEqual(environ['CURL_CA_BUNDLE'], host)
            environ2 = {'CURL_CA_BUNDLE': '/keep/me.pem'}
            self.assertEqual(ensure_lammps_curl_ca_bundle(environ2), '/keep/me.pem')
            self.assertEqual(environ2['CURL_CA_BUNDLE'], '/keep/me.pem')

    def test_lua_module_statements_use_pushenv_and_runtime_candidates(self):
        """Lua module text discovers at load time and uses pushenv (not setenv)."""
        txt = lammps_curl_ca_bundle_module_statements(_FakeModuleGenerator('Lua'))
        self.assertIn('pushenv("CURL_CA_BUNDLE"', txt)
        self.assertNotIn('setenv("CURL_CA_BUNDLE"', txt)
        self.assertIn('isFile(', txt)
        self.assertIn('os.getenv("CURL_CA_BUNDLE")', txt)
        self.assertIn('os.getenv("SSL_CERT_FILE")', txt)
        for path in HOST_CA_BUNDLE_CANDIDATES:
            self.assertIn('"%s"' % path, txt)
        # Must not bake a resolved build-host absolute choice as the only value.
        self.assertNotRegex(txt, r'pushenv\("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates\.crt"\)\s*$')

    def test_tcl_module_statements_use_pushenv_and_runtime_candidates(self):
        """Tcl module text discovers at load time and uses pushenv."""
        txt = lammps_curl_ca_bundle_module_statements(_FakeModuleGenerator('Tcl'))
        self.assertIn('pushenv CURL_CA_BUNDLE', txt)
        self.assertNotIn('setenv CURL_CA_BUNDLE', txt)
        self.assertIn('file isfile', txt)
        self.assertIn('::env(CURL_CA_BUNDLE)', txt)
        self.assertIn('::env(SSL_CERT_FILE)', txt)
        for path in HOST_CA_BUNDLE_CANDIDATES:
            self.assertIn(path, txt)


def suite(loader):
    """Return all tests in this module."""
    return loader.loadTestsFromTestCase(LammpsCaBundle)


if __name__ == '__main__':
    unittest.main()
