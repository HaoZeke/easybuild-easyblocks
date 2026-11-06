from pathlib import Path
import tempfile
import unittest

from easybuild.easyblocks.p.pytorch import configure_pytorch_xml_reports
from easybuild.tools.build_log import EasyBuildError


class PytorchXmlReportConfiguration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'common_utils.py'

    def test_pristine_report_conditions_are_enabled_and_repeatable(self):
        for condition in ['IS_CI', 'IS_IN_CI']:
            with self.subTest(condition=condition):
                self.source.write_text(
                    'sentinel = 17\n'
                    'default=_get_test_report_path() if ' + condition + ' else None\n'
                    'assert actual == expected\n')
                expected = (
                    'sentinel = 17\n'
                    'default=_get_test_report_path() if os.getenv("EB_TEST_REPORTS") else None\n'
                    'assert actual == expected\n')
                for _ in range(3):
                    configure_pytorch_xml_reports(str(self.source), 'EB_TEST_REPORTS')
                    self.assertEqual(self.source.read_text(), expected)
                self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_configured_condition_keeps_the_same_variable(self):
        text = 'default=_get_test_report_path() if os.getenv("EB_TEST_REPORTS") else None\n'
        self.source.write_text(text)
        configure_pytorch_xml_reports(str(self.source), 'EB_TEST_REPORTS')
        self.assertEqual(self.source.read_text(), text)

    def test_unrecognized_condition_is_refused(self):
        text = 'default=_get_test_report_path() if OTHER_CONDITION else None\n'
        self.source.write_text(text)
        with self.assertRaises(EasyBuildError):
            configure_pytorch_xml_reports(str(self.source), 'EB_TEST_REPORTS')
        self.assertEqual(self.source.read_text(), text)

    def test_other_report_variable_is_refused(self):
        text = 'default=_get_test_report_path() if os.getenv("OTHER_REPORTS") else None\n'
        self.source.write_text(text)
        with self.assertRaises(EasyBuildError):
            configure_pytorch_xml_reports(str(self.source), 'EB_TEST_REPORTS')
        self.assertEqual(self.source.read_text(), text)


if __name__ == '__main__':
    unittest.main()
