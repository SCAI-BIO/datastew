import importlib
import importlib.metadata
import re
import unittest
from unittest import mock

import datastew
import datastew._version


class TestVersion(unittest.TestCase):
    def test_canonical_version(self):
        version = datastew.__version__
        pep440_pattern = (
            r"^([1-9][0-9]*!)?"
            r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
            r"((a|b|rc)(0|[1-9][0-9]*))?"
            r"(\.post(0|[1-9][0-9]*))?"
            r"(\.dev(0|[1-9][0-9]*))?$"
        )
        is_canonical = bool(re.match(pep440_pattern, version))
        self.assertTrue(is_canonical, f"Version '{version}' does not match PEP 440 canonical format.")

    def test_version_comes_from_installed_metadata(self):
        installed = importlib.metadata.version("datastew")
        self.assertEqual(datastew.__version__, installed)

    def test_version_falls_back_when_package_is_not_installed(self):
        error = importlib.metadata.PackageNotFoundError("datastew")
        with mock.patch("importlib.metadata.version", side_effect=error):
            module = importlib.reload(datastew._version)

        try:
            self.assertEqual(module.__version__, "0.0.0")
        finally:
            importlib.reload(datastew._version)


if __name__ == "__main__":
    unittest.main()
