import re
import unittest

import datastew


class TestVersion(unittest.TestCase):
    def test_canonical_version(self):
        version = datastew.__version__
        pep440_pattern = r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
        is_canonical = bool(re.match(pep440_pattern, version))
        self.assertTrue(is_canonical, f"Version '{version}' does not match PEP 440 canonical format.")


if __name__ == "__main__":
    unittest.main()
