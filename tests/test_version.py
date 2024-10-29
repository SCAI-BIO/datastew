import re
from unittest import TestCase
import datastew


class Test(TestCase):
    def test_canonical_version(self):
        version = datastew.__version__
        return (
            re.match(
                r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
                version,
            )
            is not None
        )
