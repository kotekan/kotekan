import sys
import pytest

print("file", __file__)
print("path", sys.path)

import kotekan


def test_file():

    print("kotekan file", kotekan.__file__)
    assert len(kotekan.__file__) > 0
