"""Test count-geometry validation with a host C++ compiler."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def geometry_guard(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip(
            "A host C++ compiler is required for the shared geometry guard test"
        )
    root = Path(__file__).resolve().parents[1]
    work = tmp_path_factory.mktemp("n2-count-geometry")
    source = work / "guard.cpp"
    source.write_text(
        r"""#include "n2CountGeometry.hpp"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc != 3) return 3;
    try {
        const auto groups = kotekan::n2_count_station_groups(std::stoll(argv[1]), std::stoll(argv[2]));
        std::cout << groups << "\n";
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
"""
    )
    binary = work / "guard"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "lib/utils"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "polarizations,dishes,groups",
    [(2, 64, 16), (2, 512, 128), (1, 128, 16), (4, 256, 128)],
)
def test_supported_count_geometry(geometry_guard, polarizations, dishes, groups):
    result = subprocess.run(
        [str(geometry_guard), str(polarizations), str(dishes)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == groups


@pytest.mark.parametrize(
    "polarizations,dishes,diagnostic",
    [
        (2, 32, "only supports Sds=16 or Sds=128"),
        (2, 128, "only supports Sds=16 or Sds=128"),
        (2, 256, "only supports Sds=16 or Sds=128"),
        (2, 1024, "only supports Sds=16 or Sds=128"),
        (0, 64, "positive"),
        (-2, 64, "positive"),
        (2, 0, "positive"),
        (2, -64, "positive"),
        (2, 63, "divisible by eight"),
        (2, 65, "divisible by eight"),
        (2 ** 63 - 1, 16, "geometry overflow"),
        (9, 2 ** 63 - 8, "geometry overflow"),
    ],
)
def test_invalid_count_geometry(geometry_guard, polarizations, dishes, diagnostic):
    result = subprocess.run(
        [str(geometry_guard), str(polarizations), str(dishes)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert result.stdout == ""
