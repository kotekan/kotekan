import os
from pathlib import Path
from subprocess import run

import pytest

build_dir = Path(__file__).resolve().parents[1] / os.environ.get(
    "KOTEKAN_BUILD_DIRNAME", "build"
)
producer_path = build_dir / "tests" / "boost" / "dataset_broker_producer"
producer2_path = build_dir / "tests" / "boost" / "dataset_broker_producer2"
consumer_path = build_dir / "tests" / "boost" / "dataset_broker_consumer"


@pytest.mark.serial
def test_produce_consume(comet_broker_port, tmp_path):
    helpers = (producer_path, producer2_path, consumer_path)
    if not all(path.is_file() for path in helpers):
        pytest.skip("Build with -DWITH_BOOST_TESTS=ON to activate this test")

    for helper in helpers:
        run(
            [helper, "--", str(comet_broker_port)], cwd=tmp_path, check=True, timeout=60
        )
