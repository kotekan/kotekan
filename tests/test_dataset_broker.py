import pytest
import os.path

from subprocess import call

producer_path = "../build/tests/boost/dataset_broker_producer"
producer2_path = "../build/tests/boost/dataset_broker_producer2"
consumer_path = "../build/tests/boost/dataset_broker_consumer"


@pytest.mark.serial
def test_produce_consume(comet_broker_port):
    if (
        not os.path.isfile(producer_path)
        or not os.path.isfile(consumer_path)
        or not os.path.isfile(producer2_path)
    ):
        pytest.skip("Build with -DBOOST_TESTS=ON to activate this test")

    assert call([producer_path, "--", comet_broker_port]) == 0
    assert call([producer2_path, "--", comet_broker_port]) == 0
    assert call([consumer_path, "--", comet_broker_port]) == 0
