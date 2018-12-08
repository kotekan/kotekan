import pytest
import os.path
import time
from subprocess import Popen
from subprocess import call
import signal

producer_path = "./boost/dataset-broker/dataset_broker_producer"
producer2_path = "./boost/dataset-broker/dataset_broker_producer2"
consumer_path = "./boost/dataset-broker/dataset_broker_consumer"
broker_path = "../build/ext/src/ch_acq/dataset_broker.py"

def test_produce_consume():
    if not os.path.isfile(producer_path) or \
      not os.path.isfile(consumer_path) or \
      not os.path.isfile(producer2_path) or \
      not os.path.isfile(broker_path):
        print("Deactivated! Build with -DBOOST_TESTS=ON to activate this test")
        print("and make sure the dataset_broker is at", broker_path)
        return

    broker = Popen([broker_path])
    time.sleep(2)

    try:
      assert call([producer_path]) == 0
      assert call([producer2_path]) == 0
      assert call([consumer_path]) == 0
    finally:
      pid = broker.pid
      os.kill(pid, signal.SIGINT)
      broker.terminate()