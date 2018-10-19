import pytest
import numpy as np
import os.path
import time
from subprocess import Popen
from subprocess import call
import signal

import kotekan_runner

producer_path = "./boost/dataset-broker/dataset_broker_producer"
producer2_path = "./boost/dataset-broker/dataset_broker_producer2"
consumer_path = "./boost/dataset-broker/dataset_broker_consumer"
broker_path = "../build/ext/src/ch_acq/dataset_broker.py"

def test_produce_consume():
    if not os.path.isfile(producer_path) or \
      not os.path.isfile(consumer_path) or \
      not os.path.isfile(producer2_path) or \
      not os.path.isfile(broker_path):
        print "Deactivated! Build with -DBOOST_TESTS=ON to activate this test."
        return

    broker = Popen([broker_path])
    time.sleep(1.5)

    try:
      assert call([producer_path]) == 0
      assert call([producer2_path]) == 0
      assert call([consumer_path]) == 0
    finally:
      pid = broker.pid
      os.kill(pid, signal.SIGINT)
      broker.terminate()

N_THREADS = 10

def test_produce_consume_async():
    broker = Popen([broker_path])
    time.sleep(1.5)

    p = list()
    p2 = list()
    c = list()

    try:
      for i in range(N_THREADS):
          p.append(Popen([producer_path]))
          p2.append(Popen([producer2_path]))
          c.append(Popen([consumer_path]))

      for i in range(N_THREADS):
          assert p[i].wait() == 0
          assert p2[i].wait() == 0
          assert c[i].wait() == 0
    finally:
      pid = broker.pid
      os.kill(pid, signal.SIGINT)
      broker.terminate()