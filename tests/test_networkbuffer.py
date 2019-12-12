# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import concurrent.futures
import pytest
import os
import re
import shutil
import signal
import tempfile
import time

from subprocess import Popen

from kotekan import runner


params_kotekan = {
    "num_elements": 5,
    "num_ev": 0,
    "total_frames": 8,
    "cadence": 10.0,
    "wait": True,
    "mode": "default",
    "buffer_depth": 16,
    "freq_ids": list(range(1)),
    "dataset_manager": {"use_dataset_broker": True},
    "server_ip": "127.0.0.1",
}


@pytest.mark.serial
def test_send_receive(tmpdir_factory):
    # Skip test if comet not installed
    broker_path = shutil.which("comet")
    if not broker_path:
        pytest.skip(
            "Make sure PYTHONPATH is set to where the comet dataset broker is installed."
        )

    # Start comet with random port and save it's output
    with tempfile.NamedTemporaryFile(mode="w") as f_out:
        broker = Popen(
            [broker_path, "-p", "0", "--recover", "False"], stdout=f_out, stderr=f_out
        )
        time.sleep(2)

        # Find port in the log
        regex = re.compile("Selected random port: ([0-9]+)$")
        log = open(f_out.name, "r").read().split("\n")
        port = None
        for line in log:
            print(line)
            match = regex.search(line)
            if match:
                port = match.group(1)
                print("Test found comet port in log: %s" % port)
                break
        if not match:
            print("Could not find comet port in logs.")
            exit(1)

        # Pass comet port to kotekan config
        params_kotekan["dataset_manager"]["ds_broker_port"] = port

        try:
            # Run kotekan bufferRecv
            tmpdir = tmpdir_factory.mktemp("writer")
            write_buffer = runner.VisWriterBuffer(str(tmpdir), "raw")

            # the plan is: wait 5s and then kill it
            rest_commands = [("wait", 5, None), ("get", "kill", None)]

            receiver = runner.KotekanStageTester(
                "bufferRecv",
                {},
                None,
                write_buffer,
                params_kotekan,
                rest_commands=rest_commands,
            )

            # TODO: network buffer processes should use in_buf and out_buf to please the test framework
            receiver._stages["bufferRecv_test"]["buf"] = receiver._stages[
                "bufferRecv_test"
            ]["out_buf"]

            # Run kotekan bufferRecv in another thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_receiver = executor.submit(receiver.run)

                # Wait for it to start so the sender doesn't drop frames
                time.sleep(1)

                fakevis_buffer = runner.FakeVisBuffer(
                    num_frames=params_kotekan["total_frames"],
                    mode=params_kotekan["mode"],
                    freq_ids=params_kotekan["freq_ids"],
                    wait=False,
                )
                sender = runner.KotekanStageTester(
                    "bufferSend", {}, fakevis_buffer, None, params_kotekan
                )

                # TODO: network buffer processes should use in_buf and out_buf to please the test framework
                sender._stages["bufferSend_test"]["buf"] = sender._stages[
                    "bufferSend_test"
                ]["in_buf"]

                # run kotekan bufferSend
                sender.run()

                # wait for kotekan bufferRecv to finish
                future_receiver.result(timeout=7)
        finally:
            # In any case, kill comet and print its log
            pid = broker.pid
            os.kill(pid, signal.SIGINT)
            broker.terminate()
            log = open(f_out.name, "r").read().split("\n")
            for line in log:
                print(line)

    assert sender.return_code == 0
    assert receiver.return_code == 0
    vis_data = write_buffer.load()
    assert len(vis_data.valid_frames) == params_kotekan["total_frames"]
