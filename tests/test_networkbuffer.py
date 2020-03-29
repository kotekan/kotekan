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
    "dataset_manager": {"use_dataset_broker": False},
    "server_ip": "127.0.0.1",
}


@pytest.mark.serial
def test_send_receive(tmpdir_factory):

    # Run kotekan bufferRecv
    tmpdir = tmpdir_factory.mktemp("writer")
    write_buffer = runner.DumpVisBuffer(str(tmpdir))

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
    receiver._stages["bufferRecv_test"]["buf"] = receiver._stages["bufferRecv_test"][
        "out_buf"
    ]

    # Run kotekan bufferRecv in another thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_receiver = executor.submit(receiver.run)

        # Wait for it to start so the sender doesn't drop frames
        time.sleep(1)

        fakevis_buffer = runner.FakeVisBuffer(
            num_frames=params_kotekan["total_frames"],
            mode=params_kotekan["mode"],
            freq_ids=params_kotekan["freq_ids"],
            sleep_before=2,
            wait=False,
        )
        sender = runner.KotekanStageTester(
            "bufferSend", {}, fakevis_buffer, None, params_kotekan
        )

        # TODO: network buffer processes should use in_buf and out_buf to please the test framework
        sender._stages["bufferSend_test"]["buf"] = sender._stages["bufferSend_test"][
            "in_buf"
        ]

        # run kotekan bufferSend
        sender.run()

        # wait for kotekan bufferRecv to finish
        future_receiver.result(timeout=7)

    assert sender.return_code == 0
    assert receiver.return_code == 0
    vis_data = write_buffer.load()

    assert len(vis_data) == params_kotekan["total_frames"]
