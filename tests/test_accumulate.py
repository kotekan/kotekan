# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner


accumulate_params = {
    "num_elements": 4,
    "num_ev": 0,
    "num_frb_total_beams": 12,
    "num_sub_freqs": 6,
    "samples_per_data_set": 32768,
    "int_frames": 64,
    "num_gpu_frames": 64,
    "total_frames": 257,  # One extra sample to ensure we actually get 256
    "block_size": 1,
    "freq": 777,
    "dataset_manager": {"use_dataset_broker": False},
}

gaussian_params = accumulate_params.copy()
gaussian_params.update(
    {"samples_per_data_set": 100, "num_gpu_frames": 100, "total_frames": 500000}
)

drop_frame_params = gaussian_params.copy()
drop_frame_params.update({"drop_probability": 0.3})

time_params = accumulate_params.copy()
time_params.update({"integration_time": 5.0})

pulsar_params = gaussian_params.copy()
pulsar_params.update(
    {
        "pattern": "pulsar",
        "gaussian_bgnd": False,
        "wait": True,
        "samples_per_data_set": 4000,  # ~10. ms frames
        "num_frames": 200,
        #'integration_time': 0.5,
        "num_gpu_frames": 40,
        "coeff": [0.0, 0.0],
        "dm": 0.0,
        "t_ref": 58000.0,
        "phase_ref": 0.0,
        #'rot_freq': 9.765625,  # period exactly 10 frames
        "rot_freq": 8.0,
        "pulse_width": 1e-9,
    }
)


@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="accumulate",
            freq=accumulate_params["freq"],
            num_frames=accumulate_params["total_frames"],
        ),
        dump_buffer,
        accumulate_params,
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def gaussian_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("gaussian")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="gaussian",
            freq=gaussian_params["freq"],
            num_frames=gaussian_params["total_frames"],
        ),
        dump_buffer,
        gaussian_params,
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def drop_frame_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("drop_frrame")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="gaussian",
            freq=drop_frame_params["freq"],
            num_frames=drop_frame_params["total_frames"],
            drop_probability=drop_frame_params["drop_probability"],
        ),
        dump_buffer,
        drop_frame_params,
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def lostsamples_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("lostsamples")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="lostsamples",
            freq=accumulate_params["freq"],
            num_frames=accumulate_params["total_frames"],
        ),
        dump_buffer,
        accumulate_params,
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module", params=[0, 1, 4])
def lostweights_data(tmpdir_factory, request):

    tmpdir = tmpdir_factory.mktemp("lostweights")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="lostweights",
            freq=accumulate_params["freq"],
            num_frames=accumulate_params["total_frames"],
            b=request.param,
        ),
        dump_buffer,
        accumulate_params,
    )

    test.run()

    yield (request.param, dump_buffer.load())


@pytest.fixture(scope="module")
def time_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("time")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "visAccumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="accumulate",
            freq=time_params["freq"],
            num_frames=time_params["total_frames"],
        ),
        dump_buffer,
        time_params,
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def pulsar_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("pulsar")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))
    dump_buffer_gated = runner.VisWriterBuffer(str(tmpdir), "raw")
    # Insert an extra buffer for gated stream
    dump_buffer.buffer_block.update(dump_buffer_gated.buffer_block)
    dump_buffer.stage_block.update(dump_buffer_gated.stage_block)

    acc_par = pulsar_params.copy()
    acc_par.update(
        {
            "gating": {"psr0": {"mode": "pulsar", "buf": dump_buffer_gated.name}},
            "updatable_config": {"psr0": "/updatable_config/psr0_config"},
        }
    )
    updatable_params = pulsar_params.copy()
    updatable_params.update(
        {
            "updatable_config": {
                "psr0_config": {
                    "kotekan_update_endpoint": "json",
                    "enabled": True,
                    "pulsar_name": "fakepsr",
                    "segment": 100.0,
                    "coeff": [pulsar_params["coeff"]],
                    "dm": pulsar_params["dm"],
                    "t_ref": [pulsar_params["t_ref"]],
                    "phase_ref": [pulsar_params["phase_ref"]],
                    "rot_freq": pulsar_params["rot_freq"],
                    "pulse_width": 1e-3,
                }
            }
        }
    )

    test = runner.KotekanStageTester(
        "visAccumulate",
        acc_par,
        runner.FakeGPUBuffer(**pulsar_params),
        dump_buffer,
        updatable_params,
    )

    test.run()

    yield dump_buffer_gated.load()


def test_structure(accumulate_data):

    n = accumulate_params["num_elements"]

    # Check that each samples is the expected shape
    for frame in accumulate_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == 0

    # Check that we have the expected number of samples
    nsamp = accumulate_params["total_frames"] // accumulate_params["int_frames"]
    assert len(accumulate_data) == nsamp


def test_metadata(accumulate_data):

    for frame in accumulate_data:
        assert frame.metadata.freq_id == accumulate_params["freq"]


def test_time(accumulate_data):
    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    t0 = timespec_to_float(accumulate_data[0].metadata.ctime)

    delta_samp = (
        accumulate_params["samples_per_data_set"] * accumulate_params["int_frames"]
    )

    for ii, frame in enumerate(accumulate_data):
        assert frame.metadata.fpga_seq == ii * delta_samp
        assert (timespec_to_float(frame.metadata.ctime) - t0) == pytest.approx(
            ii * delta_samp * 2.56e-6, abs=1e-5, rel=0
        )
        assert frame.metadata.fpga_length == delta_samp
        assert frame.metadata.fpga_total == delta_samp


def test_accumulate(accumulate_data):

    row, col = np.triu_indices(accumulate_params["num_elements"])

    pat = (row + 1.0j * col).astype(np.complex64)

    for frame in accumulate_data:

        assert (frame.vis == pat).all()
        assert (frame.weight == 8.0).all()
        assert (frame.flags == 1.0).all()
        assert (frame.gain == 1.0).all()


# Test that we are calculating the weights correctly in the presence of lost
# data.
def test_lostweights(lostweights_data):

    row, col = np.triu_indices(accumulate_params["num_elements"])
    ns = accumulate_params["samples_per_data_set"]
    nf = accumulate_params["num_gpu_frames"]
    pat = (row + 1.0j * col).astype(np.complex64)

    b, data = lostweights_data

    weight = pytest.approx((2 * ns - b) ** 2 * nf / 16.0, rel=1e-5)

    for frame in data:

        assert (frame.vis == pat).all()
        assert frame.weight == weight


# Test that we are accumulating the RFI flagged count correctly
def test_rfi_total(lostweights_data):

    ns = accumulate_params["samples_per_data_set"]
    nf = accumulate_params["num_gpu_frames"]

    b, data = lostweights_data

    for frame in data:
        assert frame.metadata.rfi_total == ((nf // 2) * b)
        assert frame.metadata.fpga_total == ((nf // 2) * (2 * ns - b))


# Test that the statistics are being calculated correctly
def test_gaussian(gaussian_data):

    exp_vis = np.identity(4)[np.triu_indices(4)]
    frac_var = 1.0 / (
        gaussian_params["samples_per_data_set"] * gaussian_params["num_gpu_frames"]
    )

    vis_set = np.array([frame.vis for frame in gaussian_data])
    weight_set = np.array([frame.weight for frame in gaussian_data])

    # These tests need a 5 sigma fluctuation to cause failure
    assert np.allclose(vis_set.mean(axis=0), exp_vis, atol=7e-4, rtol=0)
    assert np.allclose(vis_set.var(axis=0), frac_var, rtol=7e-2, atol=0)
    assert np.allclose((1.0 / weight_set).mean(axis=0), frac_var, rtol=1e-2, atol=0)
    vm = (1.0 / weight_set).mean(axis=0)
    vv = (1.0 / weight_set).std(axis=0)
    lv = weight_set.shape[0]


# Test that we can deal with whole frames being dropped
def test_missing_frames(drop_frame_data):

    exp_vis = np.identity(4)[np.triu_indices(4)]
    frac_var = 1.0 / (
        drop_frame_params["samples_per_data_set"]
        * drop_frame_params["num_gpu_frames"]
        * (1 - drop_frame_params["drop_probability"])
    )

    vis_set = np.array([frame.vis for frame in drop_frame_data])
    weight_set = np.array([frame.weight for frame in drop_frame_data])

    # These tests need a 5 sigma fluctuation to cause failure
    assert np.allclose(vis_set.mean(axis=0), exp_vis, atol=7e-4, rtol=0)
    assert np.allclose(vis_set.var(axis=0), frac_var, rtol=7e-2, atol=0)
    assert np.allclose((1.0 / weight_set).mean(axis=0), frac_var, rtol=2e-2, atol=0)


def test_int_time(time_data):

    time_per_frame = 2.56e-6 * time_params["samples_per_data_set"]
    frames_per_int = (int(time_params["integration_time"] / time_per_frame) // 2) * 2
    delta_samp = time_params["samples_per_data_set"] * frames_per_int
    fpga0 = time_data[0].metadata.fpga_seq

    for ii, frame in enumerate(time_data):
        assert frame.metadata.fpga_seq - fpga0 == ii * delta_samp
        assert frame.metadata.fpga_length == delta_samp
        assert frame.metadata.fpga_total == delta_samp


# Test that we are correctly normalising for lost packets
def test_lostsamples(lostsamples_data):

    row, col = np.triu_indices(accumulate_params["num_elements"])

    pat = (row + 1.0j * col).astype(np.complex64)

    for frame in lostsamples_data:

        assert np.allclose(frame.vis, pat, rtol=1e-7, atol=1e-8)


def test_pulsar(pulsar_data):

    # count number of frames that span a pulse
    pulse_width = int(
        pulsar_params["pulse_width"] / (pulsar_params["samples_per_data_set"] * 2.56e-6)
    )
    pulse_width += (
        pulsar_params["pulse_width"] % (pulsar_params["samples_per_data_set"] * 2.56e-6)
    ) > 0
    # count total number of frames in an accumulation
    if "num_gpu_frames" in pulsar_params:
        num_tot = pulsar_params["num_gpu_frames"]
    else:
        num_tot = int(
            pulsar_params["integration_time"]
            / (pulsar_params["samples_per_data_set"] * 2.56e-6)
        )
    actual_integration = num_tot * (pulsar_params["samples_per_data_set"] * 2.56e-6)
    # count number of pulses in an accumulation
    num_pulse = int(actual_integration * pulsar_params["rot_freq"])

    # fudge factor for numerical uncertainty
    fudge = 1e-3

    assert pulsar_data.num_time > 0

    vis = pulsar_data.data["vis"][:, pulsar_params["freq"]]
    # allow for one frame to be added from time to time
    assert (vis >= (1 - fudge) * 10 * pulse_width * num_pulse // num_tot).all()
    assert (vis <= (1 + fudge) * 10 * pulse_width * (num_pulse + 1) // num_tot).all()


def test_pulsar_metadata(pulsar_data):
    assert pulsar_data.file_metadata["gating_type"] == "pulsar"
    assert pulsar_data.file_metadata["gating_data"]["pulsar_name"] == "fakepsr"
    assert pulsar_data.file_metadata["gating_data"]["dm"] == pulsar_params["dm"]
    assert pulsar_data.file_metadata["gating_data"]["t_ref"] == [pulsar_params["t_ref"]]
    assert pulsar_data.file_metadata["gating_data"]["phase_ref"] == [
        pulsar_params["phase_ref"]
    ]
    assert (
        pulsar_data.file_metadata["gating_data"]["rot_freq"]
        == pulsar_params["rot_freq"]
    )
    assert pulsar_data.file_metadata["gating_data"]["segment"] == 100.0
    assert pulsar_data.file_metadata["gating_data"]["pulse_width"] == pytest.approx(
        1e-3
    )
    assert pulsar_data.file_metadata["gating_data"]["coeff"] == [pulsar_params["coeff"]]
