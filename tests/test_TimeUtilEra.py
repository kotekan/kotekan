import pytest
import numpy as np
import astropy.time
import astropy.units as u

from kotekan import baseband_buffer, runner

def init_printing():
    with open("py_out.txt", "w") as f:
        pass

def my_print(*args):
    with open("py_out.txt", "a") as f:
        print(*args, file=f)


entry_size = 8
num_outputs = 3

t0_gps = astropy.time.Time(dict(year=1980, month=1, day=6, hour=0,
                            minute=0, second=19.0),
                       format='ymdhms', scale='tai')

init_printing()

@pytest.fixture(scope="module")
def runner_factory(tmpdir_factory):

    def _runner_factory(t, dUT, dAT):

        time_length = t.shape[0]
        array_length = 2 * time_length
        frame_size = array_length * entry_size

        t_sec, t_nsec = time_to_s_ns((t - t0_gps).tai)
        
        t_input = np.empty(array_length, dtype=np.int64)
        t_input[::2] = t_sec
        t_input[1::2] = t_nsec

        t_input_b = [b for b in t_input.tobytes()]
        
        base_buffer_t = baseband_buffer.BasebandBuffer.new_from_params(
                event_id=42,
                freq_id=0,
                num_elements=1,
                frame_size=frame_size,
                frame_data=t_input_b)

        tmpdir = tmpdir_factory.mktemp("gps_time_data")

        read_buffer = runner.ReadBasebandBuffer(str(tmpdir), [base_buffer_t])
        read_buffer.write()

        dump_buffer = runner.DumpBasebandBuffer(str(tmpdir), 1,
                                            frame_size=(frame_size*num_outputs))

        timeUtil_params = {
                'dUT_sec': dUT,
                'dAT_sec': dAT}

        default_params = {
                "num_elements": 1,
                "num_gpu_frames": 1,
                "samples_per_data_set": frame_size,
                "baseband_metadata_pool": {
                    "kotekan_metadata_pool": "BasebandMetadata",
                    "num_metadata_objects": 16}
                }

        test = runner.KotekanStageTester(
                "TimeUtilDump",
                timeUtil_params,
                read_buffer,
                dump_buffer,
                default_params,
                expect_failure=True)

        test.run()

        for frame in dump_buffer.load():

            data_bytes = frame._buffer[:]
            data = np.frombuffer(data_bytes, dtype=np.int64)

            yield data

    return _runner_factory


def time_to_s_ns(t):
    jd1_fracsec, jd1_sec = np.modf(t.jd1 * 86400)
    jd2_fracsec, jd2_sec = np.modf(t.jd2 * 86400)

    jd_sec = (jd1_sec + jd2_sec).astype(np.int64)

    # jd_nsec = np.round(1.0e9 * (jd1_fracsec + jd2_fracsec)).astype(np.int64)
    jd_nsec = np.round(1.0e9 * (jd1_fracsec + jd2_fracsec)).astype(np.int64)

    GIGA = 1000000000

    low = jd_nsec < 0.0
    while low.any():
        jd_sec[low] -= 1
        jd_nsec[low] += GIGA
        low = jd_nsec < 0.0
    
    high = jd_nsec >= GIGA
    while high.any():
        jd_sec[high] += 1
        jd_nsec[high] -= GIGA
        high = jd_nsec >= GIGA

    return jd_sec, jd_nsec

def get_dAT(t):
    return float(round(
        (t.tai - astropy.time.Time(t.utc.value, scale='tai')).sec))


def test_structure(runner_factory):

    N = 38
    t = t0_gps + (np.linspace(0.0, 10.0, N) * u.s)

    for data in runner_factory(t, 0.0, 0.0):

        assert data.shape == (num_outputs * 2 * N,)

def test_tai(runner_factory):
    
    N = 47

    t = astropy.time.TimeDelta(np.linspace(356.76, 98739.71, N),
                               format='sec', scale='tai')

    gps_sec, gps_nsec = time_to_s_ns(t)

    for data in runner_factory(t0_gps + t, 0.0, 0.0):

        stride = 2*num_outputs
        tai_sec = data[0::stride]
        tai_nsec = data[1::stride]

        assert (tai_sec == (gps_sec + 19)).all()
        assert (tai_nsec == gps_nsec).all()

def test_ut1(runner_factory):
    
    N = 32

    raw_tgps = np.linspace(12345.678, 13579.2468, N)
    safe_tgps = 1.0e-9 * np.round(1.0e9 * raw_tgps)

    t = astropy.time.TimeDelta(safe_tgps, format='sec', scale='tai')
    T = t0_gps + t

    my_print("delta_utc_ut1", T.delta_ut1_utc)

    dUT = float(T[0].delta_ut1_utc)
    dAT = get_dAT(T[0])

    T.delta_ut1_utc = dUT

    check_ut1_s, check_ut1_ns = time_to_s_ns(T.ut1)

    for data in runner_factory(T, dUT, dAT):

        stride = 2*num_outputs
        ut1_sec = data[2::stride]
        ut1_nsec = data[3::stride]

        my_print("ut1_sec:", ut1_sec)
        my_print("target ut1_sec:", check_ut1_s)
        my_print("diff:", ut1_sec - check_ut1_s)
        my_print("ut1_nsec:", ut1_nsec)
        my_print("target ut1_nsec:", check_ut1_ns)
        my_print("diff:", ut1_nsec - check_ut1_ns)
        my_print(t[-2])
        my_print(T[-2])
        my_print(dUT)

        assert (ut1_sec == check_ut1_s).all()
        assert (ut1_nsec == check_ut1_ns).all()

def test_era(runner_factory):
    
    N = 32

    raw_tgps = np.linspace(12345.678, 13579.2468, N)
    safe_tgps = 1.0e-9 * np.round(1.0e9 * raw_tgps)

    t = astropy.time.TimeDelta(safe_tgps, format='sec', scale='tai')
    T = t0_gps + t

    my_print("delta_utc_ut1", T.delta_ut1_utc)

    dUT = float(T[0].delta_ut1_utc)
    dAT = get_dAT(T[0])

    T.delta_ut1_utc = dUT

    target_era = T.earth_rotation_angle('tio').deg

    for i, data in enumerate(runner_factory(T, dUT, dAT)):

        my_print("frame:", i)

        stride = 2*num_outputs
        era = np.frombuffer(data[4::stride].tobytes(), dtype=np.float64)
        era2 = np.frombuffer(data[5::stride].tobytes(), dtype=np.float64)

        my_print("era:", era)
        my_print("target era:", target_era)
        my_print("diff:", era - target_era)
        my_print("era2:", era2)
        my_print("diff:", era2 - target_era)

        uas = 1.0e-6 / 3600
        nas = 1.0e-9 / 3600
        my_print("nas", nas)

        assert (np.fabs(era - target_era) < uas).all()
        assert (np.fabs(era2 - target_era) < uas).all()
        # assert (np.fabs(era - target_era) < nas).all()
        # assert (np.fabs(era2 - target_era) < nas).all()

