import pytest
import numpy as np

from kotekan import baseband_buffer, runner

array_length = 1024
entry_size = 4

frame_size = array_length * entry_size

x = np.linspace(0.0, 10.0, array_length, dtype=np.float32)
xb = [b for b in x.tobytes()]
print(xb[0])
print(xb[1])
print(xb[2])
print(xb[3])
print(xb[4])
print(xb[5])
print(xb[6])
print(xb[7])
print(len(xb))

bufferA = baseband_buffer.BasebandBuffer.new_from_params(
        event_id=42,
        freq_id=0,
        num_elements=1,
        frame_size=array_length * entry_size,
        frame_data=xb)

bufferB = baseband_buffer.BasebandBuffer.new_from_params(
        event_id=42,
        freq_id=0,
        num_elements=1,
        frame_size=array_length * entry_size,
        frame_data=xb)


baseband_buffer.BasebandBuffer.to_files([bufferA], "geoff_test_dump/input_A")
baseband_buffer.BasebandBuffer.to_files([bufferB], "geoff_test_dump/input_B")

read_bufferA = runner.ReadBasebandBuffer("geoff_test_dump/", [bufferA])
read_bufferB = runner.ReadBasebandBuffer("geoff_test_dump/", [bufferB])

buf2 = baseband_buffer.BasebandBuffer.from_file("geoff_test_dump/input_A_0000000.dump")

data = buf2._buffer[buf2.meta_size:]

print(len(data))

y = np.frombuffer(data, dtype=np.float32)

print(y.shape)
print(y[0], y[1], y[-1])

print(read_bufferA.name)
print(read_bufferB.name)

default_params = {
    "baseband_metadata_pool": {
        "kotekan_metadata_pool": "BasebandMetadata",
        "num_metadata_objects": 1,
        }
    }


@pytest.fixture(scope="module")
def produce_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("convolve")

    dump_buffer = runner.DumpBasebandBuffer(str(tmpdir),
                                            1,
                                            frame_size=frame_size)

    test = runner.KotekanStageTester(
            "GeoffProducer",
            {
                'x_period': 5.0,
                'speed': 0.0,
                'x0': 2.5,
                'lo': -1.0,
                'hi': 1.0,
                'width': 1.0,
                'type': 1,
            },
            None,
            dump_buffer,
            default_params)

    print("Running")
    test.run()

    print("Loading")
    yield dump_buffer.load()


def test_structure(produce_data):

    print("In test")

    assert (1 == 0)

    for frame in produce_data:
        print(i)
        print(frame)
        print(frame.__dict__)

