# Native CHIME detector replay

`replay_chime_native.py` is an offline correctness harness for a K=128 build.
It reads native HDF5 `baseband` bytes (`uint8[time,input]`, offset-binary 4+4-bit
samples), preserves every input stream, and trims partial frames to the absolute
FPGA origin. It runs `rawFileRead` through the normal CUDA detector stage and
compares the mask, three frame powers, rank support, frequency identity, shape,
and FPGA timing against independent CPU integer calculations.

The source bundle must be an archive-review-only candidate. The harness verifies
its file digests and creates a temporary numerical test fixture with the same
selected profile, weights, geometry, and literal word from the decisions file.
The fixture explicitly records that it is not physically calibrated or authorized
for activation. The source bundle is never modified. A channel that the decisions
file does not deploy has no word; the harness refuses it, since its frames are
masked and no detector word exists to replay. The generated graph has
only file input, the detector, and file outputs; it has no mask consumer or live
input. Keep candidate files and native-data receipts in an appropriately private
location, not in this repository.

Example (replace the paths with local inputs):

```sh
PYTHONPATH=/path/to/pilotproxy/src:$PWD/python \
  python tests/replay_chime_native.py \
  --binary /path/to/kotekan-k128 \
  --candidate /path/to/review-only-bundle \
  --decisions /path/to/decisions.json \
  --raw /path/to/native-capture.h5 \
  --output /path/to/private/replay-result --max-frames 4
```

The Python environment needs NumPy, h5py, PyYAML, and the matching PilotProxy
reference package. A CUDA-capable Kotekan binary is required; unavailable hardware
or a failed stage is an error, never a skipped replay. The harness disables host
buffer `mlock` for this bounded file test. Its wall time is not a throughput or
real-time performance measurement.

A protocol records source, candidate, binary, reference-module, and raw-file
hashes. CPU results checkpoint after each frame. Resume requires an identical
protocol; a completed result is reused only after all saved output and reference
hashes pass. Interrupted workspaces remain available for diagnosis. Successful
runs remove only their own uniquely created temporary fixture and voltage copies.

The pair diagnostic uses absolute zero-origin pairs of 16,384-sample frames.
Incomplete pairs are excluded, and both frames are retained only when both
individual decisions retain them. An individually retained frame paired with a
rejected frame is therefore rejected in this diagnostic. Stored adjacent rows
are never assumed to form a production pair. The receipt reports signed proxy
means and the clipped display values for individual selection and complete-pair
selection. A small finite capture cannot estimate an event-ensemble confidence
bound or qualify an archive-wide science policy. This harness does not implement
a runtime pair accumulator, physical calibration, or deployment.

Run the small CPU-only helper contracts with:

```sh
PYTHONPATH=tests:/path/to/pilotproxy/src:$PWD/python \
  python -m unittest discover -s tests -p test_replay_chime_native.py -v
```
