#!/usr/bin/env python3
"""Generate the 3 local 'cross-node' kotekan configs for the split-band test.

Topology (all on 127.0.0.1, separate kotekan processes):
  FRONT (REST 12048): rawFileRead -> fftwEngine -> SubbandSplit
                      -> bufferSend(subA ->:14001), bufferSend(subB ->:14002)
                      -> bufferRecv(:14003 -> recA), bufferRecv(:14004 -> recB)
                      -> GnssCoherentCombiner -> write
  WORKER A (REST 12049): bufferRecv(:14001 -> subA) -> search_a + track_a -> recA
                         -> bufferSend(recA ->:14003)
  WORKER B (REST 12050): bufferRecv(:14002 -> subB) -> search_b + track_b -> recB
                         -> bufferSend(recB ->:14004)

The chordMetadata fpga_seq stamped by the FRONT's F-engine rides the wire
(bufferSend/Recv serialize it), so each worker's search/track reference the same
absolute 'sample 0' -- the cross-node reference under test. Trackers are
config-seeded (PRN 10/23) so the data path produces combined records without the
broker; the broker can drive them dynamically afterward.
"""

HEAD = """---
type: config
log_level: info
instrument_name: airspy_gps
telescope: {{ name: ICETelescope, num_polarizations: 1, num_dishes: 1 }}
rest_server: {{ cors_allow_origins: ["http://localhost:8080"] }}

samples_per_data_set: 50000
bytes_per_sample: 2
buffer_depth: 16
sizeof_float: 4
cpu_affinity: [0, 1, 2, 3]   # bufferSend/Recv require it; inherited by all stages

spectrum_length: 50
num_taps: 4
sample_rate: 20.0e6
f_offset: 5.0e6
n_prn: 2
record_floats: 11
hops_per_record: 200
doppler_margin_hz: 900000.0

gnss_pool: {{ kotekan_metadata_pool: GnssChanMetadata, num_metadata_objects: 8 * buffer_depth }}
rec_pool:  {{ kotekan_metadata_pool: GnssChanMetadata, num_metadata_objects: 8 * buffer_depth }}
"""

# Buffer definitions reused across roles (chan slice = 500 hops * 25 ch * 8 B = 100000).
SUB = "{{ kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: 100000 }}"
REC = "{{ kotekan_buffer: standard, metadata_pool: rec_pool, num_frames: buffer_depth, frame_size: n_prn * record_floats * sizeof_float }}"

FRONT = HEAD + """
input_buf: {{ kotekan_buffer: standard, metadata_pool: none,      num_frames: buffer_depth, frame_size: samples_per_data_set * bytes_per_sample }}
chan_buf:  {{ kotekan_buffer: standard, metadata_pool: gnss_pool, num_frames: buffer_depth, frame_size: samples_per_data_set * sizeof_float }}
subA_buf:  """ + SUB + """
subB_buf:  """ + SUB + """
recA_buf:  """ + REC + """
recB_buf:  """ + REC + """
out_buf:   {{ kotekan_buffer: standard, metadata_pool: none, num_frames: buffer_depth, frame_size: n_prn * record_floats * sizeof_float }}

read_in: {{ kotekan_stage: rawFileRead, buf: input_buf, base_dir: "{infile}", file_name: "gpsin", file_ext: "raw", end_interrupt: false }}
fengine: {{ kotekan_stage: fftwEngine, in_buf: input_buf, out_buf: chan_buf, input_type: real, pfb_window: hamming }}
split:   {{ kotekan_stage: GnssSubbandSplit, in_buf: chan_buf, out_bufs: [subA_buf, subB_buf], subband_lo: [0, 25], subband_hi: [25, 50] }}

send_a: {{ kotekan_stage: bufferSend, buf: subA_buf, server_ip: 127.0.0.1, server_port: 14001, drop_frames: false }}
send_b: {{ kotekan_stage: bufferSend, buf: subB_buf, server_ip: 127.0.0.1, server_port: 14002, drop_frames: false }}
recv_a: {{ kotekan_stage: bufferRecv, buf: recA_buf, listen_port: 14003, num_threads: 1 }}
recv_b: {{ kotekan_stage: bufferRecv, buf: recB_buf, listen_port: 14004, num_threads: 1 }}

combiner: {{ kotekan_stage: GnssCoherentCombiner, in_bufs: [recA_buf, recB_buf], out_buf: out_buf }}
write_records: {{ kotekan_stage: rawFileWrite, in_buf: out_buf, base_dir: "{outdir}", file_name: "gps", file_ext: "raw", prefix_hostname: false, num_frames_per_file: 100000, exit_after_n_files: 100000 }}
"""

WORKER = HEAD + """
sub_buf: """ + SUB + """
rec_buf: """ + REC + """

recv:  {{ kotekan_stage: bufferRecv, buf: sub_buf, listen_port: {chan_port}, num_threads: 1 }}
search: {{ kotekan_stage: GnssChannelizedSearch, in_buf: sub_buf, channel_offset: {off}, n_channels: 25,
          acquire_snr: 12.0, acquire_windows: 32, doppler_min: -6000.0, doppler_max: 6000.0, doppler_step: 500.0,
          prns: [5, 10, 23] }}
track: {{ kotekan_stage: GnssChannelizedTracker, in_buf: sub_buf, out_buf: rec_buf, channel_offset: {off}, n_channels: 25,
         prns: [10, 23], doppler_hz: [-1250.0, -2750.0], code_phase_chips: [647.0, 747.51] }}
send: {{ kotekan_stage: bufferSend, buf: rec_buf, server_ip: 127.0.0.1, server_port: {rec_port}, drop_frames: false }}
"""

open("config/dist_node_front.yaml", "w").write(FRONT.format(infile="/tmp/gpsin_ref", outdir="/tmp/gpsxnode"))
open("config/dist_node_a.yaml", "w").write(WORKER.format(off=0, chan_port=14001, rec_port=14003))
open("config/dist_node_b.yaml", "w").write(WORKER.format(off=25, chan_port=14002, rec_port=14004))
print("wrote dist_node_front.yaml, dist_node_a.yaml, dist_node_b.yaml")
