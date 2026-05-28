# Upgrade plan for baseband system.
In addition to locking a few frames and memcpy'ing out the baseband data from in_buf into out_buf, form multiple beams using the full array data on the way out upon receiving a basebandRequest. The basebandRequest will come with detection beam rows, which we then turn into a list of calibrators and pass to each basebandReadoutManager via a basebandRequest.

Upon receiving the basebandRequest, generate phases on the CPU at a specified cadence (some array of shape num_beams x num_elements), and do the following. iterate through all the frames.

For each frame,
1) if needed, memcpy from in_buf to out_buf as before according to start_fpga and length_fpga in basebandRequest
2) beamform: according to start_mb_fpga and length_mb_fpga, multiply each in_buf frame of shape (num_samples, num_elements) and sum over num_elements and put the answer in outmb_buf. this will be a broader range of frames than spanned by start_fpga and length_fpga, hardcoded as datasets_per_scan. it should be centered on start_fpga_midband and be a scan no shorter than 8 seconds. since samples_per_dataset=512, datasets_per_scan = 3125000 / 512. add to config under basebandReadout.
3) every now, as specified by a number (samples_per_delay_update = 390625 / 10), re-calculate the phases using the the next time. add datasets_per_delay_update to configs/chime_bb_bk_nodpdk.j2, which we are using for development of this feature.
4) send outmb_buf to a basebandWriter stage. write it out to a separate file. 

Here are some changes to various stages:
## basebandReadout:
- don't define NUM_BEAMS in the .hpp. put it in the config as num_beams.
- Register a second output buffer, called outmb_buf, which sends out buffer frames of shape ['samples_per_data_set', 'num_local_freq' (labeled as 'num_freq_host_buf' in the config sometimes), 'num_beams']. Values for those numbers are typically (512, 8, 8) and can be found in chime_upgrade_bb_bk_nodpdk.j2 but should be kept generic.
- Study what's held in basebandMetadata and beamMetadata to design the chordMetadata that gets packed into the buffer frames sent into outmb_buf. It should pack the same info from out_metadata (event_id, freq_id, event_start_fpga, event_end_fpga, time0_fpga, time0_ctime, time0_ctime_offset, first_packet_recv_time, fpga0_ns, frame_fpga_seq, valid_to, num_elements, reserved) and also several vectors of length num_beams: (beam_number, ras, decs, source_names)
- wait_for_data: currently takes request.event_id, freq_id, stream_freq_idx, trigger_start_fpga, trigger_length_fpga. interface should be changed to wait_for_data(request, freq_id, stream_freq_idx, max_dump_samples). then do the std::min((int64_t)request.length_fpga, _max_dump_samples) within the logic  wait_for_data internally. dump_start_frame and dump_end_frame should be calculated using datasets_per_scan, not trigger_length_fpga. also remove num_beams here, consolidate into config.
- from chime_science_run_gpu.yaml, use the existing system of reading in gains (follow gain_tracking_buffer_0-3) to get the gains into basebandReadout. Fetch the gains only upon event trigger (i.e. same as lookup_cals and translate_trigger). In the config, and when a basebandReadout stage is instantiated, a gain_tracking_buffer should be sent in. note that the applyGains.cpp module is probably for visibility data, not beamforming.

# extract_data within basebandReadout 
- in place of the memcpy into outmb_buf, do beamforming: 4+4 bit data x floating-pt phases x floating-pt gains -> floating point output (float32+float32 real+imag). 
- then divide by num_elements and re-quantize to int4x2_t and put the answer in outmb_buf.
- if the size of the input buffer frame is (samples_per_dataset, num_local_freq, num_elements), the size of the outmb_buf frame should be (samples_per_dataset * num_elements / num_beams, num_local_freq, num_beams).
- replace the existing basebandMetadata code with a chordMetadata object for the data that gets dumped as a full-array that gets sent out. it should include all the information that's going into the basebandMetadata in the current implementation, the name of the gains file, and each pointing.
- use a DIFFERENT chordMetadata object to hold metadata for the multibeams.  it should include all the information that's going into the basebandMetadata in the current implementation, the name of the gains file, and each pointing.


# clBeamformPhaseData
- get_delays() function should be duplicated from here into basebandReadout.cpp. inst_lat, inst_long, fixed_time, and element_positions should be part of the basebandReadout.cpp config. and feed_positions (i.e. "element_positions" in the config) should be attributes of the CHIMETelescope object. the resulting function should have a call signature of get_delays(float* phases, time_t beamform_time, num_beams, float* ras, float* decs,). mark the version in clBeamformPhaseData.cpp as a duplicate of the one in Telescope.cpp.

## basebandReadoutManager
- basebandRequest should be modified to include uint64_t num_beams, 3 vectors of float64s of length num_beams: mb_ra and mb_dec and mb_name and std::string mb_file_name, mb_length_fpga.
This needs to send along the detection beam index as a unsigned integer between 0000 and 4000. Calibrator names will be passed in as a lookup table given each detection beam row.

- In main_thread, upon instantiation of a basebandReadoutManager, lock_range should lock at least int_frames*2 and no more than half the total number of frames - 3 frames.

## basebandApiManager
- request should include a detection_beam field: a uint64_t between 0000 and 4000. add it; unpack it in handle_request_callback.
- hack translate_trigger to get start_fpga_midband at 600.0 MHz (freq_id = 512).
- write lookup_cals(start_fpga_midband, detbeam). For now hard code this list of 8 sources to ra=0,1,2,3,90,91,92,93,180,181,182,183, 270,271,272,273, which all have dec = 89.0. make up some corresponding J-names for those sources. choose the first num_beams sources and return their ra and dec. run this alongside translate_trigger to determine calibrators for every basebandReadoutManager
- modify basebandSlice to also include num_beams RAs and num_beams Decs and num_beams source_names, e.g. the sources we will beamform to.
- in addition to translate_trigger which gets a readout_slice, call lookup_cals and pack the answers into basebandSlice.
- in handle_request_callback: report the source names, their ras, and their decs.
