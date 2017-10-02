To run kotekan FRB test mode, using 4 GPUs and no CPU verification:
sudo ./kotekan -c ../../kotekan/kotekan_test.yaml

To run kotekan FRB mode, using 4 GPUs (need dpdk):
sudo ./kotekan -c ../../kotekan/kotekan_frb.yaml

To run kotekan FRB mode with CPU verification, using only one GPU:
sudo ./kotekan -c ../../kotekan/kotekan_frb_verify-cpu-oneGPU.yaml

