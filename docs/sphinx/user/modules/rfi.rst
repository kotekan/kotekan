*******************
RFI Mitigation
*******************

A general VDIF stage for RFI removal currently exists in kotekan.

VDIF
--------------
VDIF RFI removal is implemented in 3 different manners.

1. During the read
^^^^^^^^^^^^^^^^^^^
The Kotekan Stage nDiskFileRead can be configured to do VDIF rfi detection/zeroing as it reads.

To configure nDiskFileRead for rfi removal, example config:

.. code-block:: YAML

    replay:
      kotekan_stage: nDiskFileRead
      num_disks: 10
      disk_base: /mnt/
      disk_set: A
      out_buf: vdif_input_buf_0
      capture: 20170426T110023Z_ARO_raw
      sk_step: 16384 #(The time cadence for kurtosis measurements, in units of time samples)
      rfi: True #(RFI ON/OFF)
      rfi_sensitivity: 3 #(The sensitivity of the kurtosis threshold, lower is more sensitive)
      normalize: True #(Whether or not to normalize before kurtosis measurment)

Relevant files:

* /ch_gpu/lib/processes/nDiskFileRead.cpp (.hpp)

2. During the power integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Kotekan Stage computeDualpolPower can be configured to do VDIF rfi detection/zeroing as it integrates.

To configure computeDualpolPower for rfi removal, example config:

.. code-block:: YAML

    power:
      kotekan_stage: computeDualpolPower
      rfi_removal: True #(RFI ON/OFF)
      rfi_sensitivity: 3 #(The sensitivity of the kurtosis threshold, lower is more sensitive)
      rfi_backfill: False #(Dumb Backfill of zeroed data, not recommended for use unless you need a pretty picture)
      vdif_in_buf: gpu_output_buffer_0
      power_out_buf: output_power_buf


Relevant files:

/ch_gpu/lib/processes/computeDualpolPower.cpp (.hpp)


