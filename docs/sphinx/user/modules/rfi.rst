*******************
RFI Mitigation
*******************

Two variants of RFI removal currently exist in kotekan, one for general VDIF,
and one specific to the CHIME correlator.

CHIME
--------------

CHIME RFI removal is implemented as an HSA process. 

To add RFI removal, add the following the the kotekan config under GPU->commands:

name: ``hsa_rfi``
    kernel: "../lib/hsa/kernels/rfi_chime.hsaco"

    The RFI kernel can either set input data with RFI in it to zero, 
    or output an array with the amount of RFI detected by frequency.

    Other config parameters:

    * ``rfi_zero``: True/False (Whether or not to zero input data with RFI in it)
    * ``sk_step``: 256 (The time cadence for kurtosis measurements, in units of time samples)
    * ``rfi_sensitivity``: 10 (The sensitivity of the kurtosis threshold, lower is more sensitive)

    Relevent files for CHIME RFI can be located here:

    * /ch_gpu/lib/hsa/hsaRfi.cpp (.hpp)
    * /ch_gpu/lib/hsa/hsaRFIOutput.cpp (.hpp)
    * /ch_gpu/lib/hsa/kernels/rfi_chime.cl

VDIF
--------------
VDIF RFI removal is implemented in 3 different manners.

1. During the read
^^^^^^^^^^^^^^^^^^^
The Kotekan Process nDiskFileRead can be configured to do VDIF rfi detection/zeroing as it reads.

To configure nDiskFileRead for rfi removal, example config:

.. code-block:: YAML

    replay:
      kotekan_process: nDiskFileRead
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
The Kotekan Process computeDualpolPower can be configured to do VDIF rfi detection/zeroing as it integrates.

To configure computeDualpolPower for rfi removal, example config:

.. code-block:: YAML

    power:
      kotekan_process: computeDualpolPower
      rfi_removal: True #(RFI ON/OFF)
      rfi_sensitivity: 3 #(The sensitivity of the kurtosis threshold, lower is more sensitive)
      rfi_backfill: False #(Dumb Backfill of zeroed data, not recommended for use unless you need a pretty picture)
      vdif_in_buf: gpu_output_buffer_0
      power_out_buf: output_power_buf


Relevant files:

/ch_gpu/lib/processes/computeDualpolPower.cpp (.hpp)

3. As a GPU process
^^^^^^^^^^^^^^^^^^^
VDIF RFI removal can be implemented as an HSA process. 

To add RFI removal, add the following the the kotekan config under GPU->commands:

.. code-block:: YAML

    - name: hsa_rfi_vdif
        kernel: "../lib/hsa/kernels/rfi_vdif.hsaco"

The RFI kernel will set input data with RFI in it to zero, 

Other config parameters:

* ``sk_step``: 256 (The time cadence for kurtosis measurements, in units of time samples)
* ``rfi_sensitivity``: 10 (The sensitivity of the kurtosis threshold, lower is more sensitive)

Relevent files for VDIF RFI can be located here:

* /ch_gpu/lib/hsa/hsaRfiVdif.cpp (.hpp)
* /ch_gpu/lib/hsa/hsaRFIOutput.cpp (.hpp)
* /ch_gpu/lib/hsa/kernels/rfi_vdif.cl







