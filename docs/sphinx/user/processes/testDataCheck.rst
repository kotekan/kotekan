*************
testDataCheck
*************

The ``testDataCheck`` template checks synthetic data buffers. Choose the
registered stage name for the buffer's element type:

.. list-table:: Registered stages
   :header-rows: 1

   * - Stage name
     - Element type
   * - ``testDataCheckInt``
     - ``int32_t``
   * - ``testDataCheckUchar``
     - ``uint8_t``
   * - ``testDataCheckUint``
     - ``uint32_t``
   * - ``testDataCheckLong``
     - ``int64_t``
   * - ``testDataCheckUlong``
     - ``uint64_t``
   * - ``testDataCheckFloat``
     - ``float``
   * - ``testDataCheckFloat16``
     - ``float16_t``

``testDataCheckFloat16`` is available only when the build enables
``KOTEKAN_FLOAT16``.

.. doxygenclass:: testDataCheck
