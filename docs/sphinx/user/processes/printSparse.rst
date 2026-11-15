***********
printSparse
***********

The ``printSparse`` template prints nonzero elements of a buffer. Choose the
registered stage name for the buffer's element type:

====================== ============
Stage name             Element type
====================== ============
``printSparseUint8``    ``uint8_t``
``printSparseUint32``   ``uint32_t``
``printSparseFloat16``  ``float16_t``
====================== ============

``printSparseFloat16`` is available only when the build enables
``KOTEKAN_FLOAT16``.

.. doxygenclass:: printSparse
