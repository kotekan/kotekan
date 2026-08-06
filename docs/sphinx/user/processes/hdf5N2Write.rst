.. _hdf5N2Write:

***********
hdf5N2Write
***********

.. doxygenclass:: hdf5N2Write
   :members:
   :undoc-members:

Buffered writer for CHORD/CHIME N2 visibilities. Inline Doxygen covers the file
layout, chunking/compression, configuration, metrics, and an example YAML
snippet.

.. seealso::
   :ref:`n2_vis_file_format` for a complete reference of the on-disk HDF5
   file format (every dataset and attribute, with shapes, types, and
   semantics).

.. doxygenclass:: N2FileData
   :members:
   :undoc-members:

See ``tests/boost/test_hdf5N2Write.cpp`` for end-to-end expectations and file
layout checks.
