Config File Settings
^^^^^^^^^^^^^^^^^^^^^^

Variables
+++++++++
Variables and parameters in the config file should use underscore naming, e.g. ``my_favourite_variable``.

If you need to reference the size of a datatype, use a C-like naming, e.g. ``sizeof_float = 4``.

Buffers
+++++++++++
``Buffers`` used by a ``kotekan::Stage`` should be named ``in_buf`` or ``out_buf``,
for input (being produced into) and output (being consumed from), respectively.

When multiple ``Buffers`` of a similar type are needed,
they should go into an array of buffers, similarly named.

If multiple ``Buffers`` of different types are used by a single stage,
a short identifier should be appended to their names,
e.g. ``in_buf_voltages`` and ``in_buf_pointing``.
