**********
Overview
**********


Introduction
-------------

**Kotekan** is a highly optimized framework for processing streaming data,
written in a combination of C/C++.
It is primarily designed for use on radio telescopes,
originally developed for the `CHIME <https://chime-experiment.ca/>`_ project.
It is similar in many respects to Software Defined Radio projects such as
`GNUradio <https://www.gnuradio.org/>`_ or
`Bifrost <https://arxiv.org/abs/1708.00720>`_,
though with a much greater focus on efficiency and throughput. 

The name **Kotekan** comes from a musical style traditionally played on a Balinese Gamelan,
a large chime instrument. See e.g. `Wikipedia <https://en.wikipedia.org/wiki/Kotekan>`_ or
`this excellent example <https://www.youtube.com/watch?v=Kfe3DudhY4w>`_ on Youtube.

The framework is conceptually straightforward: data is carried through the system in
a series of ring buffer objects, which are connected by processing blocks which manipulate the data.
Optional metadata structures can be passed alongside the streaming data,
tracking things like sequence numbers and stream identifiers.

The structure of a processing pipeline is defined at runtime,
following the prescription in a `YAML <http://yaml.org/>`_ configuration file
which enumerates the components required and various options for them.

**Kotekan** is known to work on Linux and MacOS systems.


``buffer`` Objects
------------------
The heart of kotekan is its ``buffer`` objects, which serve as the glue
between the various processing blocks, carrying data from one to the next.
They operate as ring buffers, each containing a number of ``frames``,
sections of contiguous host memory which can be efficiently written and read.

The ``buffer`` objects keep track of how many producers are writing into them,
and how many consumers are reading from them, marking each of their ``frames`` as
full (indicating that all producers have told the buffer they are finished),
or empty (indicating that all consumers have indicated completion).

.. note::
    For efficiency, ``frames`` are not implicitly zeroed or reset after use:
    it is the responsibility of future producers to leave them in a desired state.

``kotekanProcess`` Modules
--------------------------
The ``buffer`` objects are written to and read from by ``kotekanProcess``
signal processing modules.
These are intended to perform a variety of tasks,
ranging from gathering data from a network stream or external device,
to applying filters or other algorithmic manipulations to the data,
to sending data to the network or storing them on a local filesystem.

Each ``kotekanProcess`` registers its presence with every ``buffer`` it
needs to interact with, explicitly declaring itself as either
a producer (writing to available ``frames``),
or consumer (reading from filled ``frames``).



Co-processors and Accelerators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Kotekan** has been designed with GPU co-processing in mind,
but the model could easily be extended to service FPGA or other accelerators.
To maximize efficiency, flexibility, and to pipeline away copy-to/-from latencies,
accelerator kernels are managed within special ``kotekanProcess`` modules which handle
the device.

These modules are able to pre-fetch data into the accelerator's local memory,
execute a series of kernels without needing to copy data back to host memory,
and finally synching the end results back as needed.
Copy-to/-from operations are typically run in parallel with the kernel processing,
minimizing idle time.

Two such modules are included at present, built around the AMD GPUs used by CHIME,
but broadly capable of handling `OpenCL <https://www.khronos.org/opencl/>`_
and `HSA <http://www.hsafoundation.com/>`_ capable devices.

Configuration
--------------
Something about the YAML structure goes here.