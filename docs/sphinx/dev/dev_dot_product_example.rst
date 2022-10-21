****************
Pipeline Example
****************
The following example outlines how to create a `kotekan` pipeline that performs a dot product between two vectors.

Create a Dot Product Stage
--------------------------
First we need to create a `kotekan` `Stage` that consumes two input buffers, i.e. vector A and B, and computes the dot product. The `Stage` should also write the result of the computation, A.B, to an output buffer.

The header and source files (`DotProduct.hpp/cpp`) are shown below:

.. literalinclude:: ../../../lib/stages/ExampleDotProduct.hpp

.. literalinclude:: ../../../lib/stages/ExampleDotProduct.cpp
    :language: c++

Compile Stage
-------------
To compile the stage add the source file to `lib/stages/CMakeLists.txt`:

.. code-block:: bash

    add_library(
        kotekan_stages
        beamformingPostProcess.cpp
        chrxUplink.cpp
        ...
        ExampleProducer.cpp
        ExampleConsumer.cpp
        ExampleDotProduct.cpp)

Move to the `build` directory, call `cmake` to create the `Makefile` and compile `kotekan`:

.. code-block:: bash

    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug ..
    make

Once compilation is complete we need to create a config file for `kotekan` to parse and run.

Pipeline Config Creation
------------------------
`kotekan` runs by parsing a configuration file that describes a pipeline. Each config file is written as a `.yaml` file and describes a set of data streams (`Buffers`) through a series of `Stages`. The config file below performs a dot product on two vectors:

.. literalinclude:: ../../../config/examples/dot_product.yaml
    :language: yaml

The two buffers, `input_a_buffer` and `input_b_buffer`, represent the vectors A and B. `output_buffer` will store the result of the dot product.

.. literalinclude:: ../../../config/examples/dot_product.yaml
    :lines: 10-25
    :language: yaml

The `data_gen` section populates the input buffers with constant values: 2.0 and 3.0, using the `testDataGenFloat` `Stage`. 

.. literalinclude:: ../../../config/examples/dot_product.yaml
    :lines: 27-37
    :language: yaml

The `dot_product` section runs the `DotProduct` `Stage` on the two input buffers and writes the result to the output buffer.

.. literalinclude:: ../../../config/examples/dot_product.yaml
    :lines: 39-44
    :language: yaml

Pipeline Graph
--------------
The graph below shows the pipeline of this example and was generated using the :ref:`Pipeline Viewer <pipeline_viewer>`.

.. image:: images/dot_product_pipeline.png
    :align: center

Execute kotekan
---------------
To run `kotekan` move to the binary directory and pass the config file as an argument:

.. code-block:: bash

    cd kotekan
    ./kotekan -c ../../config/examples/dot_product.yaml
