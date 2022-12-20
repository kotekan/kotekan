******************
Writing new Stages
******************

So you want to write your own Stages!  This tutorial will walk you through the steps for building the foundations of your stage,
building a simple pipeline, and unit-testing your stage.

Writing a Consumer Stage
------------------------
Stages can have `producer components` and/or `consumer components`.
To "consume" means that the stage reads frames from a :ref:`Kotekan Buffer <dev_buffers>` . To "produce" means it writes frames into a Kotekan buffer.
This tutorial will illustrate writing and testing a stage that only has consumer components.

The steps we will follow for developing the stage are:

1. Load the classes we will be using.
2. Register the stage with abstract factory.
3. Write a skeleton constructor. Within the constructor:

   * Register the stage as a consumer of `in_buf`.

4. Write the skeleton for the framework managed pthread. Within this main thread:

   * Declare the pointer to the buffer.

   * Acquire the frame.

   * Handle what happens if a null frame is returned.

   * Release the frame.

   * Increase the ring pointer.

5. Create the header.


.. literalinclude:: ../../../lib/stages/ExampleConsumer.cpp
    :language: c++
    :linenos:

Now, let us create the header file.

.. literalinclude:: ../../../lib/stages/ExampleConsumer.hpp
    :language: c++
    :linenos:

To let the compiler know about the stage, add it to lib/stages/CMakeLists.txt.

The compiling instructions can then be found at :ref:`Compiling Kotekan <compiling>` .


Writing a Producer Stage
------------------------
A Kotekan "producer" writes data into a :ref:`Kotekan Buffer <dev_buffers>` .
This page demonstrates how to write a stage that only produces data.

The steps are:

1. Load the classes we will be using.
2. Register the stage with abstract factory.
3. Write a skeleton constructor. Within the constructor:

   * Register the stage as a producer of `in_buf`.

   * Load some configuration options.

4. Write the skeleton for the framework managed pthread. Within this main thread:

   * Declare the pointer to the buffer.

   * Acquire the frame.

   * Handle what happens if a null frame is returned.

   * Release the frame.

   * Increase the ring pointer.

5. Create the header.


.. literalinclude:: ../../../lib/stages/ExampleProducer.cpp
    :language: c++
    :linenos:

Now, let us create the header file.

.. literalinclude:: ../../../lib/stages/ExampleProducer.hpp
    :language: c++
    :linenos:

To let the compiler know about the stage, add it to lib/stages/CMakeLists.txt.


Writing a Producer/Consumer Stage
---------------------------------

Next, we will create a `kotekan` `Stage` that consumes two input buffers, i.e. vector A and B, and computes their dot product. It will write the (elementwise) dot product to an output buffer.

The header and source files (`DotProduct.hpp/cpp`) are shown below.

.. literalinclude:: ../../../lib/stages/ExampleDotProduct.hpp
    :language: c++
    :linenos:

.. literalinclude:: ../../../lib/stages/ExampleDotProduct.cpp
    :language: c++
    :linenos:


Creating a Pipeline
-------------------
See the `User Guide section <user_pipeline_example>` on creating a pipeline,
which uses these example producer, consumer, and dot-product (producer/consumer) stages.
