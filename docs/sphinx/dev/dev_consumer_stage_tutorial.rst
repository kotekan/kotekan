************************
Consumer Stage tutorial
************************

So you want to write your first Consumer Stage! This tutorial will walk you through the steps for building the foundations of your stage and its associated test.

What is a Consumer Stage?
-------------------------
Stages can have `producer components` and/or `consumer components`.
To "consume" means that the stage reads frames from a :ref:`Kotekan Buffer <dev_buffers>` . To "produce" means it writes frames into a Kotekan buffer.
This tutorial will illustrate writing and testing a stage that only has consumer components.

The steps we will follow for developing the stage are:

1. Load the classes we will be using.
2. Register the stage with abstract factory.
3. Write a skeleton constructor. Within the constructor:

   * Register the stage as a consumer of `in_buf`.

   * Load some configuration options.

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

To let the compiler know about the stage, add it to lib/stages/CMakeLists.txt.

The compiling instructions can then be found here: http://lwlab.dunlap.utoronto.ca/kotekan/compiling/general.html.
