************************
Consumer Stage tutorial
************************

So you want to write your first Consumer Stage! This tutorial will walk you through the steps for building the foundations of your stage and its associated test.

What is a Consumer Stage?
-------------------------
Stages can have `producer components` and/or `consumer components`.
To "consume" means that the stage reads frames from a [Kotekan buffer](/kotekan/dev/dev_buffers.html). To "produce" means it writes frames into a Kotekan buffer.
This tutorial will illustrate writing and testing a stage that only has consumer components.

The steps we will follow for developing the stage are:

1. Load the classes we will be using.
2. Register the stage with abstract factory.
3. Write a skeleton constructor. Within the constructor:
   3a. Register the stage as a consumer of `in_buf`.
   3b. Load some configuration options.
4. Write the skeleton for the framework managed pthread. Within this main thread:
   4a. Declare the pointer to the buffer.
   4b. Acquire the frame.
   4c. Handle what happens if a null frame is returned.
   4d. Release the frame.
   4e. Increase the ring pointer.
5. Create the header.


.. literalinclude:: ../../../lib/stages/SampleProcess.cpp
    :language: c++
    :linenos:

Now, let us create the header file.

.. literalinclude:: ../../../lib/stages/SampleProcess.hpp

To let the compiler know about the stage, add it to lib/stages/CMakeLists.txt.

The compiling instructions can then be found here: http://lwlab.dunlap.utoronto.ca/kotekan/compiling/general.html.

Finally, let us write some basic system tests, so that we can make sure our stage is behaving as expected while we work on it.

Kotekan uses pytest for its system tests. Through the pytest framework you can set up a corresponding producer stage/input buffer, automate the creation of a [pipeline configuration](), run your consumer stage, and then make assertions on its behaviour.

.. literalinclude:: examples/example_test.py
    :language: python
    :linenos:

And there you go!

To run the test, while in the kotekan repo, type:

```
kotekan$ export PYTHONPATH=python
kotekan$ pytest -sv location_of_test/example_test.py
```

If it cannot find your new stage, ensure that pytest is running the correct local kotekan build, which includes your new code.

And congratulations! You now have the foundation in place to further develop your stage.
