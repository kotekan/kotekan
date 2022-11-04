*******************
Unit Testing Stages
*******************

Let us add some basic system tests to the `ExampleConsumer` class describe earlier, so that we can make sure our stage is behaving as expected while we work on it.

Kotekan uses pytest for its system tests. Through the pytest framework you can set up a corresponding producer stage/input buffer, automate the creation of a :ref:`pipeline configuration <user_config>`, run your consumer stage, and then make assertions on its behaviour.

The steps for building a test are:

1. Set the parameters for the configuration of your stage.
2. Write a pytest fixture that runs kotekan:

   * Create a temporary directory for the data produced by your test.
   * If the stage you are testing has consumer components, set up a producer stage which generates the input buffer.
   * If the stage you are testing has producer components, set up the data dumper, which will allow you to validate your stage's behaviour.
   * Configure KotekanStageTester with your stage, and tell it where it will can find the input buffer, and/or where it can output its buffer.
   * ``yield`` the final buffer for asserting.

3. Create the actual test. Pass it your stage's pytest fixture. Assert the expected output frames.

.. literalinclude:: examples/example_test.py
    :language: python
    :linenos:

And there you go!

To run the test, while in the kotekan repo, type:

::

   kotekan$ export PYTHONPATH=python
   kotekan$ pytest -sv location_of_test/example_test.py

If it cannot find your new stage, ensure that pytest is running the correct local kotekan build, which includes your new code.

And congratulations! You now have the foundation in place to further develop your stage.
