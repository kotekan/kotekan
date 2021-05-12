************
Unit Tests
************

Testing a Stage
===============

If you wrote a new Stage or you added to it's functionality, you should
probably test if it behaves as expected in all the possible cases.
To do this you can write a test using
`pytest <https://docs.pytest.org/en/latest/>`_ and various tools from
`runner.py <https://github.com/kotekan/kotekan/blob/master/python/kotekan/runner.py>`_
like in the following example.

Example
-------

Example test ``tests/test_<name>.py``:

.. literalinclude:: examples/example_test.py

* You can run your test with ``pytest tests/test_<name>.py``


Testing a C++ Function
======================

If you want to test a single C++ Function, you can write a test using the `Boost Unit Testing Framework <https://www.boost.org/doc/libs/1_44_0/libs/test/doc/html/utf.html>`_ as demonstrated by the following example.

Example
-------

* Write a test like this one (Replace ``<name>`` with what you are testing):

Example test ``tests/boost/test_<name>.cpp``:

.. literalinclude:: examples/example_test.cpp

* Add ``add_executable(test_<name> test_<name>.cpp)`` and ``target_link_libraries(test_<name> PRIVATE <all_used_libs>)`` to ``/tests/boost/CMakeLists.txt``.

* Build kotekan with the cmake option ``-DWITH_TESTS=ON`` under ``/kotekan/build``.

* Make sure ``pytest-cpp`` is installed.

* Run your test with ``pytest tests/boost/test_<name>``.

