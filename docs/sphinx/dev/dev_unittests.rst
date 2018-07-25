************
Unit Tests
************

Testing a Process
=================

If you wrote a new Process or you added to it's functionality, you should
probably test if it behaves as expected in all the possible cases.
To do this you can write a test using
`pytest <https://docs.pytest.org/en/latest/>`_ and various tools from
`kotekan_runner.py <https://github.com/kotekan/kotekan/blob/master/tests/kotekan_runner.py>`_
like in the following example.

Example
-------

``tests/test_<name>.py``

.. literalinclude:: examples/example_test.py

* You can run your test with ``pytest tests/test_<name>.py``


Testing a C++ Function
======================

If you want to test a single C++ Function, you can write a test using the `Boost Unit Testing Framework <https://www.boost.org/doc/libs/1_44_0/libs/test/doc/html/utf.html>`_ as demonstrated by the following example.

Example
-------

* Write a test like this one (Replace ``<name>`` with what you are testing):

``tests/boost/test_<name>.cpp``

.. literalinclude:: examples/example_test.cpp

* Add ``test_<name>.cpp`` to ``/tests/boost/CMakeLists.txt``.

* Build kotekan with the cmake option ``-DBOOST_TESTS=ON``.

* Run your test with ``pytest tests/test_<name>``

