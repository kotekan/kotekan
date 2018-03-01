# Python test suite for Kotekan

This is a `py.test` based test suite for `kotekan`. It specifically focuses on
unit testing the post GPU functionality. It works by constructing special
configuration files that run `kotekan` dumping its raw buffer output
and then reading that back into Python space. It's not pretty but it works!

To run it you must:
- Have built `kotekan` into the standard location (i.e. `build/kotekan/`).
- Run the tests from within the `tests/` directory using `pytest`.

