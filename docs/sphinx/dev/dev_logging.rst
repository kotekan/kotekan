************
Logging
************

Log Levels
----------
* **OFF**: No logs at all
* **ERROR**: Serious error
* **WARN**: Warning about something wrong
* **INFO**: Helpful ideally short and infrequent, message about system status
* **DEBUG**: Message for debugging reasons only
* **DEBUG2**: Super detailed debugging messages

Note both DEBUG and DEBUG2 are removed entirely when building in release mode.

* **FATAL_ERROR**: Like *ERROR* but also exits kotekan returning an error code

In classes inheriting from `kotekan::kotekanLogging`
----------------------------------------------------
Logging macros are defined in ``lib/core/kotekanLogging.cpp``. Use `DEBUG2`, `DEBUG`, `INFO`,
`WARN` and `ERROR` from any class inheriting from `kotekan::kotekanLogging`.
Use `fmt format string syntax <https://fmt.dev/latest/syntax.html>`_.

Non-object oriented logging
---------------------------
If you need logging in a static method or outside of `kotekan::kotekanLogging`, you can use
`DEBUG2_NON_OO`, `DEBUG_NON_OO`, `INFO_NON_OO`, `WARN_NON_OO` and `ERROR_NON_OO`.
Use `fmt format string syntax <https://fmt.dev/latest/syntax.html>`_.

Format String Checks
--------------------
The fmt format strings are checked on compile-time if your compiler has sufficient `constexpr`
support (e.g. g++ 6.1).

C logging
---------
If you need logging in C code, use the macros `DEBUG2_F`, `DEBUG_F`, `INFO_F`, `WARN_F` and
`ERROR_F` defined in ``lib/core/errors.h``.
Use `printf format string
syntax <http://pubs.opengroup.org/onlinepubs/009695399/functions/fprintf.html>`_.
