C++ Code Guidelines
---------------------

Naming
^^^^^^^^^^
- Type names in kotekan should be nouns, use *CamelCase* formatting and begin
  with a capital letter, e.g. ``MyFavouriteProcess``. This includes classes,
  enums, structs and typedefs.
- With the exception of class constructors & destructors, function names should
  use underscore notation and begin with a lower case letter, e.g. ``my_func``.

Variables
^^^^^^^^^^
Variables in the code should use underscore naming, e.g.
``my_favourite_variable``.

Explicit typing should be used wherever possible, e.g. always use ``uint32_t``
rather than ``uint``.

Private member variables should start with an underscore, e.g.
``_my_private_variable``.

Namespaces
^^^^^^^^^^
Avoid `using namespace X;`. Instead specify where you are using classes or
functions from a namespace, e.g. `std::vector<std::string> my variable;`.
Never do `using namespace std;`.

Header files
^^^^^^^^^^^^^
C++ header files should be named after the class they describe and use the
ending `.hpp`. They should begin with a header file commit like

.. code-block:: c++

    /****************************************************
    @file
    @brief Brief description of the classes in this file.
    - MyClass
    - MyOtherClass : public MyOtherBaseClass
    *****************************************************/

Immediately after the header file comment include guards should follow:

.. code-block:: c++

    #ifndef MY_CLASS_H
    #define MY_CLASS_H

With a simple

.. code-block:: c++

    #endif

at the end of the file.


#include Style
^^^^^^^^^^^^^^^
The list of #includes should be in the order:

 - Header file implementing the interface of this `.cpp` file.
 - System #includes
 - Local/Private Headers
 - kotekan project headers

Each category should be sorted lexicographically by the full path.