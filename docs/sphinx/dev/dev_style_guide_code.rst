Code Guidelines
-----------------

Classes
^^^^^^^^^^
Classes in kotekan should use *CamelCase* formatting, e.g. ``myFavouriteProcess``.

Functions
^^^^^^^^^^
With the exception of class constructors & destructors, function names should use underscore notation,
e.g. ``my_func``.


Variables
^^^^^^^^^^
Variables in the code should use underscore naming, e.g. ``my_favourite_variable``.

Explicit typing should be used wherever possible, e.g. always use ``uint32_t`` rather than ``uint``.

Private member variables should start with an underscore, e.g. ``_my_private_variable``.

Structs
^^^^^^^^^^

Enums
^^^^^^^^^^

Namespaces
^^^^^^^^^^
Avoid `using namespace X;`. Instead specify where you are using classes or functions from a namespace, e.g. `std::vector<std::string> my variable;`. Never do `using namespace std;`.