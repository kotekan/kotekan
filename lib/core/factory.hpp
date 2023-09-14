/**
 * @file
 *
 * This contains a static class and helper macros for implementing an Abstract
 * Factory. The purpose is to allow derived classes to be registered and then created
 * by name and returned as pointers to the base class.
 *
 * To use it you need to use the `CREATE_FACTORY(baseclass, <constructor
 * signature>)` macro and then for any derived class you simply use the macro
 * `REGISTER_TYPE_WITH_FACTORY(baseclass, derivedclass)`.
 *
 * Example:
 * ```
 * #include <iostream>
 * #include <string>
 * #include "factory.hpp"
 *
 * class A {
 * public:
 *     A(int x, std::string b) {};
 * };
 *
 * CREATE_FACTORY(A, int, std::string);
 *
 * class B : public A {
 * public:
 *     B(int, x, std::string y) {
 *         std::cout << x << " " << y << std::endl;
 *     }
 * };
 *
 * REGISTER_TYPE_WITH_FACTORY(A, B);
 *
 * int main(int argc, char** argv) {
 *     auto t = FACTORY(A)::create_unique("B", 4, "hello");
 *     // Should print:
 *     // 4 hello
 * }
 * ```
 **/
#ifndef _FACTORY_HPP
#define _FACTORY_HPP

#include "errors.h"
#include "kotekanLogging.hpp"
#include "type.hpp"

#include "fmt.hpp"

#include <cxxabi.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

// clang-format off

/**
 * @brief Create a Factory for the specified type.
 *
 * This templated class acts as an abstract factory for the base type. You
 * *must* give the signature of the constructor that you wish to use as a
 * template argument. This class is purely static, you shouldn't try and
 * create instances of it.
 *
 * Template arguments:
 * @param  T        Type for factory.
 * @param  Args...  Argument types for the constructor.
 **/
template<typename T, typename... Args>
class Factory {
public:

    /**
     * Create a new instance of the type.
     *
     * @param  type Label of type to create.
     * @param  args Arguments for constructor.
     *
     * @return          Bare pointer to new object.
     **/
    static T* create_bare(const std::string& type, Args&&... args)
    {
        auto& r = type_registry();
        if (r.find(type) == r.end()) {
            throw std::runtime_error(fmt::format(fmt("Could not find subtype name {:s} within {:s} Factory."), type, type_demangle<T>()));
        }
        DEBUG_NON_OO("FACTORY({:s}): Creating {:s} instance.", typelabel(), type);
        return r.at(type)(std::forward<Args>(args)...);
    }

    /**
     * Create a new instance of the type.
     *
     * @param  type Label of type to create.
     * @param  args Arguments for constructor.
     *
     * @return          Unique pointer to new object.
     **/
    static std::unique_ptr<T> create_unique(const std::string& type, Args&&... args)
    {
        return std::unique_ptr<T>(create_bare(type, std::forward<Args>(args)...));
    }

    /**
     * Create a new instance of the type.
     *
     * @param  type Label of type to create.
     * @param  args Arguments for constructor.
     *
     * @return          Shared pointer to new object.
     **/
    static std::shared_ptr<T> create_shared(const std::string& type, Args&&... args)
    {
        return std::shared_ptr<T>(create_bare(type, std::forward<Args>(args)...));
    }

    /**
     * Create a new instance of the type, being forgiving if the requested type name does not exist.
     *
     * @param  type Label of type to create.
     * @param  args Arguments for constructor.
     *
     * @return          Bare pointer to new object, or NULL if type does not exist.
     **/
    static T* create_bare_if_exists(const std::string& type, Args&&... args)
    {
        auto& r = type_registry();
        if (r.find(type) == r.end()) {
            DEBUG_NON_OO("Could not find subtype name {:s} within {:s} Factory.", type, type_demangle<T>());
            return nullptr;
        }
        DEBUG_NON_OO("FACTORY({:s}): Creating {:s} instance.", typelabel(), type);
        return r.at(type)(std::forward<Args>(args)...);
    }

    /**
     * Create a new instance of the type, being forgiving if the requested type name does not exist.
     *
     * @param  type Label of type to create.
     * @param  args Arguments for constructor.
     *
     * @return          Unique pointer to new object (empty if type does not exist).
     **/
    static std::unique_ptr<T> create_unique_if_exists(const std::string& type, Args&&... args)
    {
        return std::unique_ptr<T>(create_bare_if_exists(type, std::forward<Args>(args)...));
    }

    /**
     * Create a new instance of the type, being forgiving if the requested type name does not exist.
     *
     * @param  type Label of type to create.
     * @param  args Arguments for constructor.
     *
     * @return          Shared pointer to new object (empty if type does not exist).
     **/
    static std::shared_ptr<T> create_shared_if_exists(const std::string& type, Args&&... args)
    {
        return std::shared_ptr<T>(create_bare_if_exists(type, std::forward<Args>(args)...));
    }

    /**
     * Create a new instance of the type.
     *
     * @tparam  U     Subtype to register.
     * @param  type  Label of type to register.
     **/
    template<typename U>
    static int register_type(const std::string& type)
    {
        DEBUG_NON_OO("FACTORY({:s}): Registering {:s}.", typelabel(), type);
        // Register the creation function
        type_registry()[type] = [](Args&&... args) -> T* {
            return new U(std::forward<Args>(args)...);
        };

        // Register the reverse lookup
        label_registry()[std::type_index(typeid(U))] = type;
        return 0;
    }

    /**
     * Check that the type has been registered.
     *
     * @param  type  Name of type.
     *
     * @return       Has type of name been registered.
     **/
    static bool exists(const std::string& type)
    {
        return (type_registry().count(type) > 0);
    }

    /**
     * Get the type label corresponding to the object.
     *
     * @param  obj  Pointer to an instance of the base type.
     *
     * @return      The string label the type was registered as.
     **/
    static std::string label(const T& obj)
    {
        return label_registry().at(std::type_index(typeid(obj)));
    }

    /**
     * Get the type label corresponding to the type.
     *
     * @tparam  U  A sub-class of T to get the registered label for.
     *
     * @return     The string label the type was registered as.
     **/
    template<typename U,
             typename std::enable_if_t<std::is_base_of<T, U>::value>* = nullptr>
    static std::string label()
    {
        return label_registry().at(std::type_index(typeid(U)));
    }

private:

    // Return a reference to the type registry.
    static auto& type_registry()
    {
        static std::map<std::string, std::function<T*(Args...)>> _register;
        return _register;
    }

    // Return a reference to the label registry. This allows us to look up the
    // label from a type.
    static auto& label_registry()
    {
        static std::map<std::type_index, std::string> _register;
        return _register;
    }

    // Get the name of the type by demangling
    static std::string typelabel()
    {
        int status;
        char * name = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
        std::string typelabel;

        if(status == 0) {
            typelabel = name;
        }
        else {
            typelabel = typeid(T).name();
        }

        std::free(name);
        return typelabel;
    }

};

// Internal use macros
#define _FACTORY_NAME(type, ...) FACTORY(type)
#define _TYPE_LABEL(type, ...) # type

// Public macros
/**
 * Get the factory for the given base class.
 *
 * @param type The type the factory will create.
 *
 * @returns The name of the factory class.
 **/
#define FACTORY(type) _factory_alias ## type

/**
 * Create the Factory for the base type.
 *
 * Should be called in the header where the baseclass is defined.
 *
//! @cond Doxygen_Suppress
 * @param class   Base class for factory.
 * @param args Types of arguments for constructor to use.
//! @endcond
 *
 * @note This will create an alias for the specialized factory class.
 **/
#define CREATE_FACTORY(...) \
    using _FACTORY_NAME(__VA_ARGS__) = Factory<__VA_ARGS__>;

/**
 * Register a subtype in the factory with a custom name.
 *
 * Should be called in the .cpp file of the subclass.
 *
 * @param type    Base class for factory.
 * @param subtype Sub class to register in factory.
 * @param name    Name to use for subtype.
 *
 * @note As a side effect This will assign zero to a static variable.
 *       The name is munged to avoid clashes.
 **/
#define REGISTER_NAMED_TYPE_WITH_FACTORY(type, subtype, name) \
    auto _register ## type ## subtype = \
         FACTORY(type)::register_type<subtype>(name);

/**
 * Register a subtype in the factory.
 *
 * Should be called in the .cpp file of the subclass.
 *
 * @param type    Base class for factory.
 * @param subtype Sub class to register in factory.
 *
 * @note As a side effect This will assign zero to a static variable.
 *       The name is munged to avoid clashes.
 **/
#define REGISTER_TYPE_WITH_FACTORY(type, subtype) \
    REGISTER_NAMED_TYPE_WITH_FACTORY(type, subtype, #subtype)

/**
 * Register a subtype in the factory using its RTTI name.
 *
 * Should be called in the .cpp file of the subclass.
 *
 * @param type    Base class for factory.
 * @param subtype Sub class to register in factory.
 *
 * @note As a side effect This will assign zero to a static variable.
 *       The name is munged to avoid clashes.
 **/
#define REGISTER_RTTI_TYPE_WITH_FACTORY(type, subtype) \
    REGISTER_NAMED_TYPE_WITH_FACTORY(type, subtype, typeid(subtype).name())

// clang-format on

#endif
