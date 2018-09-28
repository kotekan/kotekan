/*****************************************
@file
@brief Class to get values from and update the running config
- Config
*****************************************/
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <list>
#include <regex>
#include <vector>
#include <exception>
#include <cxxabi.h>
#include <type_traits>

#include "json.hpp"
#include "errors.h"

// Name space includes.
using json = nlohmann::json;
using std::string;
using std::vector;

/**
 * @class Config
 * @brief Access the running config.
 *
 * Provides access to values from the running config and allows to update the
 * config.
 *
 * @author Andre Renard
 */
class Config {
public:
    Config();
    Config(const Config& orig);
    virtual ~Config();

    /**
     * @brief Get a config value for a simple arithmetic type.
     *
     * This function is used if the requested type T is arithmetic type (int or float/double type)
     * and the value in the config isn't an arithmetic value, then this function will
     * attempt to evaluate it as an arithmetic expression using other values
     * declared in the config, or any constants in the expression.
     *
     * @param base_path Path to the value in the config.
     * @param name      Name of the value.
     * @return  The requested value.
     */
    template <class T,
          typename std::enable_if<std::is_arithmetic<T>::value,
          T>::type* = nullptr>
    T get(const string& base_path, const string& name) {
        json json_value = get_value(base_path, name);
        T value;
        try {
            // If the expected type is a number and the value
            // isn't already a number then try using the configEval parser
            if (std::is_arithmetic<T>::value &&
                    !std::is_same<bool, T>::value &&
                    !json_value.is_number()) {

                std::string expression = json_value.get<std::string>();
                try {
                    Config::configEval<T> eval(*this, base_path, name);
                    value = eval.compute_result();
                } catch (std::exception const & ex) {
                    throw std::runtime_error("Failed to evaluate: '" + json_value.get<std::string>()
                                         + "' with message: '" + ex.what() + "' for name " + name +
                                         " in path " + base_path);
                }
            } else {
                value = json_value.get<T>();
            }
        } catch (std::exception const & ex) {
            int status;
            throw std::runtime_error(
                        "The value " + name + " in path " + base_path
                        + " is not of type '" +
                        abi::__cxa_demangle(typeid(T).name(), NULL, NULL, &status)
                        + "' or doesn't exist");
        }

        return value;
    }

    /**
     * @brief Get a config value for a non-arithmetic type (i.e. string or vector)
     * @param base_path Path to the value in the config.
     * @param name      Name of the value.
     * @return  The requested value.
     */
    template <class T,
            typename std::enable_if<!std::is_arithmetic<T>::value,
            T>::type* = nullptr>
    T get(const string& base_path, const string& name) {
        json json_value = get_value(base_path, name);
        T value;
        try {
            value = json_value.get<T>();
        } catch (std::exception const & ex) {
            int status;
            throw std::runtime_error(
                        "The value " + name + " in path " + base_path
                        + " is not of type '" +
                        abi::__cxa_demangle(typeid(T).name(), NULL, NULL, &status)
                        + "' or doesn't exist");
        }

        return value;
    }

    /**
     * @brief Get a config value or return the default value.
     *
     * Same as get_int, but if it cannot find the value
     * (or if it has the wrong type), it returns `default_value`.
     * @param base_path     Path to the value in the config.
     * @param name          Name of the value.
     * @param default_value The default value.
     * @return  The value requested or the default value.
     */
    template<typename T>
    T get_default(const string& base_path, const string& name, T default_value);

    /**
     * @brief Checks if a value exists at the given location "base_path" + "name"
     * 
     * @param base_path JSON pointer string were the value should be serched for.
     * @param name The name of the value (the key)
     * @return true if the key exists in the path, and false otherwise.
     */
    bool exists(const string& base_path, const string& name);

    /**
     * @brief Reads the config from a JSON file.
     *
     * @param file_name The file containing the JSON config.
     */
    void parse_file(const string &file_name);

    /**
     * @brief Updates the config with a new JSON string
     *
     * @param updates Json object with values to be replaced.
     */
    void update_config(json updates);

    // This function should be moved, it doesn't really belong here...
    int32_t num_links_per_gpu(const int32_t &gpu_id);

    // @breaf Finds the value with key "name" starts looking at the
    // "base_pointer" location, and then works backwards up the config tree.
    // @param base_pointer Contains a JSON pointer which points to the
    // process's location in the config tree. i.e. /vdif_cap/disk_write
    // @param name The name of the property i.e. num_frequencies
    json get_value(const string &base_pointer, const string &name);

    /**
     * @brief Updates a config value at an existing config option
     *
     * Only accepts updates to known config locations, and should only be used
     * to update config within a processes' own @c unique_name path.
     *
     * @todo This currently will not update inherited config options.
     *       So any updatable config options must be given in the process block.
     *       The way around this is likely to copy inherited, either on access or
     *       on update with this function.
     *
     * @todo Currently this will take any type that can translate to json,
     *       we should likely make this a bit more restrictive and/or require
     *       the update type matches the base value type already in the config.
     *
     * @param base_path The unique name of the process, or "/" for the root process
     * @param name The name of the value
     * @param value The value to assign to the json pointer formed by base_path/name
     */
    template <typename T>
    void update_value(const string &base_path, const string &name, const T &value);

#ifdef WITH_SSL
    /**
     * @brief Generate an MD5 hash of the config
     *
     * The HASH is based on the json string dump with no spaces or newlines
     *
     * Only avaibile if OpenSSL was installed on a system,
     * so wrap any uses in @c #ifdef WITH_SSL
     *
     * @return The MD5sum as 32 char hex std::string
     */
    std::string get_md5sum();
#endif

    /**
     * @brief Returns the full json data structure (for internal framework use)
     * @warn This shouldn't be called outside of the core freamwork
     * @return A reference to the full JSON
     */
    json &get_full_config_json();

    /**
     * @brief Dumps the config to INFO in JSON format.
     */
    void dump_config();
private:

    /// Internal json object
    json _json;

    /**
     * @brief Helper class, gets an arithmetic expression from the config.
     *
     * Treats the value as an arithmetic expression with other variables,
     * and return the result of evaluating it.
     * Matches and computes the following EBNF grammar:
     * EXP := ['+'|'-'] TERM {('+'|'-') TERM}
     * TERM := FACTOR {('*'|'/') FACTOR}
     * FACTOR := number | var | '(' EXP ')'
     *
     * @author Andre Renard
     */
    template <class Type>
    class configEval {

    public:
        configEval(Config &_config, const std::string &base_path,
                const std::string &name);

        ~configEval();

        Type compute_result();

    private:
        Config &config;
        std::string unique_name;

        bool isNumber();
        bool isVar();

        void next();
        void expect(const std::string &symbol);
        Type factor();
        Type term();
        Type exp();

        std::list<std::string> tokens;
        std::string current_token = "";
    };

};

template <typename T>
void Config::update_value(const string &base_path, const string &name,
                          const T &value) {
    string update_path = base_path + "/" + name;
    json::json_pointer path(update_path);

    try {
        _json.at(path) = value;
    } catch (std::exception const & ex) {
        throw std::runtime_error("Failed to update config value at: "
                                 + update_path + " message: " + ex.what());
    }
}

template<typename T>
T Config::get_default(const string& base_path, const string& name,
                         T default_value) {
    try {
        T value = get<T>(base_path, name);
        return value;
    } catch (std::runtime_error const & ex) {
        return default_value;
    }
}

template <class Type>
Config::configEval<Type>::configEval(Config &_config,
                             const std::string &base_path,
                             const std::string &name)
                             : config(_config), unique_name(base_path) {

    json value = config.get_value(base_path, name);

    if (!(value.is_string() || value.is_number())) {
        throw std::runtime_error("The value " + name + " in path " + base_path
                                 + " isn't a number or string to eval or " \
                                 "does not exist.");
    }
    const std::string &expression = value.get<std::string>();

    static const std::regex re(
        R"(([0-9]*\.?[0-9]+|\+|\*|\-|\/|\)|\(|[a-zA-Z][a-zA-Z0-9_]+))",
        std::regex::ECMAScript);

    tokens = {
        std::sregex_token_iterator(expression.begin(), expression.end(), re, 1),
        std::sregex_token_iterator()
    };

    if (!tokens.empty())
        current_token = tokens.front();
}

template <class Type>
Config::configEval<Type>::~configEval() {
}

template <class Type>
Type Config::configEval<Type>::compute_result() {
    return exp();
}

template <class Type>
void Config::configEval<Type>::next() {
    tokens.pop_front();
    if (!tokens.empty()) {
        current_token = tokens.front();
    } else {
        current_token = "";
    }
}

template <class Type>
bool Config::configEval<Type>::isNumber() {
    std::regex re( R"([0-9]*\.?[0-9]+)", std::regex::ECMAScript);
    std::cmatch m;
    return std::regex_match (tokens.front().c_str(), m, re);
}

template <class Type>
bool Config::configEval<Type>::isVar() {
    std::regex re(R"([a-zA-Z][a-zA-Z0-9_]+)", std::regex::ECMAScript);
    std::cmatch m;
    return std::regex_match (tokens.front().c_str(), m, re);
}

template <class Type>
void Config::configEval<Type>::expect(const std::string& symbol) {
    if (current_token == symbol) {
        next();
    } else {
        ERROR("Expected symbol %s, got %s",
                symbol.c_str(), tokens.front().c_str());
        throw std::runtime_error("Unexpected symbol");
    }
}

template <class Type>
Type Config::configEval<Type>::exp() {
    Type ret = 0;
    if (current_token == "+" || current_token == "-") {
        if (current_token == "-") {
            next();
            ret = -term();
        } else {
            next();
            ret = term();
        }
    } else {
        ret = term();
    }
    while (current_token == "+" || current_token == "-") {
        if (current_token == "+") {
            next();
            ret += term();
        } else {
            next();
            ret -= term();
        }
    }
    return ret;
}

template <class Type>
Type Config::configEval<Type>::term() {
    Type ret = factor();
    while (current_token == "*" || current_token == "/") {
        if (current_token == "*") {
            next();
            ret *= factor();
        }
        if (current_token == "/") {
            // TODO Check for divide by zero.
            next();
            ret /= factor();
        }
    }
    return ret;
}

template <class Type>
Type Config::configEval<Type>::factor() {
    Type ret;

    if (isVar()) {
        ret = config.get<Type>(unique_name, current_token);
        next();
    } else if (isNumber()) {
        ret = (Type)stod(current_token);
        next();
    } else if (current_token == "(") {
        next();
        ret = exp();
        expect(")");
    } else {
        ERROR("Unexpected symbol '%s'", current_token.c_str());
        throw std::runtime_error("Unexpected symbol");
    }
    return ret;
}

#endif /* CONFIG_HPP */
