/*****************************************
@file
@brief Class to get values from and update the running config
- Config
*****************************************/
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "kotekanLogging.hpp" // for ERROR_NON_OO

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json

#include <complex>     // for complex  // IWYU pragma: keep
#include <cxxabi.h>    // for __cxa_demangle
#include <exception>   // for exception
#include <list>        // for list
#include <regex>       // for regex, cmatch, regex_match, sregex_token_iterator
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for int32_t
#include <string>      // for string, operator==, allocator, stod
#include <type_traits> // for is_arithmetic, enable_if, is_same
#include <typeinfo>    // for type_info
#include <vector>      // for vector


namespace kotekan {

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
    template<class T, typename std::enable_if<std::is_arithmetic<T>::value, T>::type* = nullptr>
    T get(const std::string& base_path, const std::string& name) const {
        nlohmann::json json_value = get_value(base_path, name);
        T value;
        try {
            // If the expected type is a number and the value
            // isn't already a number then try using the configEval parser
            if (std::is_arithmetic<T>::value && !std::is_same<bool, T>::value
                && !json_value.is_number()) {

                try {
                    Config::configEval<T> eval(*this, base_path, name);
                    value = eval.compute_result();
                } catch (std::exception const& ex) {
                    throw std::runtime_error(
                        fmt::format(fmt("Failed to evaluate: '{:s}' with message: '{:s}' for name "
                                        "{:s} in path {:s}"),
                                    json_value.get<std::string>(), ex.what(), name, base_path));
                }
            } else {
                value = json_value.get<T>();
            }
        } catch (std::exception const& ex) {
            int status;
            throw std::runtime_error(fmt::format(
                fmt("The value {:s} in path {:s} is not of type '{:s}' or doesn't exist."), name,
                base_path, abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status)));
        }

        return value;
    }

    /**
     * @brief Get a config value for a non-arithmetic type (i.e. string or vector)
     * @param base_path Path to the value in the config.
     * @param name      Name of the value.
     * @return  The requested value.
     */
    template<class T, typename std::enable_if<!std::is_arithmetic<T>::value, T>::type* = nullptr>
    T get(const std::string& base_path, const std::string& name) const {
        nlohmann::json json_value = get_value(base_path, name);
        T value;
        try {
            value = json_value.get<T>();
        } catch (std::exception const& ex) {
            int status;
            throw std::runtime_error(fmt::format(
                fmt("The value {:s} in path {:s} is not of type '{:s}' or doesn't exist"), name,
                base_path, abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status)));
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
    T get_default(const std::string& base_path, const std::string& name, T default_value) const;

    /**
     * @brief Checks if a value exists at the given location "base_path" + "name"
     *
     * @param base_path JSON pointer string were the value should be searched for.
     * @param name The name of the value (the key)
     * @return true if the key exists in the path, and false otherwise.
     */
    bool exists(const std::string& base_path, const std::string& name) const;

    /**
     * @brief Reads the config from a JSON file.
     *
     * @param file_name The file containing the JSON config.
     */
    void parse_file(const std::string& file_name);

    /**
     * @brief Updates the config with a new JSON string
     *
     * @param updates Json object with values to be replaced.
     */
    void update_config(nlohmann::json updates);

    // This function should be moved, it doesn't really belong here...
    int32_t num_links_per_gpu(const int32_t& gpu_id) const;

    /**
     * @brief Finds the value with key "name" starts looking at the
     * "base_pointer" location, and then works backwards up the config tree.
     *
     * @throws  std::runtime_error  If the value was not found.
     *
     * @param base_pointer  Contains a JSON pointer which points to the
     *                      stage's location in the config tree. i.e.
     *                      /vdif_cap/disk_write
     * @param name          The name of the property i.e. num_frequencies
     *
     * @return              The value that was found.
     **/
    nlohmann::json get_value(const std::string& base_pointer, const std::string& name) const;

    /**
     * @brief Finds all values with key "name". Searches the whole config tree.
     *
     * @note This should only be used by internal (core framework) systems.
     * Usage by normal stages risks unexpected side effects in the config
     * scoping logic.
     *
     * @param name  The name of the property i.e. num_frequencies
     *
     * @return      The values found or an empty list if nothing was found.
     **/
    std::vector<nlohmann::json> get_value(const std::string& name) const;

    /**
     * @brief Updates a config value at an existing config option
     *
     * Only accepts updates to known config locations, and should only be used
     * to update config within a stages' own @c unique_name path.
     *
     * @todo This currently will not update inherited config options.
     *       So any updatable config options must be given in the stage block.
     *       The way around this is likely to copy inherited, either on access or
     *       on update with this function.
     *
     * @todo Currently this will take any type that can translate to json,
     *       we should likely make this a bit more restrictive and/or require
     *       the update type matches the base value type already in the config.
     *
     * @param base_path The unique name of the stage, or "/" for the root stage
     * @param name The name of the value
     * @param value The value to assign to the json pointer formed by base_path/name
     */
    template<typename T>
    void update_value(const std::string& base_path, const std::string& name, const T& value);

#ifdef WITH_SSL
    /**
     * @brief Generate an MD5 hash of the config
     *
     * The HASH is based on the json string dump with no spaces or newlines
     *
     * Only available if OpenSSL was installed on a system,
     * so wrap any uses in @c #ifdef WITH_SSL
     *
     * @return The MD5sum as 32 char hex std::string
     */
    std::string get_md5sum() const;
#endif

    /**
     * @brief Returns the full json data structure (for internal framework use)
     * @warning This shouldn't be called outside of the core framework
     * @return A reference to the full JSON
     */
    const nlohmann::json& get_full_config_json() const;

    /**
     * @brief Dumps the config to INFO in JSON format.
     */
    void dump_config() const;

private:
    /// Internal json object
    nlohmann::json _json;

    /**
     * @brief Finds all values with key "name". Searches the given json.
     *
     * @param j         The json to look in.
     * @param name      The name of the property i.e. num_frequencies
     * @param results   Vector found values are added to.
     **/
    void get_value_recursive(const nlohmann::json& j, const std::string& name,
                             std::vector<nlohmann::json>& results) const;

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
    template<class Type>
    class configEval {

    public:
        configEval(const Config& _config, const std::string& base_path, const std::string& name);

        ~configEval();

        Type compute_result();

    private:
        const Config& config;
        std::string unique_name;

        bool isNumber();
        bool isVar();

        void next();
        void expect(const std::string& symbol);
        Type factor();
        Type term();
        Type exp();

        std::list<std::string> tokens;
        std::string current_token = "";
    };
};

template<typename T>
void Config::update_value(const std::string& base_path, const std::string& name, const T& value) {
    std::string update_path = fmt::format(fmt("{:s}/{:s}"), base_path, name);
    nlohmann::json::json_pointer path(update_path);

    try {
        _json.at(path) = value;
    } catch (std::exception const& ex) {
        throw std::runtime_error(fmt::format(
            fmt("Failed to update config value at: {:s} message: {:s}"), update_path, ex.what()));
    }
}

template<typename T>
T Config::get_default(const std::string& base_path, const std::string& name,
                      T default_value) const {
    try {
        T value = get<T>(base_path, name);
        return value;
    } catch (std::runtime_error const& ex) {
        return default_value;
    }
}

template<class Type>
Config::configEval<Type>::configEval(const Config& _config, const std::string& base_path,
                                     const std::string& name) :
    config(_config), unique_name(base_path) {

    nlohmann::json value = config.get_value(base_path, name);

    if (!(value.is_string() || value.is_number())) {
        throw std::runtime_error(fmt::format(
            fmt("The value {:s} in path {:s} isn't a number or string to eval or does not exist."),
            name, base_path));
    }
    const std::string& expression = value.get<std::string>();

    static const std::regex re(
        R"((-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+\-]?[0-9]+)?|\+|\*|\-|\/|\)|\(|[a-zA-Z][a-zA-Z0-9_]*))",
        std::regex::ECMAScript);

    tokens = {std::sregex_token_iterator(expression.begin(), expression.end(), re, 1),
              std::sregex_token_iterator()};

    if (!tokens.empty())
        current_token = tokens.front();
}

template<class Type>
Config::configEval<Type>::~configEval() {}

template<class Type>
Type Config::configEval<Type>::compute_result() {
    Type result = exp();
    if (current_token != "") {
        std::string error_msg = fmt::format("Unexpected symbol: {:s}", current_token);
        ERROR_NON_OO("{:s}", error_msg);
        throw std::runtime_error(error_msg);
    }
    return result;
}

template<class Type>
void Config::configEval<Type>::next() {
    tokens.pop_front();
    if (!tokens.empty()) {
        current_token = tokens.front();
    } else {
        current_token = "";
    }
}

template<class Type>
bool Config::configEval<Type>::isNumber() {
    std::regex re(R"(-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+\-]?[0-9]+)?)",
                  std::regex::ECMAScript);
    std::cmatch m;
    return std::regex_match(tokens.front().c_str(), m, re);
}

template<class Type>
bool Config::configEval<Type>::isVar() {
    std::regex re(R"([a-zA-Z][a-zA-Z0-9_]*)", std::regex::ECMAScript);
    std::cmatch m;
    return std::regex_match(tokens.front().c_str(), m, re);
}

template<class Type>
void Config::configEval<Type>::expect(const std::string& symbol) {
    if (current_token == symbol) {
        next();
    } else {
        ERROR_NON_OO("Expected symbol {:s}, got {:s}", symbol, tokens.front());
        throw std::runtime_error("Unexpected symbol");
    }
}

template<class Type>
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

template<class Type>
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

template<class Type>
Type Config::configEval<Type>::factor() {
    Type ret;

    if (current_token == "") {
        ERROR_NON_OO("Expected another value/symbol in expression but found none");
        throw std::runtime_error("Expected another value/symbol in expression but found none");
    } else if (isVar()) {
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
        ERROR_NON_OO("Unexpected symbol '{:s}'", current_token);
        throw std::runtime_error("Unexpected symbol");
    }
    return ret;
}

// Tell the compiler that all those are instantiated in Config.cpp,
// so that they are not built inline everywhere they are used
// (would add >60MB to the binary).
extern template float Config::get(const std::string& base_path, const std::string& name) const;
extern template double Config::get(const std::string& base_path, const std::string& name) const;
extern template uint32_t Config::get(const std::string& base_path, const std::string& name) const;
extern template uint64_t Config::get(const std::string& base_path, const std::string& name) const;
extern template int32_t Config::get(const std::string& base_path, const std::string& name) const;
extern template int16_t Config::get(const std::string& base_path, const std::string& name) const;
extern template uint16_t Config::get(const std::string& base_path, const std::string& name) const;
extern template bool Config::get(const std::string& base_path, const std::string& name) const;
extern template std::string Config::get(const std::string& base_path,
                                        const std::string& name) const;
extern template std::vector<int32_t> Config::get(const std::string& base_path,
                                                 const std::string& name) const;
extern template std::vector<uint32_t> Config::get(const std::string& base_path,
                                                  const std::string& name) const;
extern template std::vector<float> Config::get(const std::string& base_path,
                                               const std::string& name) const;
extern template std::vector<std::string> Config::get(const std::string& base_path,
                                                     const std::string& name) const;
extern template std::vector<nlohmann::json> Config::get(const std::string& base_path,
                                                        const std::string& name) const;

} // namespace kotekan

#endif /* CONFIG_HPP */
