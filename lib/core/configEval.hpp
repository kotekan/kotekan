#ifndef CONFIG_EVAL_HPP
#define CONFIG_EVAL_HPP

#include "Config.hpp"
#include "errors.h"

#include <string>
#include <list>
#include <regex>
#include <exception>


/**
 * @brief Gets an arithmetic expression from the config.
 *
 * Treats the value as an arithmetic expression with other variables,
 * and return the result of evaluating it.
 * Matches and computes the following EBNF grammar:
 * EXP := ['+'|'-'] TERM {('+'|'-') TERM}
 * TERM := FACTOR {('*'|'/') FACTOR}
 * FACTOR := number | var | '(' EXP ')'
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

template <class Type>
configEval<Type>::configEval(Config &_config,
                             const std::string &base_path,
                             const std::string &name)
                             : config(_config), unique_name(base_path) {

    json value = config.get_value(base_path, name);

    if (!(value.is_string() || value.is_number())) {
        throw std::runtime_error("The value " + name + " in path " + base_path
                                 + " isn't a number or string to eval or " \
                                 "does not exist.");
    }
    const std::string &expression = value.get<string>();

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
configEval<Type>::~configEval() {
}

template <class Type>
Type configEval<Type>::compute_result() {
    return exp();
}

template <class Type>
void configEval<Type>::next() {
    tokens.pop_front();
    if (!tokens.empty()) {
        current_token = tokens.front();
    } else {
        current_token = "";
    }
}

template <class Type>
bool configEval<Type>::isNumber() {
    std::regex re( R"([0-9]*\.?[0-9]+)", std::regex::ECMAScript);
    std::cmatch m;
    return std::regex_match (tokens.front().c_str(), m, re);
}

template <class Type>
bool configEval<Type>::isVar() {
    std::regex re(R"([a-zA-Z][a-zA-Z0-9_]+)", std::regex::ECMAScript);
    std::cmatch m;
    return std::regex_match (tokens.front().c_str(), m, re);
}

template <class Type>
void configEval<Type>::expect(const std::string& symbol) {
    if (current_token == symbol) {
        next();
    } else {
        ERROR("Expected symbol %s, got %s",
                symbol.c_str(), tokens.front().c_str());
        throw std::runtime_error("Unexpected symbol");
    }
}

template <class Type>
Type configEval<Type>::exp() {
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
Type configEval<Type>::term() {
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
Type configEval<Type>::factor() {
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


#endif /* CONFIG_EVAL_HPP */

