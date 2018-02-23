#include "configEval.hpp"

int64_t eval_compute_int64(Config &config,
                           const std::string &unique_name,
                           const std::string &expression) {

    configEval<int64_t> eval(config, unique_name, expression);
    return eval.compute_restult();
}

double eval_compute_double(Config &config,
                           const std::string &unique_name,
                           const std::string &expression) {

    configEval<double> eval(config, unique_name, expression);
    return eval.compute_restult();
}