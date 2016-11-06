#ifndef CHIME_SHUFFLE_HPP
#define CHIME_SHUFFLE_HPP

#include "kotekanMode.hpp"

class chimeShuffle : public kotekanMode {

public:
    chimeShuffle(Config &config);
    virtual ~chimeShuffle();

    void initalize_processes();

private:

};



#endif /* CHIME_HPP */
