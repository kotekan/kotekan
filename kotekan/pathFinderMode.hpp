/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   pathFinderMode.hpp
 * Author: iantretyakov
 *
 * Created on August 4, 2017, 11:36 AM
 */

#ifndef PATHFINDERMODE_HPP
#define PATHFINDERMODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

#ifdef WITH_HSA
    #include "hsaBase.h"
#endif /* WITH_HSA */

#ifdef WITH_OPENCL
    #ifdef __APPLE__
        #include "OpenCL/opencl.h"
    #else
        #include <CL/cl.h>
        #include <CL/cl_ext.h>
    #endif
#endif

class pathFinderMode : public kotekanMode {
public:
    pathFinderMode(Config &config);
    virtual ~pathFinderMode();
    
    void initalize_processes();
    
private:

};

#endif /* PATHFINDERMODE_HPP */

