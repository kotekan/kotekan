// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
* @file AssembledChimeMetadata.hpp
* @brief This file declares the assembled version of 
*        chimeMetadata structure, registered before.

* @author Mehdi Najafi
* @date   28 AUG 2022
*****************************************************/

#include "chimeMetadata.hpp"
#include "FrequencyAssembledMetadata.hpp"
typedef FrequencyAssembledMetadata<chimeMetadata> assembledChimeMetadata;

// register this new metadata
REGISTER_KOTEKAN_METADATA(assembledChimeMetadata)
