# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Kotekan Developers

#****************************************************
# @file   enum_metadatas.py
# @brief  This is a helper script to gather all 
#         metadata header files and add them to 
#         allMetadata.hpp, as well as macros to 
#         register them in allMetadata.cpp
#
# @author Mehdi Najafi
# @date   09 SEP 2022
#****************************************************

import glob
print( 'Looking for any metadata header files:' )
file_cpp = open("allMetadata.cpp",'r').read()
file_cpp = file_cpp[0:file_cpp.find("#pragma once")+13]

file_hpp = open("allMetadata.hpp",'r').read()
file_hpp = file_hpp[0:file_hpp.find("#pragma once")+13]
for header in glob.glob('*.hpp'):
    if header != "allMetadata.hpp":
        print ('    ', header)
        file_hpp += '\n#include "' + header + '"'
        
        header_txt = open(header, 'r').read()
        pos = header_txt.find('REGISTER_KOTEKAN_METADATA')
        while pos >= 0:
            pos_e = header_txt.find(')', pos)
            txt = header_txt[pos:pos_e+1]
            file_cpp += '\n' +  txt.replace('REGISTER_KOTEKAN_METADATA', 'REGISTER_KOTEKAN_METADATA_ONCE')
            pos = header_txt.find('REGISTER_KOTEKAN_METADATA', pos + 25)

open("allMetadata.hpp",'w').write(file_hpp)
open("allMetadata.cpp",'w').write(file_cpp)
