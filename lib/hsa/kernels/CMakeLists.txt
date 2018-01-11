# Find the HSACO compilers
# Find the AMD clang
# This should really test something about the clang to
# see if it is the AMD varient, and not just check a known path.
find_program(AMD_CLANG clang /opt/rocm/hcc-1.0/bin/ /opt/amd/llvm/bin/)
if(NOT AMD_CLANG)
    message(FATAL_ERROR "AMD clang not found!")
else()
    message("-- Found the AMD clang: ${AMD_CLANG}")
endif()

# List of output files
SET(outFiles)

# Generate the hsaco files from .isa files.
FILE(GLOB inISAFiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.isa")

FOREACH(infileName ${inISAFiles})

    # Generate output file names
    STRING(REGEX REPLACE ".isa\$" ".hsaco" outfileName "${infileName}")
    STRING(REGEX REPLACE ".isa\$" ".o" objectfileName "${infileName}")
    SET(outfile "${CMAKE_CURRENT_BINARY_DIR}/${outfileName}")
    SET(objectfile "${CMAKE_CURRENT_BINARY_DIR}/${objectfileName}")

    # Generate input file name
    SET(infile "${CMAKE_CURRENT_SOURCE_DIR}/${infileName}")

    # Custom command to build the hasco
    ADD_CUSTOM_COMMAND(OUTPUT "${outfile}"
        COMMAND ${AMD_CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -c -o "${objectfile}" "${infile}"
        COMMAND ${AMD_CLANG} -target amdgcn--amdhsa "${objectfile}" -o "${outfile}"
        DEPENDS "${infile}" ${AMD_CLANG}
        COMMENT "Generating hasco file from ${infile}: ${outfile}")

    # Build list of depenencies
    SET(outFiles ${outFiles} "${outfile}")
ENDFOREACH(infileName)



# Generate / copy the hsaco files from OpenCL .cl files

if (${USE_PRECOMPIPLED_OPENCL})
  FILE(GLOB inCLFiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hsaco")
else()
  find_program(CLOC_SCRIPT cloc.sh)
  if(NOT CLOC_SCRIPT)
    message(FATAL_ERROR "The cloc.sh was not found!")
  else()
    message("-- Found the cloc.sh script: ${CLOC_SCRIPT}")
  endif()
  FILE(GLOB inCLFiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cl")
endif()

FOREACH(infileName ${inCLFiles})
    # Generate output file name
    STRING(REGEX REPLACE ".cl\$" ".hsaco" outfileName "${infileName}")
    SET(outfile "${CMAKE_CURRENT_BINARY_DIR}/${outfileName}")

    # Generate input file name
    SET(infile "${CMAKE_CURRENT_SOURCE_DIR}/${infileName}")

    # Custom command to build the hasco from the OpenCL
    if (${USE_PRECOMPIPLED_OPENCL})
      set(CL_UTIL "cp" "${infile}" "${outfile}")
      set(CL_COMMENT "Copying")
    else()
      set(CL_UTIL "${CLOC_SCRIPT} -o" "${outfile}" "${infile}")
      set(CL_COMMENT "Generating")
    endif()
    ADD_CUSTOM_COMMAND(OUTPUT "${outfile}"
        COMMAND ${CL_UTIL}
        DEPENDS "${infile}" ${CLOC_SCRIPT}
        COMMENT "${CL_COMMENT} hasco file from ${infile}: ${outfile}")

    # Build list of depenencies
    SET(outFiles ${outFiles} "${outfile}")
ENDFOREACH(infileName)

ADD_CUSTOM_TARGET(makeHSACOKernels ALL DEPENDS ${outFiles})
