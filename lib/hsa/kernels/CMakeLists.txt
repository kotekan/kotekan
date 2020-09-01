# NOTE THAT THESE ARE HARDCODED FOR FIJI ARCHITECTURE!
# SHOULD ADD A FLAG OR SOMETHING TO ALLOW OTHERS.


# Find the HSACO compilers
# Find the AMD clang
# This should really test something about the clang to
# see if it is the AMD varient, and not just check a known path.
find_program(AMD_CLANG clang PATH /opt/rocm/hcc/bin/ /opt/rocm/hcc-1.0/bin/ /opt/amd/llvm/bin/ NO_DEFAULT_PATH)
if(NOT AMD_CLANG)
    message(FATAL_ERROR "AMD clang not found!")
else()
    message("-- Found the AMD clang: ${AMD_CLANG}")
endif()

#find cloc.sh, if needed
if(${USE_CLOC})
  find_program(CLOC_SCRIPT cloc.sh)
  if(NOT CLOC_SCRIPT)
    message(FATAL_ERROR "The cloc.sh was not found!")
  else()
    message("-- Found the cloc.sh script: ${CLOC_SCRIPT}")
  endif()
endif ()


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

    # Custom command to build the hsaco
    if (${USE_OLD_ROCM})
      ADD_CUSTOM_COMMAND(OUTPUT "${outfile}"
        COMMAND ${AMD_CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -c -o "${objectfile}" "${infile}"
        COMMAND ${AMD_CLANG} -target amdgcn--amdhsa "${objectfile}" -o "${outfile}"
        DEPENDS "${infile}" ${AMD_CLANG}
        COMMENT "Generating hsaco file from ${infile}: ${outfile}")
    else ()
      ADD_CUSTOM_COMMAND(OUTPUT "${outfile}"
        COMMAND ${AMD_CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -mno-code-object-v3 -c -o "${objectfile}" "${infile}"
        COMMAND ${AMD_CLANG} -target amdgcn--amdhsa "${objectfile}" -o "${outfile}"
        DEPENDS "${infile}" ${AMD_CLANG}
        COMMENT "Generating hsaco file from ${infile}: ${outfile}")
    endif ()
    # Build list of depenencies
    SET(outFiles ${outFiles} "${outfile}")
ENDFOREACH(infileName)


# Copy / generate the hsaco files from OpenCL .cl files
FILE(GLOB inCLFiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
                             "${CMAKE_CURRENT_SOURCE_DIR}/*.cl")
FOREACH(clfile ${inCLFiles})
  if (${USE_CLOC})
    STRING(REGEX REPLACE ".cl\$" ".hsaco" hsacofile "${clfile}")
    SET(clfile      "${CMAKE_CURRENT_SOURCE_DIR}/${clfile}")
    SET(hsacofile   "${CMAKE_CURRENT_BINARY_DIR}/${hsacofile}")
    ADD_CUSTOM_COMMAND(OUTPUT "${hsacofile}"
        COMMAND ${CLOC_SCRIPT} "-o" ${hsacofile} ${clfile}
        DEPENDS "${infile}" ${CLOC_SCRIPT}
        COMMENT "Using cloc.sh to compile hsaco file from ${clfile}: ${hsacofile}")
  else ()
    STRING(REGEX REPLACE ".cl\$" ".bc"        bcfile      "${clfile}")
    STRING(REGEX REPLACE ".cl\$" ".lnkd.bc"   lnkdbcfile  "${clfile}")
    STRING(REGEX REPLACE ".cl\$" ".final.bc"  finalbcfile "${clfile}")
    STRING(REGEX REPLACE ".cl\$" ".gcn"       gcnfile     "${clfile}")
    STRING(REGEX REPLACE ".cl\$" ".hsaco"     hsacofile   "${clfile}")
    STRING(REGEX REPLACE ".cl\$" ".isa"       isafile     "${clfile}")

    SET(clfile      "${CMAKE_CURRENT_SOURCE_DIR}/${clfile}")
    SET(bcfile      "${CMAKE_CURRENT_BINARY_DIR}/${bcfile}")
    SET(lnkdbcfile  "${CMAKE_CURRENT_BINARY_DIR}/${lnkdbcfile}")
    SET(finalbcfile "${CMAKE_CURRENT_BINARY_DIR}/${finalbcfile}")
    SET(gcnfile     "${CMAKE_CURRENT_BINARY_DIR}/${gcnfile}")
    SET(hsacofile   "${CMAKE_CURRENT_BINARY_DIR}/${hsacofile}")
    SET(isafile   "${CMAKE_CURRENT_BINARY_DIR}/${isafile}")

    SET(outfile "${CMAKE_CURRENT_BINARY_DIR}/${outfileName}")
    ADD_CUSTOM_COMMAND(OUTPUT "${hsacofile}"
        COMMAND "/opt/rocm/opencl/bin/x86_64/clang"
                "-x" "cl"
                "-Xclang"
                "-cl-std=CL2.0"
                "-I" "/opt/rocm/opencl/include"
                "-include" "opencl-c.h"
                "-Dcl_clang_storage_class_specifiers"
                "-Dcl_khr_fp64"
                "-target" "amdgcn-amd-amdhsa"
                "-mcpu=gfx803"
                "-c" "-emit-llvm"
                "-o" ${bcfile}
                ${clfile}
        COMMAND "/opt/rocm/opencl/bin/x86_64/llvm-link"
                ${bcfile}
                "/opt/rocm/opencl/lib/x86_64/bitcode/opencl.amdgcn.bc"
                "/opt/rocm/opencl/lib/x86_64/bitcode/ockl.amdgcn.bc"
                "/opt/rocm/opencl/lib/x86_64/bitcode/irif.amdgcn.bc"
                "/opt/rocm/opencl/lib/x86_64/bitcode/ocml.amdgcn.bc"
                "-o" ${lnkdbcfile}
        COMMAND "/opt/rocm/opencl/bin/x86_64/opt"
                "-O2"
                "-mcpu=gfx803"
                "-amdgpu-annotate-kernel-features"
                "-o" ${finalbcfile}
                ${lnkdbcfile}
        COMMAND "/opt/rocm/opencl/bin/x86_64/llc"
                "-mtriple" "amdgcn-amd-amdhsa"
                "-mcpu=gfx803"
                "-filetype=obj"
                "-o" ${gcnfile}
                ${finalbcfile}
        COMMAND "/opt/rocm/opencl/bin/x86_64/ld.lld"
                ${gcnfile}
                "--no-undefined"
                "-shared"
                "-o" ${hsacofile}
        COMMAND "/opt/rocm/opencl/bin/x86_64/llvm-objdump"
                "-disassemble"
                "-mcpu=fiji"
                ${hsacofile}
                ">" ${isafile}
        DEPENDS ${clfile}
        COMMENT "Compiling hsaco file from ${clfile}: ${hsacofile}"
      )
  endif ()
  SET(outFiles ${outFiles} "${hsacofile}")
ENDFOREACH(clfile)

# Build list of depenencies
ADD_CUSTOM_TARGET(makeHSACOKernels ALL DEPENDS ${outFiles})

INSTALL(DIRECTORY DESTINATION /var/lib/kotekan/hsa_kernels/)
INSTALL(FILES ${outFiles} DESTINATION /var/lib/kotekan/hsa_kernels/)
