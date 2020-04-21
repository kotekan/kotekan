# NOTE THAT THESE ARE HARDCODED FOR FIJI ARCHITECTURE! SHOULD ADD A FLAG OR SOMETHING TO ALLOW
# OTHERS.

# Find the HSACO compilers Find the AMD clang This should really test something about the clang to
# see if it is the AMD varient, and not just check a known path.
find_program(
    AMD_CLANG
    clang
    PATH
    /opt/rocm/hcc/bin/
    /opt/rocm/hcc-1.0/bin/
    /opt/amd/llvm/bin/
    NO_DEFAULT_PATH)
if(NOT AMD_CLANG)
    message(FATAL_ERROR "AMD clang not found!")
else()
    message("-- Found the AMD clang: ${AMD_CLANG}")
endif()

# find cloc.sh, if needed
if(${USE_CLOC})
    find_program(CLOC_SCRIPT cloc.sh)
    if(NOT CLOC_SCRIPT)
        message(FATAL_ERROR "The cloc.sh was not found!")
    else()
        message("-- Found the cloc.sh script: ${CLOC_SCRIPT}")
    endif()
endif()

# List of output files
set(OUT_FILES)

# Generate the hsaco files from .isa files.
file(
    GLOB inISAFiles
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.isa")

foreach(infileName ${inISAFiles})

    # Generate output file names
    string(REGEX REPLACE ".isa\$" ".hsaco" outfileName "${infileName}")
    string(REGEX REPLACE ".isa\$" ".o" objectfileName "${infileName}")
    set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${outfileName}")
    set(OBJECT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${objectfileName}")

    # Generate input file name
    set(IN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${infileName}")

    # Custom command to build the hsaco
    if(${USE_OLD_ROCM})
        add_custom_command(
            OUTPUT "${OUT_FILE}"
            COMMAND ${AMD_CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -c -o
                    "${OBJECT_FILE}" "${IN_FILE}"
            COMMAND ${AMD_CLANG} -target amdgcn--amdhsa "${OBJECT_FILE}" -o "${OUT_FILE}"
            DEPENDS "${IN_FILE}" ${AMD_CLANG}
            COMMENT "Generating hsaco file from ${IN_FILE}: ${OUT_FILE}")
    else()
        add_custom_command(
            OUTPUT "${OUT_FILE}"
            COMMAND ${AMD_CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -mno-code-object-v3
                    -c -o "${OBJECT_FILE}" "${IN_FILE}"
            COMMAND ${AMD_CLANG} -target amdgcn--amdhsa "${OBJECT_FILE}" -o "${OUT_FILE}"
            DEPENDS "${IN_FILE}" ${AMD_CLANG}
            COMMENT "Generating hsaco file from ${IN_FILE}: ${OUT_FILE}")
    endif()
    # Build list of depenencies
    set(OUT_FILES ${OUT_FILES} "${OUT_FILE}")
endforeach(infileName)

# Copy / generate the hsaco files from OpenCL .cl files
file(
    GLOB inCL_FILEs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cl")
foreach(clfile ${inCLFiles})
    if(${USE_CLOC})
        string(REGEX REPLACE ".cl\$" ".hsaco" HSACO_FILE "${clfile}")
        set(CL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${clfile}")
        set(HSACO_FILE "${CMAKE_CURRENT_BINARY_DIR}/${HSACO_FILE}")
        add_custom_command(
            OUTPUT "${HSACO_FILE}"
            COMMAND ${CLOC_SCRIPT} "-o" ${HSACO_FILE} ${CL_FILE}
            DEPENDS "${IN_FILE}" ${CLOC_SCRIPT}
            COMMENT "Using cloc.sh to compile hsaco file from ${CL_FILE}: ${HSACO_FILE}")
    else()
        string(REGEX REPLACE ".cl\$" ".bc" BC_FILE "${clfile}")
        string(REGEX REPLACE ".cl\$" ".lnkd.bc" LINKD_BC_FILE "${clfile}")
        string(REGEX REPLACE ".cl\$" ".final.bc" FINAL_BC_FILE "${clfile}")
        string(REGEX REPLACE ".cl\$" ".gcn" GCN_FILE "${clfile}")
        string(REGEX REPLACE ".cl\$" ".hsaco" HSACO_FILE "${clfile}")

        set(CL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${clfile}")
        set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BC_FILE}")
        set(LINKD_BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${LINKD_BC_FILE}")
        set(FINAL_BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FINAL_BC_FILE}")
        set(GCN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${GCN_FILE}")
        set(HSACO_FILE "${CMAKE_CURRENT_BINARY_DIR}/${HSACO_FILE}")

        set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${outfileName}")
        add_custom_command(
            OUTPUT "${HSACO_FILE}"
            COMMAND
                "/opt/rocm/opencl/bin/x86_64/clang" "-x" "cl" "-Xclang" "-cl-std=CL2.0" "-I"
                "/opt/rocm/opencl/include" "-include" "opencl-c.h"
                "-Dcl_clang_storage_class_specifiers" "-Dcl_khr_fp64" "-target" "amdgcn-amd-amdhsa"
                "-mcpu=gfx803" "-c" "-emit-llvm" "-o" ${BC_FILE} ${CL_FILE}
            COMMAND
                "/opt/rocm/opencl/bin/x86_64/llvm-link" ${BC_FILE}
                "/opt/rocm/opencl/lib/x86_64/bitcode/opencl.amdgcn.bc"
                "/opt/rocm/opencl/lib/x86_64/bitcode/ockl.amdgcn.bc"
                "/opt/rocm/opencl/lib/x86_64/bitcode/irif.amdgcn.bc"
                "/opt/rocm/opencl/lib/x86_64/bitcode/ocml.amdgcn.bc" "-o" ${LINKD_BC_FILE}
            COMMAND "/opt/rocm/opencl/bin/x86_64/opt" "-O2" "-mcpu=gfx803"
                    "-amdgpu-annotate-kernel-features" "-o" ${FINAL_BC_FILE} ${LINKD_BC_FILE}
            COMMAND "/opt/rocm/opencl/bin/x86_64/llc" "-mtriple" "amdgcn-amd-amdhsa" "-mcpu=gfx803"
                    "-filetype=obj" "-o" ${GCN_FILE} ${FINAL_BC_FILE}
            COMMAND "/opt/rocm/opencl/bin/x86_64/ld.lld" ${GCN_FILE} "--no-undefined" "-shared" "-o"
                    ${HSACO_FILE}
            DEPENDS ${CL_FILE}
            COMMENT "Compiling hsaco file from ${CL_FILE}: ${HSACO_FILE}")
    endif()
    set(OUT_FILES ${OUT_FILES} "${HSACO_FILE}")
endforeach()

# Build list of depenencies
add_custom_target(
    makeHSACOKernels ALL
    DEPENDS ${OUT_FILES}
    COMMENT "Building HSA kernels")

install(DIRECTORY DESTINATION /var/lib/kotekan/hsa_kernels/)
install(FILES ${OUT_FILES} DESTINATION /var/lib/kotekan/hsa_kernels/)
