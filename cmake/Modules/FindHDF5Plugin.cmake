# cmake/FindHDF5Plugin.cmake
# Try multiple strategies in order of preference

# 1. Use the HDF5_PLUGIN_DIR CLI argument without question
if(HDF5_PLUGIN_DIR)
    set(HDF5Plugin_PLUGIN_DIR "${HDF5_PLUGIN_DIR}")
endif()

# 2. Respect any existing environment HDF5_PLUGIN_PATH
if(NOT HDF5Plugin_PLUGIN_DIR)
    if(DEFINED ENV{HDF5_PLUGIN_PATH})
        # First parse the env var in case it contains multiple paths
        cmake_path(CONVERT $ENV{HDF5_PLUGIN_PATH} TO_CMAKE_PATH_LIST ENV_HDF5_PLUGIN_PATH NORMALIZE)

        # Now search in those paths
        find_path(HDF5Plugin_PLUGIN_DIR
            NAMES libh5blosc.so libh5blosc.dylib libH5Zblosc.so h5blosc.dll
            PATHS
                ${ENV_HDF5_PLUGIN_PATH}
            PATH_SUFFIXES plugins hdf5/plugins
        )
    endif()
endif()

# 3. Try Python hdf5plugin, this is where the deployment plugins are.
if(NOT HDF5Plugin_PLUGIN_DIR)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "
try:
    import hdf5plugin
    print(hdf5plugin.PLUGINS_PATH)
except ImportError:
    import site, os
    for path in site.getsitepackages():
        plugin_path = os.path.join(path, 'hdf5plugin', 'plugins')
        if os.path.exists(plugin_path):
            print(plugin_path)
            break
except: pass"
            OUTPUT_VARIABLE _hdf5plugin_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_hdf5plugin_path AND EXISTS "${_hdf5plugin_path}")
            set(HDF5Plugin_PLUGIN_DIR "${_hdf5plugin_path}")
        endif()
    endif()
endif()

# 4. Look in likely system/conda/etc locations.
if(NOT HDF5Plugin_PLUGIN_DIR)
    find_path(HDF5Plugin_PLUGIN_DIR
        NAMES libh5blosc.so libh5blosc.dylib libH5Zblosc.so h5blosc.dll
        PATHS
            ${HDF5_ROOT}/lib/plugin
            /usr/lib/x86_64-linux-gnu/hdf5/plugins
            /usr/lib/x86_64-linux-gnu/hdf5/serial/plugins
            /usr/lib/hdf5/plugins
            /usr/local/lib/hdf5/plugins
            $ENV{CONDA_PREFIX}/lib/hdf5/plugins
            $ENV{MAMBA_ROOT_PREFIX}/lib/hdf5/plugins
        PATH_SUFFIXES plugins hdf5/plugins
    )
endif()
