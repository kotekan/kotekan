# cmake/FindHDF5Plugin.cmake
# Try multiple strategies in order of preference
find_path(HDF5Plugin_PLUGIN_DIR
    NAMES libh5blosc.so libh5blosc.dylib h5blosc.dll
    PATHS
        # 1. Respect any existing HDF5_PLUGIN_PATH
        $ENV{HDF5_PLUGIN_PATH}
        # 2. System locations
        ${HDF5_ROOT}/lib/plugin
        /usr/lib/x86_64-linux-gnu/hdf5/plugins
        /usr/lib/hdf5/plugins
        /usr/local/lib/hdf5/plugins
        # 3. Common conda/mamba locations
        $ENV{CONDA_PREFIX}/lib/hdf5/plugins
        $ENV{MAMBA_ROOT_PREFIX}/lib/hdf5/plugins
    PATH_SUFFIXES plugins hdf5/plugins
)

# 4. Try Python hdf5plugin as fallback
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
