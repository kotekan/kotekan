****************
Command Cookbook
****************

This page just contains frequently-used commands that are useful for accomplishing a certain task.


Running tests locally
*********************

The following can be used to compile and run most CI tests.

.. code-block:: bash

    cd build
    cmake -DWITH_TESTS=ON -DWITH_BOOST_TESTS=ON .. && make -j

    kotekan_env && export PYTHONPATH="$(cd ../python && pwd):${PYTHONPATH}" && pytest -v -n 8 -x ../tests/  && bash ../tests/boost/run_boost_tests.sh -t 30 tests/boost/  && bash ../config/ci-tests/run_tests.sh kotekan/kotekan 30s ../config/ci-tests/cpu_batch  && bash ../config/ci-tests/run_tests.sh kotekan/kotekan 60s ../config/ci-tests/gpu_batch


Generating Simulated F-Engine Data
**********************************


This command will set up the julia environment. (See also: details in the ``julia`` directory.)
In a shared environment, /tmp/install.sh may conflict.

.. code-block:: bash

    (cd julia && env PYTHON= julia --project=@. --eval 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall"); Pkg.update()')


The following are useful for setting up different build types. (Uses ninja, ``apt install ninja-build``).

.. code-block:: bash

    kotekan_env
    rm -rf cmake-build-superdebug
    cmake -Bcmake-build-superdebug -GNinja -DUSE_ASDF=ON -DUSE_CUDA=ON -DUSE_GDAL=ON -DUSE_HDF5=ON -DUSE_OMP=ON -DCOMPILE_DOCS=OFF -DWITH_BOOST_TESTS=ON -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=ON -DSUPERDEBUG=ON
    cmake --build cmake-build-superdebug

.. code-block:: bash

    kotekan_env
    rm -rf cmake-build-debug
    cmake -Bcmake-build-debug -GNinja -DUSE_ASDF=ON -DUSE_CUDA=ON -DUSE_GDAL=ON -DUSE_HDF5=ON -DUSE_OMP=ON -DCOMPILE_DOCS=ON -DWITH_BOOST_TESTS=ON -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
    cmake --build cmake-build-debug
    cmake --build cmake-build-debug --target doc

.. code-block:: bash

    kotekan_env
    rm -rf cmake-build-test
    cmake -Bcmake-build-test -GNinja -DUSE_ASDF=ON -DUSE_CUDA=ON -DUSE_GDAL=ON -DUSE_HDF5=ON -DUSE_OMP=ON -DCOMPILE_DOCS=OFF -DWITH_BOOST_TESTS=ON -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=Test
    cmake --build cmake-build-test

.. code-block:: bash

    kotekan_env
    rm -rf cmake-build-release
    cmake -Bcmake-build-release -GNinja -DUSE_ASDF=ON -DUSE_CUDA=ON -DUSE_GDAL=ON -DUSE_HDF5=ON -DUSE_OMP=ON -DCOMPILE_DOCS=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build cmake-build-release


The following will generate fake FEngine data and run tests with it.

.. code-block:: bash

    # Smallfinder
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_init_smallfinder && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/init_smallfinder.j2 2>&1 | tee data/fengine_init_smallfinder.log
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_test_smallfinder && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/test_smallfinder.j2 2>&1 | tee data/fengine_test_smallfinder.log

    # Pathfinder
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_init_pathfinder && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/init_pathfinder.j2 2>&1 | tee data/fengine_init_pathfinder.log
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_test_pathfinder && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/test_pathfinder.j2 2>&1 | tee data/fengine_test_pathfinder.log

    # CHORD
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_init_chord && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/init_chord.j2 2>&1 | tee data/fengine_init_chord.log
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_test_chord && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/test_chord.j2 2>&1 | tee data/fengine_test_chord.log

    # CHIME
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_init_chime && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/init_chime.j2 2>&1 | tee data/fengine_init_chime.log
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_test_chime && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/test_chime.j2 2>&1 | tee data/fengine_test_chime.log

    # HIRAX
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_init_hirax && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/init_hirax.j2 2>&1 | tee data/fengine_init_hirax.log
    cmake --build cmake-build-debug --target kotekan/kotekan && rm -rf data/fengine_test_hirax && env JULIA_PROJECT=julia JULIA_NUM_THREADS=32 stdbuf -oL -eL /usr/local/cuda/bin/compute-sanitizer --print-limit 0 ./cmake-build-debug/kotekan/kotekan --bind-address localhost:23023 --config config/fengine/test_hirax.j2 2>&1 | tee data/fengine_test_hirax.log

