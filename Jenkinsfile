pipeline {
  agent any
  options {
    timeout(time: 1, unit: 'HOURS')
    disableConcurrentBuilds()
  }
  stages {
    stage('Build') {
      parallel {
        stage('Build kotekan without hardware specific options') {
          steps {
            sh '''cd build/
                  cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive \
                  -DOPENBLAS_PATH=/opt/OpenBLAS/build/ -DUSE_LAPACK=ON -DBLAZE_PATH=/opt/blaze \
                  -DUSE_OMP=ON -DBOOST_TESTS=ON ..
                  make -j 4'''
          }
        }
        stage('Build CHIME kotekan') {
          steps {
            sh '''cd build/
                  cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ \
                  -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON \
                  -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive \
                  -DOPENBLAS_PATH=/opt/OpenBLAS/build/ -DUSE_LAPACK=ON -DBLAZE_PATH=/opt/blaze \
                  -DUSE_OMP=ON -DBOOST_TESTS=ON ..
                  make -j 4'''
          }
        }
        stage('Build base kotekan') {
          steps {
            sh '''mkdir build_base
                  cd build_base
                  cmake ..
                  make -j 4'''
          }
        }
        /* stage('Build MacOS kotekan') {
          agent {label 'macos'}
          steps {
            sh '''export PATH=${PATH}:/usr/local/bin/
                  mkdir build_base
                  cd build_base/
                  cmake ..
                  make -j 4
                  cd ..
                  mkdir build_full
                  cd build_full/
                  cmake -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DUSE_FFTW=ON -DUSE_AIRSPY=ON \
                        -DUSE_LAPACK=ON -DBLAZE_PATH=/usr/local/opt/blaze \
                        -DOPENBLAS_PATH=/usr/local/opt/OpenBLAS \
                        -DUSE_HDF5=ON -DHIGHFIVE_PATH=/usr/local/opt/HighFive \
                        -DCOMPILE_DOCS=ON -DUSE_OPENCL=ON ..
                  make -j 4'''
          }
        } */
        stage('Build docs') {
          steps {
            sh '''export PATH=${PATH}:/var/lib/jenkins/.local/bin/
                  mkdir build-docs
                  cd build-docs/
                  cmake -DCOMPILE_DOCS=ON -DPLANTUML_PATH=/opt/plantuml/ ..
                  cd docs/
                  make -j 4'''
          }
        }
        stage('Check code formatting') {
          steps {
            sh '''mkdir build-check-format
                  cd build-check-format/
                  cmake ..
                  make clang-format
                  git diff --exit-code'''
          }
        }
      }
    }
    stage('Unit Tests') {
      steps {
        sh '''cd tests/
              PYTHONPATH=../python/ pytest -x -s -vvv
              cd ../build/tests/
              PYTHONPATH=../python/ pytest -x -s -vvv'''
      }
    }
  }
}
