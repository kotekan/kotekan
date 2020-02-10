pipeline {
  agent any
  options {
    timeout(time: 1, unit: 'HOURS')
    parallelsAlwaysFailFast()
  }
  environment {
    CCACHE_NOHASHDIR = 1
    CCACHE_BASEDIR = "/mnt/data/jenkins/workspace"
  }
  stages {
    stage('Pre build ccache stats') {
      steps {
        sh '''ccache -s'''
      }
    }
    stage('Build') {
      parallel {
        stage('Build kotekan without hardware specific options') {
          steps {
            sh '''cd build/
                  cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive \
                  -DOPENBLAS_PATH=/opt/OpenBLAS/build -DUSE_LAPACK=ON -DBLAZE_PATH=/opt/blaze \
                  -DUSE_OMP=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                  -DCMAKE_C_COMPILER_LAUNCHER=ccache ..
                  make -j 4'''
          }
        }
        stage('Build CHIME kotekan') {
          steps {
            sh '''mkdir -p chime-build
                  cd chime-build
                  cmake -DRTE_SDK=/opt/dpdk \
                  -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON \
                  -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive \
                  -DOPENBLAS_PATH=/opt/OpenBLAS/build -DUSE_LAPACK=ON -DBLAZE_PATH=/opt/blaze \
                  -DUSE_OMP=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DBOOST_TESTS=ON ..
                  make -j 4'''
          }
        }
        stage('Build CHIME kotekan with Clang') {
          steps {
            sh '''mkdir -p chime-build-clang
                  cd chime-build-clang
                  export CC=clang
                  export CXX=clang++
                  cmake -DRTE_SDK=/opt/dpdk \
                  -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON \
                  -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive \
                  -DOPENBLAS_PATH=/opt/OpenBLAS/build -DUSE_LAPACK=ON -DBLAZE_PATH=/opt/blaze \
                  -DUSE_OMP=ON -DBOOST_TESTS=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                  -DCMAKE_C_COMPILER_LAUNCHER=ccache ..
                  make -j 4'''
          }
        }
        stage('Build base kotekan') {
          steps {
            sh '''mkdir -p build_base
                  cd build_base
                  cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache ..
                  make -j 4'''
          }
        }
        stage('Build MacOS kotekan') {
          agent {label 'macos'}
          steps {
            sh '''export PATH=${PATH}:/usr/local/bin/
                  source ~/.bash_profile
                  mkdir -p build_base
                  cd build_base/
                  cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
                  make -j
                  cd ..
                  mkdir build_full
                  cd build_full/
                  cmake -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DUSE_FFTW=ON -DUSE_AIRSPY=ON \
                        -DUSE_LAPACK=ON -DBLAZE_PATH=/usr/local/opt/blaze \
                        -DOPENBLAS_PATH=/usr/local/opt/OpenBLAS \
                        -DUSE_HDF5=ON -DHIGHFIVE_PATH=/usr/local/opt/HighFive \
                        -DCOMPILE_DOCS=ON -DUSE_OPENCL=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                        -DCMAKE_C_COMPILER_LAUNCHER=ccache ..
                  make -j'''
          }
        }
        stage('Build docs') {
          steps {
            sh '''export PATH=${PATH}:/var/lib/jenkins/.local/bin/
                  mkdir -p build-docs
                  cd build-docs/
                  cmake -DCOMPILE_DOCS=ON -DPLANTUML_PATH=/opt/plantuml ..
                  make doc
                  make sphinx'''
          }
        }
        stage('Check code formatting') {
          steps {
            sh '''mkdir -p build-check-format
                  cd build-check-format/
                  cmake ..
                  make clang-format
                  git diff --exit-code
                  black --check --exclude docs ..'''
          }
        }
        stage('Install comet') {
          steps {
            catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
              sh '''python3.7 -m pip install --user git+https://github.com/chime-experiment/comet.git@master'''
            }
          }
        }
      }
    }
    stage('Post build ccache stats') {
      steps {
        sh '''ccache -s'''
      }
    }
    stage('Unit Tests') {
        parallel {
            stage('Python Unit Tests') {
              steps {
                sh '''cd tests/
                      PATH=~/.local/bin:$PATH PYTHONPATH=../python/ python3 -m pytest -n auto -x -vvv -m "not serial"
                      PATH=~/.local/bin:$PATH PYTHONPATH=../python/ python3 -m pytest -x -vvv -m serial'''
              }
            }
            stage('Boost Unit Tests') {
              steps {
                sh '''cd chime-build/tests/
                      python3 -m pytest -x -vvv'''
              }
            }
         }
     }
  }
}
