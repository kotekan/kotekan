pipeline {
  agent any
  options {
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('Build') {
      parallel {
        stage('Build kotekan without hardware specific options') {
          steps {
            sh '''cd build/
                  cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive -DOPENBLAS_PATH=/opt/OpenBLAS/build/ -DUSE_LAPACK=ON -DUSE_OMP=ON -DBOOST_TESTS=ON -DUSE_EVENT=ON ..
                  make'''
          }
        }
        stage('Build CHIME kotekan') {
          steps {
            sh '''mkdir build_chime
                  cd build_chime/
                  cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive -DOPENBLAS_PATH=/opt/OpenBLAS/build/ -DUSE_LAPACK=ON -DUSE_OMP=ON -DBOOST_TESTS=ON -DUSE_EVENT=ON ..
                  make'''
          }
        }
        stage('Build base kotekan') {
          steps {
            sh '''mkdir build_base
                  cd build_base
                  cmake ..
                  make'''
          }
        }
        stage('Build docs') {
          steps {
            sh '''export PATH=${PATH}:/var/lib/jenkins/.local/bin/
                  mkdir build-docs
                  cd build-docs/
                  cmake -DCOMPILE_DOCS=ON ..
                  cd docs/
                  make'''
          }
        }
      }
    }
    stage('Unit Tests') {
      steps {
        sh '''cd tests/
pytest -s -vvv'''
      }
    }
  }
}