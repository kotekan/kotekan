pipeline {
  agent any
  stages {
    stage('Build') {
      parallel {
        stage('Build CHIME kotekan') {
          steps {
            sh '''mkdir build-chime
                  cd build-chime/
                  cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/opt/HighFive -DOPENBLAS_PATH=/opt/OpenBLAS/build/ -DUSE_LAPACK=ON ..
                  make'''
          }
        }
        stage('Build base kotekan') {
          steps {
            sh '''cd build
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
        stage('Build MacOS kotekan') {
          agent {label 'macos silver'}
          steps {
            sh '''export PATH=${PATH}:/usr/local/bin/
                  mkdir build-macos
                  cd build-macos/
                  cmake ..
                  make'''
          }
        }

      }
    }
  }
}