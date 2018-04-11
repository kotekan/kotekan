pipeline {
  agent any
  stages {
    stage('Build') {
      parallel {
        stage('Build CHIME kotekan') {
          steps {
            sh '''mkdir build-chime
cd build-chime/
cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..
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
      }
    }
  }
}