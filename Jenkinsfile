pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh '''cd build/
cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..
make'''
      }
    }
  }
}