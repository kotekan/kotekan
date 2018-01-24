# Use an official Python runtime as a base image
FROM ubuntu:trusty

## The maintainer name and email
MAINTAINER Richard Shaw <richard@phas.ubc.ca>

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y cmake
RUN apt-get install -y libhdf5-7 libhdf5-7-dbg libhdf5-dev h5utils
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y python python-setuptools python-pip
RUN pip install pyyaml
#RUN apt-get install -y software-properties-common
#RUN apt-add-repository -y ppa:ubuntu-toolchain-r/test
#RUN apt-get update
#RUN apt-get install -y gcc-5 g++-5

# Run kotekan when the container launches
CMD ["/code/build/kotekan/kotekan", "-c", "$KOTEKAN_CONFIG"]
