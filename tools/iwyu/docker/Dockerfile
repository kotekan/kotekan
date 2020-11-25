FROM docker.pkg.github.com/kotekan/kotekan/kotekan-core:develop

## The maintainer name and email
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install iwyu for clang 9.0
RUN wget http://launchpadlibrarian.net/461878080/libstdc++6_9.2.1-25ubuntu1_amd64.deb && \
    wget http://launchpadlibrarian.net/458626653/iwyu_8.0-3_amd64.deb && \
    wget http://launchpadlibrarian.net/461877892/gcc-9-base_9.2.1-25ubuntu1_amd64.deb && \
    wget http://launchpadlibrarian.net/462079080/libllvm9_9.0.1-8build1_amd64.deb && \
    wget http://launchpadlibrarian.net/453587373/libc6_2.30-0ubuntu3_amd64.deb && \
    wget http://launchpadlibrarian.net/461284144/libffi7_3.3-3_amd64.deb && \
    wget http://launchpadlibrarian.net/448587516/libtinfo6_6.1+20191019-1ubuntu1_amd64.deb && \
    dpkg -i libtinfo6_6.1+20191019-1ubuntu1_amd64.deb && \
    dpkg -i libffi7_3.3-3_amd64.deb && \
    dpkg -i libc6_2.30-0ubuntu3_amd64.deb && \
    dpkg -i gcc-9-base_9.2.1-25ubuntu1_amd64.deb && \
    dpkg -i libstdc++6_9.2.1-25ubuntu1_amd64.deb && \
    dpkg -i libllvm9_9.0.1-8build1_amd64.deb && \
    dpkg -i iwyu_8.0-3_amd64.deb

# Set compiler to clang (to make cmake choose the right flags for iwyu)
ENV CC clang-9
ENV CXX clang++-9

# Help iwyu to find builtin headers
ENV CFLAGS "-isystem /usr/lib/llvm-9/lib/clang/9.0.0/include/"
ENV CXXFLAGS "-isystem /usr/lib/llvm-9/lib/clang/9.0.0/include/"
