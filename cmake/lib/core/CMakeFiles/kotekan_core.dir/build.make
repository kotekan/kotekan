# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/arun/kotekan

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/arun/kotekan/cmake

# Include any dependencies generated for this target.
include lib/core/CMakeFiles/kotekan_core.dir/depend.make

# Include the progress variables for this target.
include lib/core/CMakeFiles/kotekan_core.dir/progress.make

# Include the compile flags for this target's objects.
include lib/core/CMakeFiles/kotekan_core.dir/flags.make

lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o: ../lib/core/buffer.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/kotekan_core.dir/buffer.c.o   -c /root/arun/kotekan/lib/core/buffer.c

lib/core/CMakeFiles/kotekan_core.dir/buffer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/kotekan_core.dir/buffer.c.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /root/arun/kotekan/lib/core/buffer.c > CMakeFiles/kotekan_core.dir/buffer.c.i

lib/core/CMakeFiles/kotekan_core.dir/buffer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/kotekan_core.dir/buffer.c.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /root/arun/kotekan/lib/core/buffer.c -o CMakeFiles/kotekan_core.dir/buffer.c.s

lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.requires

lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.provides: lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.provides

lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o


lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o: ../lib/core/bufferContainer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o -c /root/arun/kotekan/lib/core/bufferContainer.cpp

lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/bufferContainer.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/bufferContainer.cpp > CMakeFiles/kotekan_core.dir/bufferContainer.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/bufferContainer.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/bufferContainer.cpp -o CMakeFiles/kotekan_core.dir/bufferContainer.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o: ../lib/core/bufferFactory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o -c /root/arun/kotekan/lib/core/bufferFactory.cpp

lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/bufferFactory.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/bufferFactory.cpp > CMakeFiles/kotekan_core.dir/bufferFactory.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/bufferFactory.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/bufferFactory.cpp -o CMakeFiles/kotekan_core.dir/bufferFactory.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o: ../lib/core/Config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/Config.cpp.o -c /root/arun/kotekan/lib/core/Config.cpp

lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/Config.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/Config.cpp > CMakeFiles/kotekan_core.dir/Config.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/Config.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/Config.cpp -o CMakeFiles/kotekan_core.dir/Config.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o: ../lib/core/configEval.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/configEval.cpp.o -c /root/arun/kotekan/lib/core/configEval.cpp

lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/configEval.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/configEval.cpp > CMakeFiles/kotekan_core.dir/configEval.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/configEval.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/configEval.cpp -o CMakeFiles/kotekan_core.dir/configEval.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/errors.c.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/errors.c.o: ../lib/core/errors.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object lib/core/CMakeFiles/kotekan_core.dir/errors.c.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/kotekan_core.dir/errors.c.o   -c /root/arun/kotekan/lib/core/errors.c

lib/core/CMakeFiles/kotekan_core.dir/errors.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/kotekan_core.dir/errors.c.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /root/arun/kotekan/lib/core/errors.c > CMakeFiles/kotekan_core.dir/errors.c.i

lib/core/CMakeFiles/kotekan_core.dir/errors.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/kotekan_core.dir/errors.c.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /root/arun/kotekan/lib/core/errors.c -o CMakeFiles/kotekan_core.dir/errors.c.s

lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.requires

lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.provides: lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.provides

lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/errors.c.o


lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o: ../lib/core/kotekanLogging.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o -c /root/arun/kotekan/lib/core/kotekanLogging.cpp

lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/kotekanLogging.cpp > CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/kotekanLogging.cpp -o CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o: ../lib/core/KotekanProcess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o -c /root/arun/kotekan/lib/core/KotekanProcess.cpp

lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/KotekanProcess.cpp > CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/KotekanProcess.cpp -o CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o: ../lib/core/metadata.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/kotekan_core.dir/metadata.c.o   -c /root/arun/kotekan/lib/core/metadata.c

lib/core/CMakeFiles/kotekan_core.dir/metadata.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/kotekan_core.dir/metadata.c.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /root/arun/kotekan/lib/core/metadata.c > CMakeFiles/kotekan_core.dir/metadata.c.i

lib/core/CMakeFiles/kotekan_core.dir/metadata.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/kotekan_core.dir/metadata.c.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /root/arun/kotekan/lib/core/metadata.c -o CMakeFiles/kotekan_core.dir/metadata.c.s

lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.requires

lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.provides: lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.provides

lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o


lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o: ../lib/core/metadataFactory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o -c /root/arun/kotekan/lib/core/metadataFactory.cpp

lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/metadataFactory.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/metadataFactory.cpp > CMakeFiles/kotekan_core.dir/metadataFactory.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/metadataFactory.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/metadataFactory.cpp -o CMakeFiles/kotekan_core.dir/metadataFactory.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o: ../lib/core/processFactory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/processFactory.cpp.o -c /root/arun/kotekan/lib/core/processFactory.cpp

lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/processFactory.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/processFactory.cpp > CMakeFiles/kotekan_core.dir/processFactory.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/processFactory.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/processFactory.cpp -o CMakeFiles/kotekan_core.dir/processFactory.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o: ../lib/core/prometheusMetrics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o -c /root/arun/kotekan/lib/core/prometheusMetrics.cpp

lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/prometheusMetrics.cpp > CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/prometheusMetrics.cpp -o CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o


lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o: lib/core/CMakeFiles/kotekan_core.dir/flags.make
lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o: ../lib/core/restServer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kotekan_core.dir/restServer.cpp.o -c /root/arun/kotekan/lib/core/restServer.cpp

lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kotekan_core.dir/restServer.cpp.i"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/arun/kotekan/lib/core/restServer.cpp > CMakeFiles/kotekan_core.dir/restServer.cpp.i

lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kotekan_core.dir/restServer.cpp.s"
	cd /root/arun/kotekan/cmake/lib/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/arun/kotekan/lib/core/restServer.cpp -o CMakeFiles/kotekan_core.dir/restServer.cpp.s

lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.requires:

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.requires

lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.provides: lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.requires
	$(MAKE) -f lib/core/CMakeFiles/kotekan_core.dir/build.make lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.provides.build
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.provides

lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.provides.build: lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o


# Object files for target kotekan_core
kotekan_core_OBJECTS = \
"CMakeFiles/kotekan_core.dir/buffer.c.o" \
"CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o" \
"CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o" \
"CMakeFiles/kotekan_core.dir/Config.cpp.o" \
"CMakeFiles/kotekan_core.dir/configEval.cpp.o" \
"CMakeFiles/kotekan_core.dir/errors.c.o" \
"CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o" \
"CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o" \
"CMakeFiles/kotekan_core.dir/metadata.c.o" \
"CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o" \
"CMakeFiles/kotekan_core.dir/processFactory.cpp.o" \
"CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o" \
"CMakeFiles/kotekan_core.dir/restServer.cpp.o"

# External object files for target kotekan_core
kotekan_core_EXTERNAL_OBJECTS =

lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/errors.c.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/build.make
lib/core/libkotekan_core.a: lib/core/CMakeFiles/kotekan_core.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/arun/kotekan/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX static library libkotekan_core.a"
	cd /root/arun/kotekan/cmake/lib/core && $(CMAKE_COMMAND) -P CMakeFiles/kotekan_core.dir/cmake_clean_target.cmake
	cd /root/arun/kotekan/cmake/lib/core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kotekan_core.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/core/CMakeFiles/kotekan_core.dir/build: lib/core/libkotekan_core.a

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/build

lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/buffer.c.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/bufferContainer.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/bufferFactory.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/Config.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/configEval.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/errors.c.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/kotekanLogging.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/KotekanProcess.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/metadata.c.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/metadataFactory.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/processFactory.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/prometheusMetrics.cpp.o.requires
lib/core/CMakeFiles/kotekan_core.dir/requires: lib/core/CMakeFiles/kotekan_core.dir/restServer.cpp.o.requires

.PHONY : lib/core/CMakeFiles/kotekan_core.dir/requires

lib/core/CMakeFiles/kotekan_core.dir/clean:
	cd /root/arun/kotekan/cmake/lib/core && $(CMAKE_COMMAND) -P CMakeFiles/kotekan_core.dir/cmake_clean.cmake
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/clean

lib/core/CMakeFiles/kotekan_core.dir/depend:
	cd /root/arun/kotekan/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/arun/kotekan /root/arun/kotekan/lib/core /root/arun/kotekan/cmake /root/arun/kotekan/cmake/lib/core /root/arun/kotekan/cmake/lib/core/CMakeFiles/kotekan_core.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/core/CMakeFiles/kotekan_core.dir/depend

