# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/holo/Code/holovideo/holovideo-src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/holo/Code/holovideo/holovideo-src/build

# Utility rule file for copy-remote-scripts.

# Include the progress variables for this target.
include remote/scripts/CMakeFiles/copy-remote-scripts.dir/progress.make

remote/scripts/CMakeFiles/copy-remote-scripts:

copy-remote-scripts: remote/scripts/CMakeFiles/copy-remote-scripts
copy-remote-scripts: remote/scripts/CMakeFiles/copy-remote-scripts.dir/build.make
	cd /home/holo/Code/holovideo/holovideo-src/build/remote/scripts && /usr/bin/cmake -E copy /home/holo/Code/holovideo/holovideo-src/remote/scripts/holovideo-disable.sh /home/holo/Code/holovideo/holovideo-src/bin
	cd /home/holo/Code/holovideo/holovideo-src/build/remote/scripts && /usr/bin/cmake -E copy /home/holo/Code/holovideo/holovideo-src/remote/scripts/nvidia-framelock-enable.sh /home/holo/Code/holovideo/holovideo-src/bin
	cd /home/holo/Code/holovideo/holovideo-src/build/remote/scripts && /usr/bin/cmake -E copy /home/holo/Code/holovideo/holovideo-src/remote/scripts/nvidia-framelock-disable.sh /home/holo/Code/holovideo/holovideo-src/bin
	cd /home/holo/Code/holovideo/holovideo-src/build/remote/scripts && /usr/bin/cmake -E copy /home/holo/Code/holovideo/holovideo-src/remote/scripts/holovideo-enable.sh /home/holo/Code/holovideo/holovideo-src/bin
.PHONY : copy-remote-scripts

# Rule to build all files generated by this target.
remote/scripts/CMakeFiles/copy-remote-scripts.dir/build: copy-remote-scripts
.PHONY : remote/scripts/CMakeFiles/copy-remote-scripts.dir/build

remote/scripts/CMakeFiles/copy-remote-scripts.dir/clean:
	cd /home/holo/Code/holovideo/holovideo-src/build/remote/scripts && $(CMAKE_COMMAND) -P CMakeFiles/copy-remote-scripts.dir/cmake_clean.cmake
.PHONY : remote/scripts/CMakeFiles/copy-remote-scripts.dir/clean

remote/scripts/CMakeFiles/copy-remote-scripts.dir/depend:
	cd /home/holo/Code/holovideo/holovideo-src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/holo/Code/holovideo/holovideo-src /home/holo/Code/holovideo/holovideo-src/remote/scripts /home/holo/Code/holovideo/holovideo-src/build /home/holo/Code/holovideo/holovideo-src/build/remote/scripts /home/holo/Code/holovideo/holovideo-src/build/remote/scripts/CMakeFiles/copy-remote-scripts.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : remote/scripts/CMakeFiles/copy-remote-scripts.dir/depend

