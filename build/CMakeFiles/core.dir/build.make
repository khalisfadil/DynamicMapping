# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build

# Include any dependencies generated for this target.
include CMakeFiles/core.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/core.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/core.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/core.dir/flags.make

CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o: /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/callback_navMsg.cpp
CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o -MF CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o.d -o CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o -c /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/callback_navMsg.cpp

CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/callback_navMsg.cpp > CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.i

CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/callback_navMsg.cpp -o CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.s

CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o: /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/navMath.cpp
CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o -MF CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o.d -o CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o -c /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/navMath.cpp

CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/navMath.cpp > CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.i

CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/GNSSKompass/src/navMath.cpp -o CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.s

CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o: /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/LidarDecode/src/UdpSocket.cpp
CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o -MF CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o.d -o CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o -c /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/LidarDecode/src/UdpSocket.cpp

CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/LidarDecode/src/UdpSocket.cpp > CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.i

CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/extern/LidarDecode/src/UdpSocket.cpp -o CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.s

CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o: /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/OusterLidarCallback_c.cpp
CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o -MF CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o.d -o CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o -c /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/OusterLidarCallback_c.cpp

CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/OusterLidarCallback_c.cpp > CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.i

CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/OusterLidarCallback_c.cpp -o CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.s

CMakeFiles/core.dir/src/pipeline.cpp.o: CMakeFiles/core.dir/flags.make
CMakeFiles/core.dir/src/pipeline.cpp.o: /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/pipeline.cpp
CMakeFiles/core.dir/src/pipeline.cpp.o: CMakeFiles/core.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/core.dir/src/pipeline.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/core.dir/src/pipeline.cpp.o -MF CMakeFiles/core.dir/src/pipeline.cpp.o.d -o CMakeFiles/core.dir/src/pipeline.cpp.o -c /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/pipeline.cpp

CMakeFiles/core.dir/src/pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core.dir/src/pipeline.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/pipeline.cpp > CMakeFiles/core.dir/src/pipeline.cpp.i

CMakeFiles/core.dir/src/pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core.dir/src/pipeline.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/src/pipeline.cpp -o CMakeFiles/core.dir/src/pipeline.cpp.s

# Object files for target core
core_OBJECTS = \
"CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o" \
"CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o" \
"CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o" \
"CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o" \
"CMakeFiles/core.dir/src/pipeline.cpp.o"

# External object files for target core
core_EXTERNAL_OBJECTS =

libcore.a: CMakeFiles/core.dir/extern/GNSSKompass/src/callback_navMsg.cpp.o
libcore.a: CMakeFiles/core.dir/extern/GNSSKompass/src/navMath.cpp.o
libcore.a: CMakeFiles/core.dir/extern/LidarDecode/src/UdpSocket.cpp.o
libcore.a: CMakeFiles/core.dir/src/OusterLidarCallback_c.cpp.o
libcore.a: CMakeFiles/core.dir/src/pipeline.cpp.o
libcore.a: CMakeFiles/core.dir/build.make
libcore.a: CMakeFiles/core.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libcore.a"
	$(CMAKE_COMMAND) -P CMakeFiles/core.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/core.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/core.dir/build: libcore.a
.PHONY : CMakeFiles/core.dir/build

CMakeFiles/core.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/core.dir/cmake_clean.cmake
.PHONY : CMakeFiles/core.dir/clean

CMakeFiles/core.dir/depend:
	cd /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build /home/khalis/Sync/SensorSOW/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/build/CMakeFiles/core.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/core.dir/depend

