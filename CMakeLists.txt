cmake_minimum_required(VERSION 3.10)
project(dynamicMapping)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Open3D REQUIRED)
find_library(CPUID_LIBRARY NAMES cpuid libcpuid)
if(CPUID_LIBRARY)
    message(STATUS "Found cpuid library: ${CPUID_LIBRARY}")
else()
    message(WARNING "cpuid library not found; runtime CPU detection disabled")
    add_compile_definitions(NO_CPUID)
endif()

# Check for AVX2 and FMA support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    message(STATUS "AVX2 and FMA are supported by the compiler")
    add_compile_options(-mavx2 -mfma)
else()
    message(WARNING "AVX2 or FMA not supported; using scalar implementation")
endif()

# Enable optimizations for Release builds
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-O3 -ffast-math)
endif()

# Define include directories for reuse
set(INCLUDE_DIRECTORIES 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/LidarDecode/include
    ${CMAKE_SOURCE_DIR}/extern/GNSSKompass/include
    ${CMAKE_SOURCE_DIR}/3rdparty/robin-map/include/tsl
)

# Core library
add_library(core STATIC
    extern/GNSSKompass/src/callback_navMsg.cpp
    extern/GNSSKompass/src/navMath.cpp
    extern/LidarDecode/src/UdpSocket.cpp
    src/OusterLidarCallback_c.cpp
    src/pipeline.cpp
)

# Include directories for the core library
target_include_directories(core 
    PUBLIC ${INCLUDE_DIRECTORIES}
)

# Link libraries
target_link_libraries(core
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        Boost::system
        Boost::thread
        Open3D::Open3D
        ${CPUID_LIBRARY}
        pthread
)

#  add header
add_executable(dynamicmap src/main.cpp)
target_include_directories(dynamicmap PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dynamicmap PRIVATE core)