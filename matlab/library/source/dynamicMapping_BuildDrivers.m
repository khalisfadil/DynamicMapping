clear all;

% Generate specifications for all drivers
defs = [];

% Define the current directory
thisDirectory = pwd;

% Define paths for 'code' and 'code/tsl' within the current directory or relative to current directory structure
directorySourceCode = fullfile(thisDirectory, 'code');
directorySourceTsl = fullfile(thisDirectory, 'code', 'tsl');

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Driver: Dynamic Mapping
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def = legacy_code('initialize');
def.SFunctionName                   = 'SFunctionDynamicMapping';
def.StartFcnSpec                    = 'void CreateDynamicMapping()';
def.OutputFcnSpec                   = ['void OutputDynamicMapping(uint32 u1, single u2[128*1024][3], single u3[128*1024][1],'                   ...
                                                                'single u4[128*1024][1], single u5[128*1024][1],'                               ...
                                                                'double u6[3][1], uint32 u7, double u8, double u9, double u10[3][1],'           ...
                                                                'double u11, int32 u12, int32 u13, double u14, double u15, double u16, double u17,' ...
                                                                'double u18, double u19, double u20,'                                           ...
                                                                'double y1[128*1024*10][3], uint32 y2, double y3[128*1024][3], uint32 y4,'      ...
                                                                'int32 y5[128*1024*10][3], int32 y6[128*1024*10][3], int32 y7[128*1024*10][3],' ...
                                                                'int32 y8[128*1024*10][3], int32 y9[128*1024][3])'];                                                        
def.TerminateFcnSpec                = 'void DeleteDynamicMapping()';
def.HeaderFiles                     = {'dynamicMapping.hpp', 'M_occupancyMap.hpp', 'M_clusterExtractor.hpp', 'M_EKFVelocity2D.hpp', 'robin_growth_policy.h', 'robin_hash.h', 'robin_map.h', 'robin_set.h'};
def.SourceFiles                     = {'dynamicMapping.cpp', 'M_occupancyMap.cpp', 'M_clusterExtractor.cpp', 'M_EKFVelocity2D.cpp'};
def.IncPaths                        = {directorySourceCode, directorySourceTsl};  
def.SrcPaths                        = {directorySourceCode, directorySourceTsl};
def.LibPaths                        = {''};
def.HostLibFiles                    = {};
def.Options.language                = 'C++';
def.Options.useTlcWithAccel         = false;   
def.SampleTime                      = 'parameterized';      
defs = [defs; def];

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Compile and generate all required files
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Generate SFunctions
legacy_code('sfcn_cmex_generate', defs);

% Define the OpenMP flags and other compiler flags inline
mexFlags = {
    'CFLAGS="-Wall -Wextra -Og -mtune=native -fPIC -Winit-self -fdiagnostics-show-option"', ...
    'CXXFLAGS="-Wall -Wextra -Og -mtune=native -std=c++20 -fopenmp -fPIC -Winit-self -fdiagnostics-show-option"', ...
    'LDFLAGS="-Wall -Wextra -Og -mtune=native -std=c++20 -fopenmp -fdiagnostics-show-option"'
};


% Include directories for required libraries
includes = {
    '-I/usr/include',              ...      % General include directory
    '-I/usr/include/tbb',          ...      % Intel TBB headers
    '-I/usr/include/pcl-1.12',     ...      % Point Cloud Library (PCL) headers
    '-I/usr/include/eigen3'              % Eigen library headers (often required by PCL)
};

% Libraries and library paths
libraries = {
    '-L/usr/lib',                                                   ...      % Standard library path
    '-L/usr/local/lib',                                             ...      % Additional library path
    '-lstdc++',                                                     ...      % Standard C++ library
    '-lpthread',                                                    ...      % POSIX thread library
    '-ltbb',                                                        ...      % Intel TBB library
    '-ltbbmalloc',                                                  ...      % TBB scalable memory allocator
    '-lpcl_common', '-lpcl_io', '-lpcl_filters',                    ...      % PCL libraries
    '-lpcl_kdtree', '-lpcl_search', '-lpcl_features',               ...
    '-lpcl_surface', '-lpcl_sample_consensus',                      ...
    '-lpcl_octree', '-lpcl_visualization', '-lpcl_segmentation',    ...
    '-lboost_system', '-lboost_filesystem'                                   % Boost libraries required by PCL
};

% Compile using legacy_code with inline mexFlags, includes, and libraries
legacy_code('compile', defs, {mexFlags{:}, includes{:}, libraries{:}});

% Generate TLC
legacy_code('sfcn_tlc_generate', defs);

% Generate RTWMAKECFG
legacy_code('rtwmakecfg_generate', defs);

% Generate Simulink blocks (not required, all blocks are already in the library)
legacy_code('slblock_generate', defs);
