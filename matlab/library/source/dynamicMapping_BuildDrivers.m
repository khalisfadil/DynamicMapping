defs = [];
directorySource = '';  % Path to the source files

% Driver: Info Request Encode
def = legacy_code('initialize');
def.SFunctionName = 'SFunction_DynamicMapping';
def.StartFcnSpec  = 'void CreateDynamicMapping()';
def.OutputFcnSpec = 'void OutputDynamicMapping()';                                                        % output GICP fittness score
def.TerminateFcnSpec = 'void DeleteDynamicMapping()';
def.HeaderFiles   = {'dynamicMapping.h'};
def.SourceFiles   = {'dynamicMapping.cpp'};
def.IncPaths      = {directorySource};   % Adding source directory to include path
def.SrcPaths      = {directorySource};
def.Options.language = 'C++';
def.Options.useTlcWithAccel = false;   % Change to true if needed
def.SampleTime = 'parameterized';      % Adjust based on your design
defs = [defs; def];

% Compile and generate all required files
legacy_code('sfcn_cmex_generate', defs);

% Define paths and libraries based on the OS
if(ispc())
    includes = {''};
    libraries = {''};
elseif(isunix())
    includes = { ...
        '-I/usr/include/pcl-1.12', ...
        '-I/usr/include/eigen3', ...
        '-I/usr/include/vtk-9.1', ...
        ['-I' directorySource '/library/include']  % Adjust path if necessary
    };
    
    libraries = {
        '-L/usr/lib/x86_64-linux-gnu', ...
        '-lpcl_common', '-lpcl_io', '-lpcl_filters', '-lpcl_kdtree', ...
        '-lpcl_search', '-lpcl_features', '-lpcl_surface', '-lpcl_sample_consensus', ...
        '-lpcl_octree', '-lpcl_visualization', '-lpcl_segmentation', ...
        '-lvtkCommonCore-9.1', '-lboost_system', '-lboost_filesystem'
    };

    mexOpenMPFlags = {
        'CXXFLAGS="\$CXXFLAGS -fopenmp"', ...
        'LDFLAGS="\$LDFLAGS -fopenmp"'
    };
else
    includes = {''};
    libraries = {''};
end

% Compile using legacy_code
legacy_code('compile', defs, {includes{:}, libraries{:}, mexOpenMPFlags{:}});

% Generate TLC and Simulink blocks
legacy_code('sfcn_tlc_generate', defs);
legacy_code('rtwmakecfg_generate', defs);
legacy_code('slblock_generate', defs);

% Clean up variables
clear def defs directorySource includes libraries mexOpenMPFlags;
