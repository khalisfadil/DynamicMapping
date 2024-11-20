function makeInfo = rtwmakecfg()
%RTWMAKECFG adds include and source directories to the make files.
%    makeInfo=
%    Simulink version    : 23.2 (R2023b) 01-Aug-2023
%    MATLAB file generated on : 20-Nov-2024 09:50:55

% Verify the Simulink version
verify_simulink_version();

% Get the current directory
currDir = pwd;

% Get the MATLAB search path, remove the toolbox sub-directories except LCT demos
matlabPath = legacycode.lct.util.getSearchPath();

% Declare cell arrays for storing the paths
allIncPaths = {};
allSrcPaths = {};

% Get the build type
isSimTarget = is_simulation_target();
% Get the serialized paths information
info = get_serialized_info();

% Get all S-Function's name in the current model
sfunNames = {};
if ~isempty(bdroot)
    sfunBlks = find_system(bdroot,...
        'MatchFilter', @Simulink.match.allVariants, ...
        'LookUnderMasks', 'all',...
        'FollowLinks', 'on',...
        'BlockType', 'S-Function'...
    );
    sfunNames = get_param(sfunBlks, 'FunctionName');
end

for ii = 1:numel(info)
    sfunInfo = info(ii);
    % If the S-Function isn't part of the current build then skip its path info
    if ~any(strcmp(sfunInfo.SFunctionName, sfunNames))
        continue
    end

    % Call the helper
    pathInfo = legacycode.lct.util.resolvePaths(sfunInfo, isSimTarget==1, currDir, matlabPath);

    % Concatenate known include and source directories
    allIncPaths = RTW.uniquePath([allIncPaths pathInfo.IncPaths(:)']);
    allSrcPaths = RTW.uniquePath([allSrcPaths pathInfo.SrcPaths(:)']);
end

% Additional include directories
makeInfo.includePath = correct_path_name(allIncPaths);

% Additional source directories
makeInfo.sourcePath = correct_path_name(allSrcPaths);

%% ------------------------------------------------------------------------
function info = get_serialized_info()

% Allocate the output structure array
info(1:1) = struct(...
    'SFunctionName', '',...
    'IncPaths', {{}},...
    'SrcPaths', {{}},...
    'LibPaths', {{}},...
    'SourceFiles', {{}},...
    'HostLibFiles', {{}},...
    'TargetLibFiles', {{}},...
    'singleCPPMexFile', false,...
    'Language', ''...
    );

info(1).SFunctionName = 'SFunctionDynamicMapping';
info(1).IncPaths = {'/home/khalis/Workspace/SensorSOW/SynologyDrive/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/matlab/library/source/code', '/home/khalis/Workspace/SensorSOW/SynologyDrive/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/matlab/library/source/code/tsl'};
info(1).SrcPaths = {'/home/khalis/Workspace/SensorSOW/SynologyDrive/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/matlab/library/source/code', '/home/khalis/Workspace/SensorSOW/SynologyDrive/Arbeitspakete/MATLAB/Developement/020_DynamicMapping/matlab/library/source/code/tsl'};
info(1).SourceFiles = {'dynamicMapping.cpp', 'M_occupancyMap.cpp', 'M_clusterExtractor.cpp', 'M_EKFVelocity2D.cpp'};
info(1).Language = 'C++';

%% ------------------------------------------------------------------------
function verify_simulink_version()

% Retrieve Simulink version
slVerStruct = ver('simulink');
slVer = str2double(strsplit(slVerStruct.Version,'.'));
% Verify that the actual platform supports the function used
if slVer(1)<8 || (slVer(1)==8 && slVer(2)<7)
    DAStudio.error('Simulink:tools:LCTErrorBadSimulinkVersion', slVerStruct.Version)
end

%% ------------------------------------------------------------------------
function pathSet = correct_path_name(pathSet)

for ii = 1:numel(pathSet)
    pathSet{ii} = coder.make.internal.transformPaths(pathSet{ii});
end
pathSet = RTW.uniquePath(pathSet);

%% ------------------------------------------------------------------------
function isSimTarget = is_simulation_target()

% Default output value
isSimTarget = 0;

% Get the current model and the code generation type to decide
% if we must link with the host libraries or with the target libraries
try
    modelName = get_param(0, 'CurrentSystem');
    if ~isempty(modelName)
        modelName = bdroot(modelName);
        sysTarget = get_param(modelName, 'RTWSystemTargetFile');
        isSimTarget = ~isempty([strfind(sysTarget, 'rtwsfcn') strfind(sysTarget, 'accel')]);

        mdlRefSimTarget = get_param(modelName,'ModelReferenceTargetType');
        isSimTarget = strcmpi(mdlRefSimTarget, 'SIM') || isSimTarget;

        % Verify again it's not Accelerator
        if ~isSimTarget
            simMode = get_param(modelName, 'SimulationMode');
            simStat = get_param(modelName, 'SimulationStatus');
            isSimTarget = strcmp(simStat, 'initializing') & strcmp(simMode, 'accelerator');
        end
    end
catch
end
