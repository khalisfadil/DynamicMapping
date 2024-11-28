username = 'target7';
% address = '192.168.42.32';
% address = '139.30.200.112';
address = '139.30.200.83';
%address = '10.0.10.32';
<<<<<<< HEAD
model = 'SimulinkModel';
=======
model = 'GenericTargetTemplate';
>>>>>>> 4c66c70 (Resolve merge conflicts)

target = GT.GenericTarget(username, address);

target.terminateAtTaskOverload = false;
target.terminateAtCPUOverload = false;
<<<<<<< HEAD
target.targetSoftwareDirectory = '~/home/target7/Khalis_ws/dynamicMapping/'; % bitte neue File
target.targetBitmaskCPUCores = '0x1FFFFF';
=======
target.targetSoftwareDirectory = '~/Khalis_ws/dynamicMapping/'; % bitte neue File
%target.targetBitmaskCPUCores = '0x1FFFFF';
>>>>>>> 4c66c70 (Resolve merge conflicts)
%target.additionalCompilerFlags.DEBUG_MODE = true;
%target.portAppSocket = 65535;
% target.DownloadAllData; % to download all log data
%GT.DecodeDataFiles; % when to read log file
%#########################################################
%last deploy
%target.GenerateCode;
<<<<<<< HEAD
target.Deploy(model);
=======
target.Deploy(model);
>>>>>>> 4c66c70 (Resolve merge conflicts)
