function EEG = loadEEGSet(folderPath)
%LOADEEGSET Load a BIDS-style EEGLAB .set/.fdt pair into an EEG struct.
%
%   EEG = LOADEEGSET(folderPath) locates the EEGLAB .set file inside
%   folderPath (or its 'eeg' subfolder, if folderPath is a ses-xx level
%   directory) and loads it using EEGLAB's pop_loadset, returning the
%   populated EEG struct. Useful for pulling a subject/session's EEG
%   data into the base workspace ahead of a Simulink model run (e.g.
%   from a model InitFcn or a standalone setup script).
%
%   Expected folder layout:
%       ses-xx/
%           eeg/
%               sub-xxx_ses-xx_task-GXtESCTT_eeg.set
%               sub-xxx_ses-xx_task-GXtESCTT_eeg.fdt
%
%   Example:
%       EEG = loadEEGSet('/data/sub-001');            % finds ses-xx/eeg
%       EEG = loadEEGSet('/data/sub-001/ses-01');
%       EEG = loadEEGSet('/data/sub-001/ses-01/eeg');
%
%   Requires EEGLAB to be on the MATLAB path (for pop_loadset).

taskLabel = 'GXtESCTT';

% Force to a plain char array. If this is called with double-quoted
% input (e.g. loadEEGSet("C:\...")), folderPath comes in as a MATLAB
% "string" object rather than a char array. EEGLAB's older functions
% (pop_loadset, vararg2str) aren't string-aware and can misbehave badly
% on that type - vararg2str throws "Unrecognized input", and
% pop_loadset's own filename/extension parsing can silently mishandle
% it. Converting here makes the function work the same regardless of
% which quote style is used to call it.
folderPath = char(folderPath);

if ~isfolder(folderPath)
    error('loadEEGSet:badFolder', 'Folder does not exist: %s', folderPath);
end

% Accept the eeg folder itself, its parent (a ses-xx folder), or a
% sub-xxx folder that contains one or more ses-xx folders.
[~, leafName] = fileparts(folderPath);
if strcmpi(leafName, 'eeg')
    eegFolder = folderPath;
elseif isfolder(fullfile(folderPath, 'eeg'))
    eegFolder = fullfile(folderPath, 'eeg');
else
    sesListing = dir(fullfile(folderPath, 'ses-*'));
    sesListing = sesListing([sesListing.isdir]);

    candidateEegFolders = {};
    for k = 1:numel(sesListing)
        candidate = fullfile(folderPath, sesListing(k).name, 'eeg');
        if isfolder(candidate)
            candidateEegFolders{end+1} = candidate; %#ok<AGROW>
        end
    end

    if isempty(candidateEegFolders)
        error('loadEEGSet:noEegFolder', ...
            'No ''eeg'' folder found under: %s (looked directly, and under any ses-xx subfolders)', ...
            folderPath);
    elseif numel(candidateEegFolders) > 1
        error('loadEEGSet:multipleSessions', ...
            ['Multiple ses-xx/eeg folders found under: %s\n' ...
             'Point loadEEGSet at a specific session folder instead.'], folderPath);
    end

    eegFolder = candidateEegFolders{1};
end

% Locate the .set file matching the expected naming pattern.
pattern = sprintf('*task-%s_eeg.set', taskLabel);
setListing = dir(fullfile(eegFolder, pattern));

if isempty(setListing)
    error('loadEEGSet:noSetFile', ...
        'No .set file matching "%s" found in: %s', pattern, eegFolder);
elseif numel(setListing) > 1
    error('loadEEGSet:multipleSetFiles', ...
        'Multiple .set files matching "%s" found in: %s', pattern, eegFolder);
end

setFileName = setListing(1).name;

% pop_loadset locates the companion .fdt automatically via EEG.data
% pointing at the same base filename, as long as it sits alongside
% the .set file in eegFolder.
if exist('pop_loadset', 'file') ~= 2
    error('loadEEGSet:noEeglab', ...
        'pop_loadset not found. Add EEGLAB to the MATLAB path first.');
end

% Called with a single full file path rather than the 'filename'/
% 'filepath' name-value pair: some EEGLAB installs throw an error inside
% pop_loadset's internal vararg2str (history-string building) when using
% the name-value form, often due to a shadowed vararg2str from another
% toolbox on the path. This calling form avoids that code path.
fullSetPath = char(fullfile(eegFolder, setFileName));
EEG = pop_loadset(fullSetPath);

end