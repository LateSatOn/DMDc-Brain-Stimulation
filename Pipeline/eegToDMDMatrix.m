function X = eegToDMDMatrix(EEG)
%EEGTODMDMATRIX Convert an EEGLAB EEG struct into a DMD-ready matrix.
%
%   X = EEGTODMDMATRIX(EEG) converts an EEGLAB EEG struct (e.g. from
%   loadEEGSet) into a time x channels matrix X suitable for direct use
%   as Big_X in dmd_rom:
%
%       [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, ...
%           GrowthRates, POD_Mode_Energies] = dmd_rom(X, r, dt);
%
%   where r <= size(X,2) (number of channels kept) and dt = 1/EEG.srate.

if ~isfield(EEG, 'data') || ~isfield(EEG, 'srate')
    error('eegToDMDMatrix:badInput', ...
        'EEG must be a valid EEGLAB EEG struct (missing .data or .srate).');
end

data = double(EEG.data);   % channels x time
nChannelsAll = size(data, 1);

% ---- Select EEG-type channels only, if type info is available ----
keepIdx = 1:nChannelsAll;

if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
    if isfield(EEG.chanlocs, 'type') && any(~cellfun(@isempty, {EEG.chanlocs.type}))
        types = {EEG.chanlocs.type};
        isEEGType = cellfun(@(t) ischar(t) && strcmpi(t, 'EEG'), types);
        if any(isEEGType)
            keepIdx = find(isEEGType);
        end
    end
end

data = data(keepIdx, :);

% ---- Orient as time x channels for dmd_rom ----
X = data.';   % nTime x nChannels

% ---- Warn (not error) if boundary events are present ----
if isfield(EEG, 'event') && ~isempty(EEG.event)
    rawType = {EEG.event.type};
    isBoundary = cellfun(@(t) ischar(t) && strcmpi(t, 'boundary'), rawType);
    if any(isBoundary)
        warning('eegToDMDMatrix:boundaryEvents', ...
            ['%d ''boundary'' event(s) found in EEG.event. DMD assumes one ' ...
             'continuous, uniformly-sampled time series - consider splitting ' ...
             'X at these latencies before running dmd_rom.'], sum(isBoundary));
    end
end

end
