function [segments, segmentSampleIdx, segmentInfo] = windowStimSegments(EEG, X, bufferSamples, minSegmentSamples)
%WINDOWSTIMSEGMENTS Split a DMD matrix into one matrix per between-stim window.
%
%   [segments, segmentSampleIdx] = WINDOWSTIMSEGMENTS(EEG, X, bufferSamples)
%   finds every stimulation onset->offset span (same marker convention as
%   removeStimSegments) and, instead of deleting those rows and
%   concatenating what's left into one matrix, returns each contiguous
%   NON-stimulation run of X as its own separate matrix. This avoids the
%   discontinuity problem you get from removeStimSegments: since each
%   segment is internally contiguous in time, running DMD independently
%   on each segment is well-posed (fixed, real sample-to-sample dt),
%   whereas running DMD on one concatenated matrix silently invents a
%   fake one-step transition across every removed gap.
%
%   STIMULATION MARKER CONVENTION (same as removeStimSegments - see that
%   file for the full rationale):
%       type == 16 or 'Stim Start'  -> stimulation ONSET
%       type == 32 or 'Stim Stop'   -> stimulation OFFSET
%   Every onset is paired with the next offset that follows it, and the
%   entire onset-to-offset span (plus bufferSamples after the offset) is
%   treated as stimulation and excluded from the windows below.
%
%   INPUTS
%   EEG               : the EEGLAB EEG struct the events come from.
%   X                 : nTime x nChannels matrix, as produced by
%                       eegToDMDMatrix(EEG). X's row count must match
%                       EEG.pnts, so its time axis lines up sample-for-
%                       sample with EEG.event(:).latency.
%   bufferSamples     : nonnegative integer. Extra samples to exclude
%                       AFTER each stimulation OFFSET before a new
%                       window is allowed to start. 0 = window starts
%                       right at the offset sample.
%   minSegmentSamples : (optional, default 0) nonnegative integer.
%                       Segments shorter than this are dropped (with a
%                       warning identifying which one and its length)
%                       instead of returned. Use this to filter out
%                       windows too short to reliably resolve the modes
%                       you care about.
%
%   OUTPUTS
%   segments          : 1xK cell array. segments{k} is an
%                       nSamplesK x nChannels matrix - the k-th
%                       contiguous between-stimulation window of X, in
%                       time order. Each one is internally contiguous
%                       (uniformly sampled), so it's safe to hand
%                       directly to a DMD routine.
%   segmentSampleIdx  : 1xK cell array. segmentSampleIdx{k} is an
%                       nSamplesK x 1 vector of the ORIGINAL sample
%                       indices (into X/EEG) that make up segments{k}.
%                       segments{k}(r,:) corresponds to original sample
%                       segmentSampleIdx{k}(r). Useful for lining up
%                       per-window DMD results (e.g. mode timestamps for
%                       your moving-average-across-windows step) back to
%                       real recording time.
%   segmentInfo       : Kx1 struct array with fields startSample,
%                       endSample, nSamples - a quick-reference summary
%                       of each returned segment, in the same order as
%                       segments/segmentSampleIdx. Segments dropped for
%                       being too short are NOT included here.
%
%   Typical next step:
%       [segments, idx] = windowStimSegments(EEG, X, 0, 200);
%       for k = 1:numel(segments)
%           [Phi{k}, lambda{k}, ...] = runYourDMD(segments{k});
%       end
%       % then match modes across k by eigenvalue proximity / spatial
%       % correlation, phase-align, and take your moving average.

ONSET_MARKERS  = {16, 'Stim Start'};
OFFSET_MARKERS = {32, 'Stim Stop'};

if nargin < 4 || isempty(minSegmentSamples)
    minSegmentSamples = 0;
end

if ~isfield(EEG, 'event')
    error('windowStimSegments:noEvents', 'EEG.event is missing.');
end
if ~isfield(EEG, 'pnts')
    error('windowStimSegments:noPnts', 'EEG.pnts is missing.');
end
if size(X, 1) ~= EEG.pnts
    error('windowStimSegments:sizeMismatch', ...
        ['size(X,1) (%d) does not match EEG.pnts (%d). Pass in X exactly as ' ...
         'produced by eegToDMDMatrix(EEG), before any other trimming, so its ' ...
         'rows still line up with EEG.event(:).latency.'], size(X,1), EEG.pnts);
end
if ~isscalar(bufferSamples) || bufferSamples < 0 || bufferSamples ~= round(bufferSamples)
    error('windowStimSegments:badBuffer', ...
        'bufferSamples must be a nonnegative integer.');
end
if ~isscalar(minSegmentSamples) || minSegmentSamples < 0 || minSegmentSamples ~= round(minSegmentSamples)
    error('windowStimSegments:badMinSegment', ...
        'minSegmentSamples must be a nonnegative integer.');
end

nTime = size(X, 1);

if isempty(EEG.event)
    % No events at all -> the whole recording is one segment.
    segments = {X};
    segmentSampleIdx = {(1:nTime)'};
    segmentInfo = struct('startSample', 1, 'endSample', nTime, 'nSamples', nTime);
    return;
end

latencies = round([EEG.event.latency]);
rawType   = {EEG.event.type};

% Sort by latency in case events aren't already in time order.
[latencies, sortOrder] = sort(latencies);
rawType = rawType(sortOrder);

nOnsets  = sum(cellfun(@(t) isAnyMarker(t, ONSET_MARKERS),  rawType));
nOffsets = sum(cellfun(@(t) isAnyMarker(t, OFFSET_MARKERS), rawType));
if nOnsets ~= nOffsets
    warning('windowStimSegments:unbalancedMarkers', ...
        ['%d onset event(s) but %d offset event(s) - some stimulation periods ' ...
         'may not pair up cleanly. Check EEG.event manually if this matters.'], ...
        nOnsets, nOffsets);
end

% Same pairing walk as removeStimSegments, but here we build up a mask
% instead of deleting rows, so the two functions stay consistent with
% each other about what counts as "stimulation".
removeMask = false(nTime, 1);
nPairs = 0;
n = numel(latencies);
i = 1;
while i <= n
    if isAnyMarker(rawType{i}, ONSET_MARKERS)
        onsetIdx = latencies(i);
        offsetIdx = [];
        j = i + 1;
        while j <= n
            if isAnyMarker(rawType{j}, OFFSET_MARKERS)
                offsetIdx = latencies(j);
                break;
            elseif isAnyMarker(rawType{j}, ONSET_MARKERS)
                break; % another onset before any offset - malformed pairing, stop looking
            end
            j = j + 1;
        end

        if isempty(offsetIdx)
            warning('windowStimSegments:unmatchedOnset', ...
                ['Stimulation onset at sample %d has no matching offset - ' ...
                 'treating everything from onset to the end of the recording ' ...
                 'as stimulation.'], onsetIdx);
            offsetIdx = nTime;
        end

        startIdx = max(onsetIdx, 1);
        endIdx   = min(offsetIdx + bufferSamples, nTime);
        removeMask(startIdx:endIdx) = true;
        nPairs = nPairs + 1;
        i = j + 1;
    else
        i = i + 1;
    end
end

% Find contiguous runs of KEPT (non-stimulation) samples. Pad with a
% false sentinel on each side so runs touching the very start/end of the
% recording are found the same way as interior runs.
kept = ~removeMask;
padded = [false; kept; false];
d = diff(padded);
runStarts = find(d == 1);       % index into padded -> corresponds to original index runStarts
runEnds   = find(d == -1) - 1;  % inclusive end, still in padded's index space

segments = {};
segmentSampleIdx = {};
segmentInfo = struct('startSample', {}, 'endSample', {}, 'nSamples', {});
nDropped = 0;

for r = 1:numel(runStarts)
    s = runStarts(r); % since padded is offset by 1 relative to original, this is the correct original index
    e = runEnds(r);
    nSamplesRun = e - s + 1;

    if nSamplesRun < minSegmentSamples
        nDropped = nDropped + 1;
        warning('windowStimSegments:segmentTooShort', ...
            'Dropping segment [%d:%d] (%d samples) - below minSegmentSamples (%d).', ...
            s, e, nSamplesRun, minSegmentSamples);
        continue;
    end

    segments{end+1}         = X(s:e, :); %#ok<AGROW>
    segmentSampleIdx{end+1} = (s:e)';    %#ok<AGROW>
    segmentInfo(end+1) = struct( ...     %#ok<AGROW>
        'startSample', s, 'endSample', e, 'nSamples', nSamplesRun);
end

fprintf(['windowStimSegments: found %d stimulation period(s) (buffer=%d), ' ...
    'producing %d window(s) (%d dropped for being shorter than %d samples).\n'], ...
    nPairs, bufferSamples, numel(segments), nDropped, minSegmentSamples);

end

function tf = isAnyMarker(t, markers)
% True if event type t matches ANY entry in markers (a cell array that
% can mix numeric codes, e.g. 16, and text labels, e.g. 'Stim Start').
tf = false;
for k = 1:numel(markers)
    if isMatch(t, markers{k})
        tf = true;
        return;
    end
end
end

function tf = isMatch(t, marker)
% EEG.event(:).type can be stored as a true number, a numeric string
% (e.g. '16'), or a text label (e.g. 'Stim Start') depending on the
% dataset/how it was processed - handle all of these against either a
% numeric or text marker.
if isnumeric(marker)
    if isnumeric(t)
        tf = isscalar(t) && t == marker;
    elseif ischar(t) || isstring(t)
        numVal = str2double(t);
        tf = ~isnan(numVal) && numVal == marker;
    else
        tf = false;
    end
else
    % marker is text: only match against text-type events, case-insensitive.
    if ischar(t) || isstring(t)
        tf = strcmpi(strtrim(char(t)), strtrim(char(marker)));
    else
        tf = false;
    end
end
end

