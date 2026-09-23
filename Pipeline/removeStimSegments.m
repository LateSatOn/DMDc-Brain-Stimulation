function [Xclean, keptSampleIdx] = removeStimSegments(EEG, X, bufferSamples)
%REMOVESTIMSEGMENTS Remove stimulation periods from a DMD matrix.
%
%   Xclean = REMOVESTIMSEGMENTS(EEG, X, bufferSamples) removes rows
%   (time samples) from X that fall within a stimulation period, and
%   returns the shortened matrix Xclean.
%
%   STIMULATION MARKER CONVENTION (confirmed against this dataset's
%   events and cross-checked visually with plotChannelWithStim; and
%   cross-referenced against a differently-processed copy of this same
%   task that stores human-readable labels instead of numeric codes):
%       type == 16 or 'Stim Start'  -> stimulation ONSET
%       type == 32 or 'Stim Stop'   -> stimulation OFFSET
%   Every onset is paired with the next offset that follows it, and the
%   ENTIRE onset-to-offset span is removed (not just a window around a
%   single instant). Other event types seen in this task (type == 2 /
%   'Block Start' = block/trial boundary markers, type == 0 /
%   'Impedance' = impedance checks) are left alone - they are not
%   stimulation.
%
%   Matching accepts EITHER form because different exports of the same
%   task can store EEG.event(:).type differently - as a number, a
%   numeric string ('16'), or a text label ('Stim Start') - depending on
%   how the dataset was processed. If your dataset uses different
%   codes/labels entirely, add them to ONSET_MARKERS / OFFSET_MARKERS
%   below (mix numeric codes and text labels freely in those lists).
%
%   INPUTS
%   EEG           : the EEGLAB EEG struct the events come from (e.g.
%                   from loadEEGSet).
%   X             : nTime x nChannels matrix, as produced by
%                   eegToDMDMatrix(EEG). X's row count must match
%                   EEG.pnts, so its time axis lines up sample-for-
%                   sample with EEG.event(:).latency.
%   bufferSamples : nonnegative integer. Extra samples to exclude AFTER
%                   each stimulation OFFSET. 0 = remove exactly the
%                   onset-to-offset span, nothing extra. N = also
%                   removes the N samples immediately following the
%                   offset.
%
%   OUTPUTS
%   Xclean        : X with the stimulation rows deleted. Rows are
%                   removed (not zeroed), so Xclean's time axis is no
%                   longer evenly spaced across the removed gaps.
%   keptSampleIdx : nTimeClean x 1 vector of the ORIGINAL sample indices
%                   (into X/EEG) that survived, in order. Xclean(k,:)
%                   corresponds to original sample keptSampleIdx(k).
%                   Pass this to plotCleanedChannel so it can break the
%                   plotted line at each removed gap instead of drawing
%                   a straight (and misleading-looking) connecting line
%                   across it.

ONSET_MARKERS  = {16, 'Stim Start'};
OFFSET_MARKERS = {32, 'Stim Stop'};

if ~isfield(EEG, 'event')
    error('removeStimSegments:noEvents', 'EEG.event is missing.');
end
if ~isfield(EEG, 'pnts')
    error('removeStimSegments:noPnts', 'EEG.pnts is missing.');
end
if size(X, 1) ~= EEG.pnts
    error('removeStimSegments:sizeMismatch', ...
        ['size(X,1) (%d) does not match EEG.pnts (%d). Pass in X exactly as ' ...
         'produced by eegToDMDMatrix(EEG), before any other trimming, so its ' ...
         'rows still line up with EEG.event(:).latency.'], size(X,1), EEG.pnts);
end
if ~isscalar(bufferSamples) || bufferSamples < 0 || bufferSamples ~= round(bufferSamples)
    error('removeStimSegments:badBuffer', ...
        'bufferSamples must be a nonnegative integer.');
end

nTime = size(X, 1);

if isempty(EEG.event)
    Xclean = X;
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
    warning('removeStimSegments:unbalancedMarkers', ...
        ['%d onset event(s) but %d offset event(s) - some stimulation periods ' ...
         'may not pair up cleanly. Check EEG.event manually if this matters.'], ...
        nOnsets, nOffsets);
end

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
            warning('removeStimSegments:unmatchedOnset', ...
                ['Stimulation onset at sample %d has no matching offset - ' ...
                 'removing from onset to the end of the recording.'], onsetIdx);
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

fprintf(['removeStimSegments: removing %d of %d samples (%.2f%%) across %d ' ...
    'stimulation period(s) (onset->offset pairs, buffer=%d).\n'], ...
    sum(removeMask), nTime, 100 * sum(removeMask) / nTime, nPairs, bufferSamples);

Xclean = X(~removeMask, :);
keptSampleIdx = find(~removeMask);

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