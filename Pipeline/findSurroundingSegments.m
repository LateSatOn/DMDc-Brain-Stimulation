function [beforeIdx, afterIdx] = findSurroundingSegments(segmentInfo, targetSample)
%FINDSURROUNDINGSEGMENTS Find the windowStimSegments segments right before/after a sample.
%
%   [beforeIdx, afterIdx] = FINDSURROUNDINGSEGMENTS(segmentInfo, targetSample)
%
%   windowStimSegments splits a recording into contiguous between-
%   stimulation windows (segments{k}), each described by segmentInfo(k)
%   .startSample/.endSample. Given the sample index of a SPECIFIC
%   stimulation event you care about (e.g. EEG.event(i).latency for the
%   onset you want to look at), this finds which segment ends right
%   before it ("before this stimulation") and which segment starts right
%   after it ("after this stimulation") - without assuming segments{k}
%   and segments{k+1} are always adjacent in index (minSegmentSamples >
%   0 in windowStimSegments can drop short segments, which would break a
%   naive k/k+1 assumption).
%
%   INPUTS
%   segmentInfo   : Kx1 struct array from windowStimSegments (fields
%                   startSample, endSample, nSamples).
%   targetSample  : scalar positive integer, the sample index of the
%                   stimulation event you're centering on (e.g. the
%                   onset latency of the N-th "Stim Start"/16 event from
%                   EEG.event - round it first if it's not already an
%                   integer sample index).
%
%   OUTPUTS
%   beforeIdx     : index into segmentInfo/segments for the segment
%                   immediately preceding targetSample (i.e. the most
%                   recent segment whose endSample < targetSample).
%                   Empty ([]) if targetSample is before every segment.
%   afterIdx      : index into segmentInfo/segments for the segment
%                   immediately following targetSample (the earliest
%                   segment whose startSample > targetSample). Empty
%                   ([]) if targetSample is after every segment.
%
%   USAGE
%       [segments, segIdx, segmentInfo] = windowStimSegments(EEG, X, 0, 200);
%       stimOnsetSample = round(EEG.event(7).latency);  % pick the stim you care about
%       [beforeIdx, afterIdx] = findSurroundingSegments(segmentInfo, stimOnsetSample);
%
%       Xbefore = segments{beforeIdx};
%       Xafter  = segments{afterIdx};
%
%       rBefore = min(size(Xbefore,2), size(Xbefore,1)-1);
%       rAfter  = min(size(Xafter,2),  size(Xafter,1)-1);
%       [EigBefore, VecBefore, ~, ~, ~, ~] = runDMDOnMatrix(Xbefore, EEG.srate, rBefore);
%       [EigAfter,  VecAfter,  ~, ~, ~, ~] = runDMDOnMatrix(Xafter,  EEG.srate, rAfter);

if ~isstruct(segmentInfo) || isempty(segmentInfo)
    error('findSurroundingSegments:badSegmentInfo', 'segmentInfo must be a nonempty struct array from windowStimSegments.');
end
if ~all(isfield(segmentInfo, {'startSample', 'endSample'}))
    error('findSurroundingSegments:missingFields', 'segmentInfo must have startSample and endSample fields.');
end
if ~isscalar(targetSample) || targetSample <= 0 || targetSample ~= round(targetSample)
    error('findSurroundingSegments:badTarget', 'targetSample must be a positive integer sample index.');
end

startSamples = [segmentInfo.startSample];
endSamples   = [segmentInfo.endSample];

beforeCandidates = find(endSamples < targetSample);
if isempty(beforeCandidates)
    beforeIdx = [];
else
    [~, relIdx] = max(endSamples(beforeCandidates));
    beforeIdx = beforeCandidates(relIdx);
end

afterCandidates = find(startSamples > targetSample);
if isempty(afterCandidates)
    afterIdx = [];
else
    [~, relIdx] = min(startSamples(afterCandidates));
    afterIdx = afterCandidates(relIdx);
end

if isempty(beforeIdx)
    warning('findSurroundingSegments:noBefore', 'No segment ends before sample %d.', targetSample);
end
if isempty(afterIdx)
    warning('findSurroundingSegments:noAfter', 'No segment starts after sample %d.', targetSample);
end

end
