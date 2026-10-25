function resultsTable = evaluateSegmentDMDs(segments, segmentInfo, EEG, windowSamples, overlapSamples, r, trainFraction)
%EVALUATESEGMENTDMDS Held-out moving-window DMD fit for every between-stim segment, named by stim.
%
%   resultsTable = EVALUATESEGMENTDMDS(segments, segmentInfo, EEG, ...
%       windowSamples, overlapSamples, r, trainFraction)
%
%   For every segment returned by windowStimSegments, this:
%     1. Labels the segment using the stimulation event(s) immediately
%        before and after it (see NAMING below).
%     2. Splits it into train/test via splitTrainTestDMD(trainFraction).
%     3. Fits a moving-window-averaged DMD (runMovingWindowDMD) on the
%        TRAIN portion.
%     4. Scores one-step-ahead prediction (evaluateDMDOneStepFit) on the
%        held-out TEST portion.
%   and returns everything as one table: RMSE, R^2, and the K-S/A-D
%   residual goodness-of-fit results per segment - no plots, just the
%   numbers.
%
%   This is a genuine held-out generalization check (unlike
%   combineSegmentDMDs, which fits each segment on ALL of its own data
%   to extract the best natural-dynamics model). Use this one to ask
%   "how well does the moving-window DMD actually generalize, segment by
%   segment" and combineSegmentDMDs to ask "what IS the dynamics in each
%   segment, combined."
%
%   NAMING
%   Segment names are built from the ONSET event (type 16 / 'Stim Start',
%   same convention as removeStimSegments/windowStimSegments) immediately
%   before the segment starts and immediately after it ends, e.g.
%   "Seg 5: after M30, before M5" IF your EEG.event(:).type field
%   actually carries distinct condition labels (like 'M30'/'F5'). If your
%   dataset's event types are generic codes (e.g. everything is just 16 /
%   'Stim Start' with no condition info - which is the case for the
%   sub-001 GXtESCTT dataset this pipeline was built around), the names
%   will just show that generic code for every segment; in that case the
%   segment index is still the practical identifier, and this function
%   still tells you which stimulation (by order) each segment sits
%   between.
%
%   INPUTS
%   segments, segmentInfo : outputs of windowStimSegments.
%   EEG             : EEG struct (EEG.event and EEG.srate are used).
%   windowSamples, overlapSamples, r : passed to runMovingWindowDMD.
%   trainFraction   : passed to splitTrainTestDMD (e.g. 0.7).
%
%   OUTPUT
%   resultsTable with columns:
%       segmentIdx, segmentName, nSamples, evaluated,
%       rmseOverall, r2Overall,
%       residualKSTestH, residualKSTestP, residualADTestH, residualADTestP
%   'evaluated' is false (numeric columns NaN) for any segment too short
%   to split/window, or for which fitting failed.
%
%   USAGE
%       [segments, segIdx, segmentInfo] = windowStimSegments(EEG, X, 0, 200);
%       resultsTable = evaluateSegmentDMDs(segments, segmentInfo, EEG, 2500, 1250, 15, 0.7);
%       disp(resultsTable)
%       sortrows(resultsTable(resultsTable.evaluated, :), 'r2Overall', 'descend')

if ~iscell(segments) || isempty(segments)
    error('evaluateSegmentDMDs:badSegments', 'segments must be a nonempty cell array from windowStimSegments.');
end
if ~isstruct(segmentInfo) || numel(segmentInfo) ~= numel(segments)
    error('evaluateSegmentDMDs:badSegmentInfo', 'segmentInfo must be a struct array matching segments in length.');
end
if ~isfield(EEG, 'srate') || ~isfield(EEG, 'event')
    error('evaluateSegmentDMDs:badEEG', 'EEG.srate and EEG.event are required.');
end

nSegments = numel(segments);

% ---- build the sorted list of onset events used for naming ----
ONSET_MARKERS = {16, 'Stim Start'};
if isempty(EEG.event)
    onsetLatencies = [];
    onsetLabels = {};
else
    latencies = round([EEG.event.latency]);
    rawType = {EEG.event.type};
    [latencies, sortOrder] = sort(latencies);
    rawType = rawType(sortOrder);
    onsetMask = cellfun(@(t) isAnyMarker(t, ONSET_MARKERS), rawType);
    onsetLatencies = latencies(onsetMask);
    onsetLabels = cellfun(@labelToStr, rawType(onsetMask), 'UniformOutput', false);
end

% ---- preallocate ----
segmentIdx      = (1:nSegments)';
segmentName     = cell(nSegments, 1);
nSamplesArr     = zeros(nSegments, 1);
evaluated       = false(nSegments, 1);
rmseOverall     = NaN(nSegments, 1);
r2Overall       = NaN(nSegments, 1);
residualKSTestH = NaN(nSegments, 1);
residualKSTestP = NaN(nSegments, 1);
residualADTestH = NaN(nSegments, 1);
residualADTestP = NaN(nSegments, 1);

minSegmentSamplesNeeded = ceil(windowSamples / (1 - trainFraction)) * 3;

for k = 1:nSegments
    Xseg = segments{k};
    nSamplesArr(k) = size(Xseg, 1);
    segmentName{k} = labelOneSegment(k, segmentInfo(k), onsetLatencies, onsetLabels);

    if nSamplesArr(k) < minSegmentSamplesNeeded
        warning('evaluateSegmentDMDs:segmentTooShort', ...
            '%s: %d samples, below the ~%d needed - skipping.', ...
            segmentName{k}, nSamplesArr(k), minSegmentSamplesNeeded);
        continue;
    end

    try
        evalc([ ...
            '[trS,~,teS,~] = splitTrainTestDMD(Xseg, [], trainFraction);' ...
            '[EigS,VecS,~,~,~,~] = runMovingWindowDMD(trS, EEG.srate, windowSamples, overlapSamples, r);' ...
            'repS = evaluateDMDOneStepFit(teS, EigS, VecS);' ]);
    catch ME
        warning('evaluateSegmentDMDs:segmentFailed', '%s: failed - %s', segmentName{k}, ME.message);
        continue;
    end

    evaluated(k)       = true;
    rmseOverall(k)     = repS.rmseOverall;
    r2Overall(k)       = repS.r2Overall;
    residualKSTestH(k) = repS.residualKSTestH;
    residualKSTestP(k) = repS.residualKSTestP;
    residualADTestH(k) = repS.residualADTestH;
    residualADTestP(k) = repS.residualADTestP;

    fprintf('%-45s  RMSE=%-10.4f R2=%-8.4f (n=%d)\n', segmentName{k}, rmseOverall(k), r2Overall(k), nSamplesArr(k));
end

resultsTable = table(segmentIdx, segmentName, nSamplesArr, evaluated, ...
    rmseOverall, r2Overall, residualKSTestH, residualKSTestP, residualADTestH, residualADTestP, ...
    'VariableNames', {'segmentIdx', 'segmentName', 'nSamples', 'evaluated', ...
        'rmseOverall', 'r2Overall', 'residualKSTestH', 'residualKSTestP', 'residualADTestH', 'residualADTestP'});

nEvaluated = sum(evaluated);
fprintf('\nevaluateSegmentDMDs: evaluated %d / %d segments (%d skipped - too short or failed).\n', ...
    nEvaluated, nSegments, nSegments - nEvaluated);
if nEvaluated > 0
    fprintf('  Median RMSE: %.4f | Median R^2: %.4f\n', median(rmseOverall(evaluated)), median(r2Overall(evaluated)));
end

end

function name = labelOneSegment(k, segInfo, onsetLatencies, onsetLabels)
%LABELONESEGMENT Build a human-readable name for segment k from the
% nearest onset events before/after it.
precedingIdx = find(onsetLatencies < segInfo.startSample, 1, 'last');
followingIdx = find(onsetLatencies > segInfo.endSample, 1, 'first');

if isempty(precedingIdx) && isempty(followingIdx)
    name = sprintf('Seg %d (only segment)', k);
elseif isempty(precedingIdx)
    name = sprintf('Seg %d: baseline, before %s', k, onsetLabels{followingIdx});
elseif isempty(followingIdx)
    name = sprintf('Seg %d: after %s (recording end)', k, onsetLabels{precedingIdx});
else
    name = sprintf('Seg %d: after %s, before %s', k, onsetLabels{precedingIdx}, onsetLabels{followingIdx});
end
end

function s = labelToStr(t)
%LABELTOSTR Convert an EEG.event(:).type value (numeric, char, or string) to a clean string label.
if isnumeric(t)
    s = num2str(t);
else
    s = strtrim(char(string(t)));
end
end

function tf = isAnyMarker(t, markers)
tf = false;
for k = 1:numel(markers)
    if isMatch(t, markers{k})
        tf = true;
        return;
    end
end
end

function tf = isMatch(t, marker)
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
    if ischar(t) || isstring(t)
        tf = strcmpi(strtrim(char(t)), strtrim(char(marker)));
    else
        tf = false;
    end
end
end
