function resultsTable = compareBeforeAfterStimDMD(segments, EEG, windowSamples, overlapSamples, r, trainFraction)
%COMPAREBEFOREAFTERSTIMDMD Moving-window DMD held-out fit for every before/after stim pair.
%
%   resultsTable = COMPAREBEFOREAFTERSTIMDMD(segments, EEG, windowSamples, ...
%       overlapSamples, r, trainFraction)
%
%   Repeats the before-vs-after-stimulation workflow you just ran by hand
%   (splitTrainTestDMD -> runMovingWindowDMD on the train portion ->
%   evaluateDMDOneStepFit on the held-out test portion) for EVERY
%   consecutive pair of segments returned by windowStimSegments, instead
%   of just the one stimulation you picked with findSurroundingSegments.
%   segments{k} is treated as "before" the k-th gap and segments{k+1} as
%   "after" it, since that's exactly how windowStimSegments produces them
%   (each pair of consecutive returned segments has exactly one removed
%   stimulation period between them, UNLESS windowStimSegments dropped a
%   short segment via minSegmentSamples, in which case a pair can span
%   more than one stimulation event - segmentPairIdx below just counts
%   gaps in order, it doesn't re-derive which original event(s) each one
%   corresponds to).
%
%   Segments too short to usefully split/window are skipped (with a
%   warning identifying which pair and why) rather than erroring out the
%   whole run, since recordings can have some gaps much shorter than
%   others.
%
%   INPUTS
%   segments        : 1xK cell array from windowStimSegments.
%   EEG             : the EEG struct (only EEG.srate is used).
%   windowSamples, overlapSamples, r : passed straight through to
%                     runMovingWindowDMD for both the before- and
%                     after-segment fits.
%   trainFraction   : passed straight through to splitTrainTestDMD (e.g.
%                     0.7 for a 70/30 train/test split within each
%                     segment).
%
%   OUTPUT
%   resultsTable    : table, one row per before/after pair, with columns:
%       segmentPairIdx, evaluated,
%       beforeNSamples, beforeRMSE, beforeR2, beforeKSH, beforeADH,
%       afterNSamples,  afterRMSE,  afterR2,  afterKSH,  afterADH
%     'evaluated' is false (and the rest NaN) for any pair that was
%     skipped because one of its segments was too short.
%
%   USAGE
%       [segments, segIdx, segmentInfo] = windowStimSegments(EEG, X, 0, 200);
%       resultsTable = compareBeforeAfterStimDMD(segments, EEG, 2500, 1250, 15, 0.7);
%       disp(resultsTable)
%       figure; boxplot([resultsTable.beforeR2, resultsTable.afterR2], ...
%           'Labels', {'Before stim','After stim'});
%       ylabel('Held-out one-step R^2'); title('DMD fit quality: before vs. after stimulation');

if ~iscell(segments) || numel(segments) < 2
    error('compareBeforeAfterStimDMD:badSegments', ...
        'segments must be a cell array (from windowStimSegments) with at least 2 entries.');
end
if ~isfield(EEG, 'srate')
    error('compareBeforeAfterStimDMD:noSrate', 'EEG.srate is missing.');
end

nPairs = numel(segments) - 1;

segmentPairIdx  = (1:nPairs)';
evaluated       = false(nPairs, 1);
beforeNSamples  = NaN(nPairs, 1);
beforeRMSE      = NaN(nPairs, 1);
beforeR2        = NaN(nPairs, 1);
beforeKSH       = NaN(nPairs, 1);
beforeADH       = NaN(nPairs, 1);
afterNSamples   = NaN(nPairs, 1);
afterRMSE       = NaN(nPairs, 1);
afterR2         = NaN(nPairs, 1);
afterKSH        = NaN(nPairs, 1);
afterADH        = NaN(nPairs, 1);

% Rough minimum length so the train split leaves enough samples for at
% least a few DMD windows; anything shorter gets skipped rather than
% allowed to error out of runMovingWindowDMD/splitTrainTestDMD partway
% through the loop.
minSegmentSamplesNeeded = ceil(windowSamples / (1 - trainFraction)) * 3;

for k = 1:nPairs
    segBefore = segments{k};
    segAfter  = segments{k + 1};
    beforeNSamples(k) = size(segBefore, 1);
    afterNSamples(k)  = size(segAfter, 1);

    if size(segBefore, 1) < minSegmentSamplesNeeded || size(segAfter, 1) < minSegmentSamplesNeeded
        warning('compareBeforeAfterStimDMD:segmentTooShort', ...
            'Pair %d/%d: before=%d / after=%d samples, below the ~%d needed - skipping.', ...
            k, nPairs, beforeNSamples(k), afterNSamples(k), minSegmentSamplesNeeded);
        continue;
    end

    try
        evalc([ ...
            '[trB,~,teB,~] = splitTrainTestDMD(segBefore, [], trainFraction);' ...
            '[EigB,VecB,~,~,~,~] = runMovingWindowDMD(trB, EEG.srate, windowSamples, overlapSamples, r);' ...
            'repB = evaluateDMDOneStepFit(teB, EigB, VecB);' ]);
        beforeRMSE(k) = repB.rmseOverall;
        beforeR2(k)   = repB.r2Overall;
        beforeKSH(k)  = repB.residualKSTestH;
        beforeADH(k)  = repB.residualADTestH;
    catch ME
        warning('compareBeforeAfterStimDMD:beforeFailed', 'Pair %d/%d before-segment failed: %s', k, nPairs, ME.message);
        continue;
    end

    try
        evalc([ ...
            '[trA,~,teA,~] = splitTrainTestDMD(segAfter, [], trainFraction);' ...
            '[EigA,VecA,~,~,~,~] = runMovingWindowDMD(trA, EEG.srate, windowSamples, overlapSamples, r);' ...
            'repA = evaluateDMDOneStepFit(teA, EigA, VecA);' ]);
        afterRMSE(k) = repA.rmseOverall;
        afterR2(k)   = repA.r2Overall;
        afterKSH(k)  = repA.residualKSTestH;
        afterADH(k)  = repA.residualADTestH;
    catch ME
        warning('compareBeforeAfterStimDMD:afterFailed', 'Pair %d/%d after-segment failed: %s', k, nPairs, ME.message);
        continue;
    end

    evaluated(k) = true;
    fprintf('Pair %d/%d: before R^2=%.4f RMSE=%.4f | after R^2=%.4f RMSE=%.4f\n', ...
        k, nPairs, beforeR2(k), beforeRMSE(k), afterR2(k), afterRMSE(k));
end

deltaR2   = afterR2 - beforeR2;
deltaRMSE = afterRMSE - beforeRMSE;

resultsTable = table(segmentPairIdx, evaluated, ...
    beforeNSamples, beforeRMSE, beforeR2, beforeKSH, beforeADH, ...
    afterNSamples, afterRMSE, afterR2, afterKSH, afterADH, ...
    deltaR2, deltaRMSE);

nEvaluated = sum(evaluated);
fprintf('\ncompareBeforeAfterStimDMD: evaluated %d / %d pairs (%d skipped - too short or failed).\n', ...
    nEvaluated, nPairs, nPairs - nEvaluated);
if nEvaluated > 0
    fprintf('  Median held-out R^2 - before: %.4f | after: %.4f\n', ...
        median(beforeR2(evaluated)), median(afterR2(evaluated)));
    % NOTE: comparing the overall before/after medians above is misleading on
    % its own - segments{k+1} is simultaneously "after" pair k and "before"
    % pair k+1, so the before/after lists are the same values shifted by one
    % position, not independent groups. The per-pair deltaR2 (after - before,
    % computed within the SAME row/gap) below is the comparison that actually
    % reflects whether that specific stimulation changed anything.
    validDelta = deltaR2(evaluated);
    nPos = sum(validDelta > 0);
    nNeg = sum(validDelta < 0);
    fprintf('  deltaR2 (after - before) per event: mean=%.4f, median=%.4f\n', ...
        mean(validDelta), median(validDelta));
    fprintf('  %d/%d events show improved fit after stim, %d/%d show worse fit after stim\n', ...
        nPos, nEvaluated, nNeg, nEvaluated);
end

end
