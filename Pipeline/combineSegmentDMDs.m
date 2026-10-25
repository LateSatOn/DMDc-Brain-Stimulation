function [segmentResults, Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, GrowthRates, POD_Mode_Energies] = combineSegmentDMDs(segments, EEG, windowSamples, overlapSamples, r)
%COMBINESEGMENTDMDS Moving-window DMD on every between-stim segment, then combined into one model.
%
%   [segmentResults, Eigenvalues, Eigenvectors, ModeAmplitudes, ...
%    ModeFrequencies, GrowthRates, POD_Mode_Energies] = ...
%       COMBINESEGMENTDMDS(segments, EEG, windowSamples, overlapSamples, r)
%
%   This is the actual next step in characterizing the recording's
%   natural dynamics: treat EACH segment returned by windowStimSegments
%   (i.e. each contiguous stretch of data between stimulations) as its
%   own dynamical regime, fit runMovingWindowDMD on the WHOLE segment,
%   and then combine all of those segment-level models into one overall
%   model - using the exact same greedy eigenvalue-matching-then-
%   averaging runMovingWindowDMD already uses across windows, just one
%   level up (across segments instead of across windows within one
%   segment).
%
%   Note this is NOT a train/test split: unlike
%   evaluateDMDOneStepFit/splitTrainTestDMD, which exist to SCORE whether
%   a model generalizes, this function exists to EXTRACT the best
%   available description of each segment's dynamics and pool them - so
%   it fits each segment on all of its own data.
%
%   INPUTS
%   segments        : 1xK cell array from windowStimSegments.
%   EEG             : EEG struct (EEG.srate is used).
%   windowSamples, overlapSamples, r : passed straight through to
%                     runMovingWindowDMD for every segment.
%
%   OUTPUTS
%   segmentResults  : struct array, one entry PER SEGMENT THAT WAS
%                     SUCCESSFULLY FIT (segments shorter than
%                     windowSamples are skipped, with a warning, and
%                     excluded from everything below). Fields:
%                       .segmentIdx         - which entry of `segments`
%                                              this came from
%                       .nSamples           - length of that segment
%                       .Eigenvalues, .Eigenvectors, .ModeAmplitudes,
%                       .ModeFrequencies, .GrowthRates,
%                       .POD_Mode_Energies  - that segment's own
%                                              averaged moving-window DMD
%                     Use this to inspect or group individual segments
%                     (e.g. by before/after a specific stimulation, or by
%                     StimType) rather than only looking at the combined
%                     model below.
%   Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies,
%   GrowthRates, POD_Mode_Energies : the SAME quantities, mode-matched to
%                     segmentResults(1)'s ordering and averaged across
%                     EVERY successfully-fit segment - one combined
%                     "natural dynamics" model for the whole recording's
%                     between-stimulation behavior.
%
%   USAGE
%       [segments, segIdx, segmentInfo] = windowStimSegments(EEG, X, 0, 200);
%       [segResults, Eig, Vec, Amp, Freq, Growth, PODe] = ...
%           combineSegmentDMDs(segments, EEG, 2500, 1250, 15);
%
%       % one segment's own dynamics:
%       segResults(3).ModeFrequencies
%
%       % the combined, overall natural-dynamics model:
%       theta = linspace(0, 2*pi, 200);
%       figure; plot(cos(theta), sin(theta), 'k--'); hold on;
%       scatter(real(Eig), imag(Eig), 60, abs(Eig), 'filled');
%       axis equal; grid on; title('Combined natural dynamics across all between-stim segments');

if ~iscell(segments) || isempty(segments)
    error('combineSegmentDMDs:badSegments', 'segments must be a nonempty cell array from windowStimSegments.');
end
if ~isfield(EEG, 'srate')
    error('combineSegmentDMDs:noSrate', 'EEG.srate is missing.');
end

nSegmentsTotal = numel(segments);

segmentResults = struct('segmentIdx', {}, 'nSamples', {}, 'Eigenvalues', {}, ...
    'Eigenvectors', {}, 'ModeAmplitudes', {}, 'ModeFrequencies', {}, ...
    'GrowthRates', {}, 'POD_Mode_Energies', {});

for k = 1:nSegmentsTotal
    Xseg = segments{k};
    nSamplesSeg = size(Xseg, 1);

    if nSamplesSeg < windowSamples
        warning('combineSegmentDMDs:segmentTooShort', ...
            'Segment %d/%d has %d samples, below windowSamples (%d) - skipping.', ...
            k, nSegmentsTotal, nSamplesSeg, windowSamples);
        continue;
    end

    try
        [eVals, eVecs, amps, freqs, growth, podE] = ...
            runMovingWindowDMD(Xseg, EEG.srate, windowSamples, overlapSamples, r);
    catch ME
        warning('combineSegmentDMDs:segmentFailed', ...
            'Segment %d/%d failed: %s - skipping.', k, nSegmentsTotal, ME.message);
        continue;
    end

    newEntry = struct('segmentIdx', k, 'nSamples', nSamplesSeg, ...
        'Eigenvalues', eVals, 'Eigenvectors', eVecs, 'ModeAmplitudes', amps, ...
        'ModeFrequencies', freqs, 'GrowthRates', growth, 'POD_Mode_Energies', podE);
    segmentResults(end + 1) = newEntry; %#ok<AGROW>
end

nSegmentsUsed = numel(segmentResults);
if nSegmentsUsed == 0
    error('combineSegmentDMDs:noUsableSegments', ...
        'No segment was long enough to fit (all shorter than windowSamples = %d).', windowSamples);
end

% ---- mode-match every segment's result to segment 1's ordering, then average ----
r_ = numel(segmentResults(1).Eigenvalues);
nChannels = size(segmentResults(1).Eigenvectors, 2);
refEig = segmentResults(1).Eigenvalues;

matchedEig    = zeros(r_, nSegmentsUsed);
matchedVec    = zeros(r_, nChannels, nSegmentsUsed);
matchedAmp    = zeros(r_, nSegmentsUsed);
matchedFreq   = zeros(r_, nSegmentsUsed);
matchedGrowth = zeros(r_, nSegmentsUsed);

matchedEig(:, 1)    = segmentResults(1).Eigenvalues;
matchedVec(:, :, 1) = segmentResults(1).Eigenvectors;
matchedAmp(:, 1)    = segmentResults(1).ModeAmplitudes;
matchedFreq(:, 1)   = segmentResults(1).ModeFrequencies;
matchedGrowth(:, 1) = segmentResults(1).GrowthRates;

for s = 2:nSegmentsUsed
    curEig = segmentResults(s).Eigenvalues;
    if numel(curEig) ~= r_
        warning('combineSegmentDMDs:rankMismatch', ...
            'Segment index %d has a different number of modes than segment 1 - skipping it in the combined average.', ...
            segmentResults(s).segmentIdx);
        matchedEig(:, s)    = NaN;
        matchedVec(:, :, s) = NaN;
        matchedAmp(:, s)    = NaN;
        matchedFreq(:, s)   = NaN;
        matchedGrowth(:, s) = NaN;
        continue;
    end
    assignment = matchModesGreedy(refEig, curEig);
    matchedEig(:, s)    = curEig(assignment);
    matchedVec(:, :, s) = segmentResults(s).Eigenvectors(assignment, :);
    matchedAmp(:, s)    = segmentResults(s).ModeAmplitudes(assignment);
    matchedFreq(:, s)   = segmentResults(s).ModeFrequencies(assignment);
    matchedGrowth(:, s) = segmentResults(s).GrowthRates(assignment);
end

Eigenvalues     = mean(matchedEig, 2, 'omitnan');
Eigenvectors    = mean(matchedVec, 3, 'omitnan');
ModeAmplitudes  = mean(matchedAmp, 2, 'omitnan');
ModeFrequencies = mean(matchedFreq, 2, 'omitnan');
GrowthRates     = mean(matchedGrowth, 2, 'omitnan');

podLengths = arrayfun(@(s) numel(s.POD_Mode_Energies), segmentResults);
minPodLen = min(podLengths);
podMat = zeros(minPodLen, nSegmentsUsed);
for s = 1:nSegmentsUsed
    podMat(:, s) = segmentResults(s).POD_Mode_Energies(1:minPodLen);
end
POD_Mode_Energies = mean(podMat, 2);

fprintf('combineSegmentDMDs: combined %d / %d segments into one natural-dynamics model (r=%d modes).\n', ...
    nSegmentsUsed, nSegmentsTotal, r_);

end

function assignment = matchModesGreedy(refEig, curEig)
%MATCHMODESGREEDY Greedy nearest-neighbor match of curEig onto refEig's order.
%   Same approach as runMovingWindowDMD's internal mode matching - locks
%   in the globally closest remaining pair first, each curEig index used
%   at most once. Not a globally optimal (Hungarian) assignment, but good
%   enough when modes are reasonably well separated in the complex plane.
r_ = numel(refEig);
assignment = zeros(r_, 1);
usedCols = false(1, r_);

distMat = abs(refEig(:) - curEig(:).');

for n = 1:r_
    distMat(:, usedCols) = Inf;
    rowsAssigned = assignment ~= 0;
    distMat(rowsAssigned, :) = Inf;

    [~, linIdx] = min(distMat(:));
    [i, j] = ind2sub(size(distMat), linIdx);

    assignment(i) = j;
    usedCols(j) = true;
end
end
