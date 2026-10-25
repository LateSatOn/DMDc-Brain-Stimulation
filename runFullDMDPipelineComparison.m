%RUNFULLDMDPIPELINECOMPARISON End-to-end DMD pipeline, parameterized for apples-to-apples reruns.
%
%   Runs the whole project pipeline built so far - load, convert, remove/
%   segment around stimulation, Hankel-embed, moving-window DMD, and all
%   three evaluation layers (one-step, horizon sweep, per-segment table)
%   - and saves everything to a tagged .mat file. Run this unchanged on a
%   different recording (e.g. the planned higher-sample-rate EEG) by
%   only editing the EDIT-ME block below, then compare the two saved
%   results files directly (see bottom of this script for how).
%
%   This is a SCRIPT, not a function - run it with F5 / Run, or paste
%   section by section into the command window.

%% ======================= EDIT-ME: per-dataset settings =======================
folderPath   = "C:\Users\Layth\Downloads\eeg";  % BIDS folder for this recording
runLabel     = "lowFreq250";                     % tag for saved results, e.g. "lowFreq250" / "highFreq1000"

bufferSamples      = 0;      % removeStimSegments/windowStimSegments buffer after offset
minSegmentSamples  = 200;    % windowStimSegments: drop segments shorter than this

d                  = 5;      % Hankel embedding delay steps
windowSamples      = 2500;   % runMovingWindowDMD window length (samples)
overlapSamples     = 1250;   % runMovingWindowDMD window overlap (samples)
r                  = 15;     % number of DMD modes to keep

trainFraction      = 0.7;    % splitTrainTestDMD train fraction
horizonsToSweep    = [1 5 10 25 50 100 250 500]; % evaluateDMDHorizonSweep horizons
segmentIdxForSplit = 2;      % which segment to use for the single train/test demo fit

%% ======================= Load and convert =======================
EEG = loadEEGSet(folderPath);
X = eegToDMDMatrix(EEG);

fprintf('--- %s: EEG.srate = %d Hz, %d samples, %d channels ---\n', ...
    runLabel, EEG.srate, size(X, 1), size(X, 2));

%% ======================= Segment around stimulation =======================
[segments, segIdx, segmentInfo] = windowStimSegments(EEG, X, bufferSamples, minSegmentSamples);

%% ======================= Hankel-embed every segment =======================
hankelSegments = cell(size(segments));
for k = 1:numel(segments)
    hankelSegments{k} = buildHankelMatrix(segments{k}, d);
end

%% ======================= Combined natural-dynamics model across all segments =======================
[segResultsHankel, EigHankel, VecHankel, AmpHankel, FreqHankel, GrowthHankel, PODeHankel] = ...
    combineSegmentDMDs(hankelSegments, EEG, windowSamples, overlapSamples, r);

%% ======================= Held-out one-step and horizon-sweep fit (one representative segment) =======================
[HTrain, ~, HTest, ~] = splitTrainTestDMD(hankelSegments{segmentIdxForSplit}, [], trainFraction);
[EigH, VecH, ~, ~, ~, ~] = runMovingWindowDMD(HTrain, EEG.srate, windowSamples, overlapSamples, r);

oneStepReport = evaluateDMDOneStepFit(HTest, EigH, VecH);
sweepTable    = evaluateDMDHorizonSweep(HTest, EigH, VecH, horizonsToSweep, []);
disp(sweepTable)

%% ======================= Full per-segment table (Hankel-embedded, not just one segment) =======================
hankelSegmentInfo = segmentInfo; % startSample/endSample still describe the pre-Hankel segments for naming purposes
segmentTable = evaluateSegmentDMDs(hankelSegments, hankelSegmentInfo, EEG, windowSamples, overlapSamples, r, trainFraction);
disp(segmentTable)

%% ======================= Visual check =======================
plotDMDOneStepPrediction(HTest, EigH, VecH, 1, 1:3000);
plotAllChannelsDMDOneStepPrediction(HTest, EigH, VecH, 1:3000);

%% ======================= Save everything, tagged with runLabel =======================
resultsFileName = sprintf('dmdResults_%s.mat', runLabel);
save(resultsFileName, 'runLabel', 'folderPath', 'EEG', ...
    'bufferSamples', 'minSegmentSamples', 'd', 'windowSamples', 'overlapSamples', 'r', 'trainFraction', ...
    'segmentInfo', 'segResultsHankel', 'EigHankel', 'VecHankel', 'AmpHankel', 'FreqHankel', 'GrowthHankel', 'PODeHankel', ...
    'oneStepReport', 'sweepTable', 'segmentTable');
fprintf('Saved results to %s\n', resultsFileName);

%% ======================= To compare two runs later =======================
% lowRes  = load('dmdResults_lowFreq250.mat');
% highRes = load('dmdResults_highFreq1000.mat');
%
% fprintf('One-step R^2:   low=%.4f  high=%.4f\n', lowRes.oneStepReport.r2Overall, highRes.oneStepReport.r2Overall);
% disp(table(lowRes.sweepTable.horizon, lowRes.sweepTable.r2Overall, highRes.sweepTable.r2Overall, ...
%     'VariableNames', {'horizon', 'r2Overall_low', 'r2Overall_high'}));
