function [Xtrain, keptIdxTrain, Xtest, keptIdxTest] = splitTrainTestDMD(X, keptSampleIdx, trainFraction)
%SPLITTRAINTESTDMD Split a DMD matrix into a train and held-out test portion.
%
%   [Xtrain, keptIdxTrain, Xtest, keptIdxTest] = SPLITTRAINTESTDMD(X, keptSampleIdx, trainFraction)
%
%   WHY THIS EXISTS
%   evaluateDMDOneStepFit measured "perfect" R^2 for the full-rank DMD
%   model because it was being tested on the exact same data it was fit
%   on - any reasonably parameterized linear model will ace that, so it
%   doesn't tell you anything about whether the model generalizes. The
%   honest test is to fit DMD on one stretch of the recording and check
%   one-step prediction only on a stretch it never saw. This function
%   does the split (by time, keeping the first trainFraction of samples
%   for training and the rest for testing) so you can feed the two
%   halves into runDMDOnMatrix/runMovingWindowDMD and
%   evaluateDMDOneStepFit separately.
%
%   INPUTS
%   X               : nTime x nChannels DMD matrix (e.g. Y from
%                     removeStimSegments).
%   keptSampleIdx   : nTime x 1 vector from removeStimSegments, OR [] if
%                     you don't have one (e.g. X is raw, contiguous
%                     data). Split the same way as X so the two stay in
%                     correspondence; pass the resulting keptIdxTrain/
%                     keptIdxTest straight into evaluateDMDOneStepFit.
%   trainFraction   : scalar in (0, 1), fraction of samples (by time,
%                     first chunk) to use for training. 0.7 is a
%                     reasonable default (70% train / 30% held-out test).
%
%   OUTPUTS
%   Xtrain, Xtest             : the two row-wise splits of X.
%   keptIdxTrain, keptIdxTest : the matching splits of keptSampleIdx, or
%                               [] for each if keptSampleIdx was [].
%
%   USAGE (fair, held-out comparison of full vs. moving-window DMD)
%       [Ytrain, keptTrain, Ytest, keptTest] = splitTrainTestDMD(Y, keptIdx, 0.7);
%
%       % Full DMD: fit on train, score one-step prediction on held-out test
%       [EigTr, VecTr, ~, ~, ~, ~] = runDMDOnMatrix(Ytrain, EEG.srate, 35);
%       reportFullHeldOut = evaluateDMDOneStepFit(Ytest, EigTr, VecTr, keptTest);
%
%       % Moving-window DMD: same idea
%       [EigMWTr, VecMWTr, ~, ~, ~, ~] = runMovingWindowDMD(Ytrain, EEG.srate, 2500, 1250, 15);
%       reportMWHeldOut = evaluateDMDOneStepFit(Ytest, EigMWTr, VecMWTr, keptTest);
%
%       fprintf('Full DMD   held-out: RMSE=%.4f  R2=%.4f\n', reportFullHeldOut.rmseOverall, reportFullHeldOut.r2Overall);
%       fprintf('Moving DMD held-out: RMSE=%.4f  R2=%.4f\n', reportMWHeldOut.rmseOverall, reportMWHeldOut.r2Overall);

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('splitTrainTestDMD:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
nTime = size(X, 1);

if ~isscalar(trainFraction) || trainFraction <= 0 || trainFraction >= 1
    error('splitTrainTestDMD:badFraction', 'trainFraction must be a scalar strictly between 0 and 1.');
end

if ~isempty(keptSampleIdx)
    keptSampleIdx = keptSampleIdx(:);
    if numel(keptSampleIdx) ~= nTime
        error('splitTrainTestDMD:keptIdxMismatch', ...
            'keptSampleIdx has %d entries but X has %d rows.', numel(keptSampleIdx), nTime);
    end
end

nTrain = round(trainFraction * nTime);
nTrain = max(1, min(nTime - 1, nTrain));  % keep at least 1 sample on each side

Xtrain = X(1:nTrain, :);
Xtest  = X(nTrain+1:end, :);

if isempty(keptSampleIdx)
    keptIdxTrain = [];
    keptIdxTest  = [];
else
    keptIdxTrain = keptSampleIdx(1:nTrain);
    keptIdxTest  = keptSampleIdx(nTrain+1:end);
end

fprintf('splitTrainTestDMD: %d train samples (%.1f%%), %d test samples (%.1f%%).\n', ...
    nTrain, 100 * nTrain / nTime, nTime - nTrain, 100 * (nTime - nTrain) / nTime);

end
