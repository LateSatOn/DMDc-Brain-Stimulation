function fitReport = evaluateDMDOneStepFit(X, Eigenvalues, Eigenvectors, keptSampleIdx)
%EVALUATEDMDONESTEPFIT One-step-ahead goodness-of-fit for a DMD model.
%
%   fitReport = EVALUATEDMDONESTEPFIT(X, Eigenvalues, Eigenvectors)
%   fitReport = EVALUATEDMDONESTEPFIT(X, Eigenvalues, Eigenvectors, keptSampleIdx)
%
%   WHY THIS EXISTS (vs. evaluateDMDFit / evaluateMovingWindowDMDFit)
%   Those two score DMD as an OPEN-LOOP FORECAST: start from one initial
%   snapshot and let the model run forward on its own for the whole
%   window/recording. For oscillatory, roughly-constant-amplitude signals
%   like EEG, any eigenvalue with |lambda| even slightly below 1 decays
%   toward zero over a long enough horizon (0.99^2500 is already ~0), so
%   open-loop forecasts collapse to ~0 and look like a total non-fit even
%   when the model's underlying frequencies/operator are actually
%   reasonable. That's a property of forecasting long horizons with a
%   slightly-damped linear model, not necessarily evidence the DMD fit
%   itself is bad.
%
%   This function asks a different, standard question instead: "if you
%   give the model the REAL previous sample, how well does it predict
%   just the NEXT one?" It reconstructs the (reduced-rank) DMD state
%   operator A from (Eigenvalues, Eigenvectors) and applies
%       x_pred(t+1) = A * x_actual(t)
%   re-anchoring to the real data at every step, so nothing compounds or
%   decays across a long horizon - this is the metric DMD is actually
%   suited for, and is a much more sensitive test of whether the fitted
%   dynamics are locally correct.
%
%   INPUTS
%   X               : nTime x nChannels matrix the DMD was fit on (same
%                     matrix passed into runDMDOnMatrix / runMovingWindowDMD).
%   Eigenvalues     : r x 1 complex (from either runDMDOnMatrix or
%                     runMovingWindowDMD - works the same way for both,
%                     since this doesn't raise eigenvalues to any large
%                     power).
%   Eigenvectors    : r x nChannels complex (from either of those).
%   keptSampleIdx   : OPTIONAL, nTime x 1 vector of original sample
%                     indices, exactly as returned by removeStimSegments.
%                     If X came from removeStimSegments (i.e. it's a
%                     concatenation of non-contiguous time segments with
%                     stimulation cut out), pass this so the function can
%                     find and EXCLUDE the handful of one-step
%                     transitions that happen to straddle a removed gap -
%                     those are fake jumps in time, not model error, and
%                     would otherwise unfairly inflate the error. Omit
%                     this if X is already time-contiguous (e.g. raw
%                     data, or one segment from windowStimSegments).
%
%   OUTPUT
%   fitReport       : struct with:
%       .rmseOverall, .rmsePerChannel   - one-step prediction RMSE
%       .r2Overall, .r2PerChannel       - one-step prediction R^2
%       .nUnstableModes, .maxAbsEigenvalue
%       .residualKSTestH/.residualKSTestP   - K-S test, residuals vs normal
%       .residualADTestH/.residualADTestP   - Anderson-Darling, same idea
%       .nSamplesEvaluated              - how many one-step transitions
%                                          were actually scored
%       .nExcludedGapTransitions        - how many were skipped because
%                                          keptSampleIdx flagged them as
%                                          crossing a removed-stim gap
%
%   USAGE
%       [Eig, Vec, Amp, ~, ~, ~] = runDMDOnMatrix(Y, EEG.srate, 35);
%       reportFull = evaluateDMDOneStepFit(Y, Eig, Vec, keptIdx);
%
%       [EigMW, VecMW, ~, ~, ~, ~] = runMovingWindowDMD(Y, EEG.srate, 2500, 1250, 15);
%       reportMW = evaluateDMDOneStepFit(Y, EigMW, VecMW, keptIdx);
%
%       fprintf('Full DMD   one-step: RMSE=%.4f  R2=%.4f\n', reportFull.rmseOverall, reportFull.r2Overall);
%       fprintf('Moving DMD one-step: RMSE=%.4f  R2=%.4f\n', reportMW.rmseOverall, reportMW.r2Overall);

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('evaluateDMDOneStepFit:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

Eigenvalues = Eigenvalues(:);
r = numel(Eigenvalues);

if size(Eigenvectors, 1) ~= r || size(Eigenvectors, 2) ~= nChannels
    error('evaluateDMDOneStepFit:badEigenvectors', ...
        'Eigenvectors must be r x nChannels (%d x %d); got %dx%d.', ...
        r, nChannels, size(Eigenvectors, 1), size(Eigenvectors, 2));
end

if nTime < 2
    error('evaluateDMDOneStepFit:tooShort', 'X must have at least 2 time samples.');
end

% ---- unstable mode check (informational - doesn't blow up one-step fit, but still worth knowing) ----
absEig = abs(Eigenvalues);
nUnstableModes = sum(absEig > 1);
maxAbsEigenvalue = max(absEig);

% ---- reconstruct the (reduced-rank) state operator A from the modes ----
% Our convention stores Eigenvectors as r x nChannels; the math wants
% modes as nChannels x r (one mode per column), so transpose back.
PhiMat = Eigenvectors.';                 % nChannels x r
A = PhiMat * diag(Eigenvalues) * pinv(PhiMat); % nChannels x nChannels, rank r

% ---- one-step prediction over every consecutive pair ----
Xm = X - mean(X, 1);
Xcurrent = Xm(1:end-1, :);   % (nTime-1) x nChannels
Xnext    = Xm(2:end, :);     % (nTime-1) x nChannels
Xpred    = real(Xcurrent * A.');  % x_pred(t+1) = A * x(t), applied row-wise

% ---- exclude one-step transitions that cross a removed-stim gap, if flagged ----
nExcludedGapTransitions = 0;
if nargin >= 4 && ~isempty(keptSampleIdx)
    keptSampleIdx = keptSampleIdx(:);
    if numel(keptSampleIdx) ~= nTime
        error('evaluateDMDOneStepFit:keptIdxMismatch', ...
            'keptSampleIdx has %d entries but X has %d rows.', numel(keptSampleIdx), nTime);
    end
    gapAfter = find(diff(keptSampleIdx) > 1);  % transition index t -> t+1 that is a fake jump in time
    validMask = true(nTime - 1, 1);
    validMask(gapAfter) = false;
    nExcludedGapTransitions = sum(~validMask);

    Xcurrent = Xcurrent(validMask, :);
    Xnext    = Xnext(validMask, :);
    Xpred    = Xpred(validMask, :);
end

nSamplesEvaluated = size(Xnext, 1);

% ---- fit metrics ----
residuals = Xnext - Xpred;

rmsePerChannel = sqrt(mean(residuals.^2, 1))';
rmseOverall = sqrt(mean(residuals(:).^2));

ssRes = sum(residuals.^2, 1);
ssTot = sum(Xnext.^2, 1);
r2PerChannel = (1 - ssRes ./ ssTot)';
r2Overall = 1 - sum(ssRes) / sum(ssTot);

% ---- goodness-of-fit test on residuals ----
resFlat = residuals(:);
resStd = (resFlat - mean(resFlat)) / std(resFlat);

try
    [ksH, ksP] = kstest(resStd);
catch
    ksH = NaN; ksP = NaN;
    warning('evaluateDMDOneStepFit:noKSTest', 'kstest unavailable (needs Statistics and Machine Learning Toolbox).');
end

try
    [adH, adP] = adtest(resStd);
catch
    adH = NaN; adP = NaN;
end

fitReport = struct( ...
    'rmseOverall', rmseOverall, ...
    'rmsePerChannel', rmsePerChannel, ...
    'r2Overall', r2Overall, ...
    'r2PerChannel', r2PerChannel, ...
    'nUnstableModes', nUnstableModes, ...
    'maxAbsEigenvalue', maxAbsEigenvalue, ...
    'residualKSTestH', ksH, ...
    'residualKSTestP', ksP, ...
    'residualADTestH', adH, ...
    'residualADTestP', adP, ...
    'nSamplesEvaluated', nSamplesEvaluated, ...
    'nExcludedGapTransitions', nExcludedGapTransitions);

fprintf('=== DMD One-Step-Ahead Goodness-of-Fit ===\n');
fprintf('  Samples evaluated     : %d (excluded %d gap-crossing transitions)\n', ...
    nSamplesEvaluated, nExcludedGapTransitions);
fprintf('  RMSE (overall)        : %.4f\n', rmseOverall);
fprintf('  R^2  (overall)        : %.4f\n', r2Overall);
fprintf('  Unstable modes        : %d / %d (max |lambda| = %.4f)\n', nUnstableModes, r, maxAbsEigenvalue);
if ~isnan(ksH)
    fprintf('  Residual K-S test     : H=%d (1=non-normal/structured) p=%.4g\n', ksH, ksP);
end
if ~isnan(adH)
    fprintf('  Residual A-D test     : H=%d (1=non-normal/structured) p=%.4g\n', adH, adP);
end
fprintf('\n');

end
