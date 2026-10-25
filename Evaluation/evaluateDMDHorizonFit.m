function fitReport = evaluateDMDHorizonFit(X, Eigenvalues, Eigenvectors, horizon, keptSampleIdx)
%EVALUATEDMDHORIZONFIT Multi-step-ahead (horizon) goodness-of-fit for a DMD model.
%
%   fitReport = EVALUATEDMDHORIZONFIT(X, Eigenvalues, Eigenvectors, horizon)
%   fitReport = EVALUATEDMDHORIZONFIT(X, Eigenvalues, Eigenvectors, horizon, keptSampleIdx)
%
%   Generalizes evaluateDMDOneStepFit from "predict 1 sample ahead" to
%   "predict `horizon` samples ahead" - still re-anchored on the REAL
%   data every time (predict x(t+horizon) from the actual x(t), not from
%   the model's own earlier guess), so nothing compounds across more than
%   one jump of size `horizon`. This sits between the one-step test
%   (always looks great for EEG, since adjacent samples barely change -
%   not a very demanding test) and the full open-loop forecast in
%   evaluateDMDFit/evaluateMovingWindowDMDFit (lets error compound across
%   the ENTIRE window/recording with no correction, which can look like a
%   total non-fit even for a locally-good model). Sweeping `horizon` up
%   (see evaluateDMDHorizonSweep) directly answers "how far into the
%   future can this model actually predict something useful."
%
%   Rebuilds the horizon-step operator as A^horizon = Phi * diag(lambda
%   .^ horizon) * pinv(Phi) directly from the eigendecomposition (exact,
%   not computed by repeatedly multiplying A by itself, since pinv(Phi) *
%   Phi = I for a full-column-rank Phi - no extra numerical error from
%   iterating).
%
%   INPUTS
%   X               : nTime x nChannels matrix the DMD was fit on.
%   Eigenvalues     : r x 1 complex (from runDMDOnMatrix or runMovingWindowDMD).
%   Eigenvectors    : r x nChannels complex (from either of those).
%   horizon         : positive integer, how many samples ahead to predict
%                     (1 reduces to the same thing as evaluateDMDOneStepFit).
%   keptSampleIdx   : OPTIONAL, nTime x 1 vector from removeStimSegments.
%                     If X is a concatenation of non-contiguous segments,
%                     pass this so any (t, t+horizon) pair that straddles
%                     a removed gap is excluded - otherwise a span this
%                     long is much more likely to cross a gap than a
%                     single one-step transition was.
%
%   OUTPUT
%   fitReport       : same fields as evaluateDMDOneStepFit's output, plus:
%       .horizon                - the horizon this was evaluated at
%
%   USAGE (single horizon)
%       report100 = evaluateDMDHorizonFit(XbeforeTest, EigBeforeMW, VecBeforeMW, 100);
%       fprintf('100-step: RMSE=%.4f R2=%.4f\n', report100.rmseOverall, report100.r2Overall);
%
%   See evaluateDMDHorizonSweep to compare several horizons at once in one table.

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('evaluateDMDHorizonFit:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

Eigenvalues = Eigenvalues(:);
r = numel(Eigenvalues);

if size(Eigenvectors, 1) ~= r || size(Eigenvectors, 2) ~= nChannels
    error('evaluateDMDHorizonFit:badEigenvectors', ...
        'Eigenvectors must be r x nChannels (%d x %d); got %dx%d.', ...
        r, nChannels, size(Eigenvectors, 1), size(Eigenvectors, 2));
end

if ~isscalar(horizon) || horizon < 1 || horizon ~= round(horizon)
    error('evaluateDMDHorizonFit:badHorizon', 'horizon must be a positive integer.');
end
if horizon >= nTime
    error('evaluateDMDHorizonFit:horizonTooLarge', ...
        'horizon (%d) must be smaller than size(X,1) (%d).', horizon, nTime);
end

absEig = abs(Eigenvalues);
nUnstableModes = sum(absEig > 1);
maxAbsEigenvalue = max(absEig);

% ---- build the exact horizon-step operator from the eigendecomposition ----
PhiMat = Eigenvectors.';                              % nChannels x r
Ah = PhiMat * diag(Eigenvalues .^ horizon) * pinv(PhiMat); % nChannels x nChannels

% ---- predict x(t+horizon) from the real x(t), re-anchored every time ----
Xm = X - mean(X, 1);
Xcurrent = Xm(1:end-horizon, :);
Xnext    = Xm(1+horizon:end, :);
Xpred    = real(Xcurrent * Ah.');

nExcludedGapTransitions = 0;
if nargin >= 5 && ~isempty(keptSampleIdx)
    keptSampleIdx = keptSampleIdx(:);
    if numel(keptSampleIdx) ~= nTime
        error('evaluateDMDHorizonFit:keptIdxMismatch', ...
            'keptSampleIdx has %d entries but X has %d rows.', numel(keptSampleIdx), nTime);
    end
    % A (t, t+horizon) pair is only a real, contiguous span if the original
    % sample indices advanced by exactly `horizon` - any removed gap in
    % between would make this difference larger than horizon.
    validMask = (keptSampleIdx(1+horizon:end) - keptSampleIdx(1:end-horizon)) == horizon;
    nExcludedGapTransitions = sum(~validMask);

    Xcurrent = Xcurrent(validMask, :); %#ok<NASGU>
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

resFlat = residuals(:);
resStd = (resFlat - mean(resFlat)) / std(resFlat);

try
    [ksH, ksP] = kstest(resStd);
catch
    ksH = NaN; ksP = NaN;
end

try
    [adH, adP] = adtest(resStd);
catch
    adH = NaN; adP = NaN;
end

fitReport = struct( ...
    'horizon', horizon, ...
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

fprintf('=== DMD %d-Step-Ahead Goodness-of-Fit ===\n', horizon);
fprintf('  Samples evaluated     : %d (excluded %d gap-crossing transitions)\n', ...
    nSamplesEvaluated, nExcludedGapTransitions);
fprintf('  RMSE (overall)        : %.4f\n', rmseOverall);
fprintf('  R^2  (overall)        : %.4f\n', r2Overall);
if ~isnan(ksH)
    fprintf('  Residual K-S test     : H=%d p=%.4g\n', ksH, ksP);
end
if ~isnan(adH)
    fprintf('  Residual A-D test     : H=%d p=%.4g\n', adH, adP);
end
fprintf('\n');

end
