function fitReport = evaluateDMDFit(X, Eigenvalues, Eigenvectors, ModeAmplitudes)
%EVALUATEDMDFIT Goodness-of-fit report for a DMD model against its data.
%
%   fitReport = EVALUATEDMDFIT(X, Eigenvalues, Eigenvectors, ModeAmplitudes)
%
%   DMD doesn't fit a probability distribution, so classic goodness-of-
%   fit tests (chi-squared, Kolmogorov-Smirnov, Anderson-Darling) don't
%   apply directly to X itself. What DOES carry over from that framework
%   is the core idea: reconstruct what the model predicts, compare it to
%   the real data, and test whether what's LEFT OVER (the residual) looks
%   like structureless noise. If a DMD model is "good enough", the
%   residual should have no remaining temporal/spatial structure, i.e. it
%   should look like noise - which is exactly what an EDF-style test
%   (K-S/Anderson-Darling against a normal) checks for.
%
%   This function:
%     1. Reconstructs the model's predicted time series from
%        (Eigenvalues, Eigenvectors, ModeAmplitudes) - the same DMD
%        output convention as runDMDOnMatrix / runMovingWindowDMD.
%     2. Compares it to X (mean-subtracted, since DMD here is always fit
%        on mean-subtracted data - see runDMDOnMatrix) via RMSE and R^2,
%        overall and per channel.
%     3. Runs goodness-of-fit tests on the residuals: a one-sample
%        Kolmogorov-Smirnov test and (if available) an Anderson-Darling
%        test, both against a normal distribution with the residuals'
%        own mean/std. A model that captured all the real structure
%        should leave residuals that don't reject normality; residuals
%        that do reject it (and still have big batches of structure left
%        when you plot them) mean the model is missing dynamics, not
%        just "slightly off".
%     4. Flags any unstable modes (|eigenvalue| > 1), since those blow up
%        the reconstruction over a long horizon and will dominate the
%        fit error even if the model is otherwise good.
%
%   INPUTS
%   X               : nTime x nChannels matrix the DMD was fit on (same
%                     matrix you passed into runDMDOnMatrix or
%                     runMovingWindowDMD - e.g. Y after
%                     removeStimSegments). Do NOT pre-subtract the mean;
%                     this function does that itself to match what the
%                     DMD routines actually fit.
%   Eigenvalues     : r x 1 complex, from runDMDOnMatrix / runMovingWindowDMD.
%   Eigenvectors    : r x nChannels complex, from the same.
%   ModeAmplitudes  : r x 1 complex, from the same.
%
%   OUTPUT
%   fitReport       : scalar struct with fields:
%       .rmseOverall        - scalar RMSE across all channels/samples
%       .rmsePerChannel      - nChannels x 1 RMSE
%       .r2Overall           - scalar R^2 (1 - SSres/SStot), pooled
%       .r2PerChannel        - nChannels x 1 R^2
%       .nUnstableModes      - count of modes with |eigenvalue| > 1
%       .maxAbsEigenvalue    - largest |eigenvalue| (>1 means the
%                              reconstruction grows without bound over a
%                              long enough horizon)
%       .residualKSTestH     - 1 = rejects normality (residuals look
%                              non-noise-like / model likely missing
%                              structure), 0 = fails to reject
%       .residualKSTestP     - p-value for that test
%       .residualADTestH     - same idea, Anderson-Darling (more
%                              sensitive to tail behavior); NaN if the
%                              Statistics and Machine Learning Toolbox's
%                              adtest isn't available
%       .residualADTestP     - p-value, or NaN
%       .Xreconstructed      - nTime x nChannels reconstructed signal
%                              (mean-subtracted scale), for your own
%                              plotting
%       .residuals           - nTime x nChannels residuals (Xm - Xrecon)
%
%   USAGE (regular DMD on the full cleaned recording)
%       rMax = min(size(Y,2), size(Y,1)-1);
%       [Eig, Vec, Amp, Freq, Growth, PODe] = runDMDOnMatrix(Y, EEG.srate, 35);
%       report = evaluateDMDFit(Y, Eig, Vec, Amp);
%
%   USAGE (moving-window-averaged DMD)
%       [Eig, Vec, Amp, Freq, Growth, PODe] = runMovingWindowDMD(Y, EEG.srate, 2500, 1250, 15);
%       reportMW = evaluateDMDFit(Y, Eig, Vec, Amp);
%
%   Compare report.rmseOverall / report.r2Overall between the two to see
%   whether windowing+averaging actually fits the data better, and check
%   residualKSTestH/residualADTestH on both - a model whose residuals
%   still reject normality is leaving real signal on the table no matter
%   how good its RMSE looks.

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('evaluateDMDFit:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

Eigenvalues = Eigenvalues(:);
ModeAmplitudes = ModeAmplitudes(:);
r = numel(Eigenvalues);

if size(Eigenvectors, 1) ~= r || size(Eigenvectors, 2) ~= nChannels
    error('evaluateDMDFit:badEigenvectors', ...
        'Eigenvectors must be r x nChannels (%d x %d); got %dx%d.', ...
        r, nChannels, size(Eigenvectors, 1), size(Eigenvectors, 2));
end
if numel(ModeAmplitudes) ~= r
    error('evaluateDMDFit:badAmplitudes', 'ModeAmplitudes must have r = %d entries.', r);
end

% ---- reconstruct the predicted signal from the modes ----
Xm = X - mean(X, 1);  % DMD routines fit mean-subtracted data; match that here

nVec = (0:nTime-1)';                      % nTime x 1
% Use bsxfun instead of relying on implicit expansion, for compatibility
% across MATLAB versions (and because Eigenvalues/nVec.' shape mismatches
% otherwise throw "incompatible sizes" on older releases).
timeDynamics = bsxfun(@power, Eigenvalues, nVec.');         % r x nTime
timeDynamics = bsxfun(@times, timeDynamics, ModeAmplitudes); % r x nTime
Xreconstructed = real(timeDynamics.' * Eigenvectors);        % nTime x nChannels

% ---- unstable mode check (affects how trustworthy long-horizon fit is) ----
absEig = abs(Eigenvalues);
nUnstableModes = sum(absEig > 1);
maxAbsEigenvalue = max(absEig);
if nUnstableModes > 0
    warning('evaluateDMDFit:unstableModes', ...
        ['%d of %d modes have |eigenvalue| > 1 (max %.4f). The reconstruction ' ...
         'will grow without bound over a long horizon, which can make RMSE/R^2 ' ...
         'look artificially bad even if the model captures the right frequencies.'], ...
        nUnstableModes, r, maxAbsEigenvalue);
end

% ---- fit metrics ----
residuals = Xm - Xreconstructed;

rmsePerChannel = sqrt(mean(residuals.^2, 1))';            % nChannels x 1
rmseOverall = sqrt(mean(residuals(:).^2));

ssRes = sum(residuals.^2, 1);
ssTot = sum(Xm.^2, 1);
r2PerChannel = (1 - ssRes ./ ssTot)';                      % nChannels x 1
r2Overall = 1 - sum(ssRes) / sum(ssTot);

% ---- goodness-of-fit test on residuals: do they look like noise? ----
resFlat = residuals(:);
resStd = (resFlat - mean(resFlat)) / std(resFlat);

try
    [ksH, ksP] = kstest(resStd);
catch
    ksH = NaN; ksP = NaN;
    warning('evaluateDMDFit:noKSTest', 'kstest unavailable (needs Statistics and Machine Learning Toolbox).');
end

try
    [adH, adP] = adtest(resStd);
catch
    adH = NaN; adP = NaN; % adtest is newer/toolbox-only - degrade gracefully
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
    'Xreconstructed', Xreconstructed, ...
    'residuals', residuals);

fprintf('=== DMD Goodness-of-Fit ===\n');
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