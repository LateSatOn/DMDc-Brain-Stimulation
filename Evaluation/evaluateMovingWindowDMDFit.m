function fitReport = evaluateMovingWindowDMDFit(X, windowSamples, overlapSamples, Eigenvalues, Eigenvectors, ModeAmplitudes)
%EVALUATEMOVINGWINDOWDMDFIT Goodness-of-fit for an averaged moving-window DMD model.
%
%   fitReport = EVALUATEMOVINGWINDOWDMDFIT(X, windowSamples, overlapSamples, ...
%       Eigenvalues, Eigenvectors, ModeAmplitudes)
%
%   WHY THIS IS A SEPARATE FUNCTION FROM evaluateDMDFit
%   evaluateDMDFit reconstructs a signal using ONE global time index
%   running from 0 to nTime-1. That's correct for runDMDOnMatrix (one
%   fit, one continuous trajectory), but it's the WRONG test for
%   runMovingWindowDMD's averaged modes: those modes describe local,
%   short-horizon dynamics inside each window, not one trajectory meant
%   to be extrapolated across millions of samples. If you feed
%   runMovingWindowDMD's output into evaluateDMDFit directly, eigenvalues
%   with |lambda| slightly below 1 (e.g. 0.99) get raised to a
%   multi-million-sample exponent and decay to ~0, making the
%   reconstruction collapse to zero everywhere except the very start.
%   That produces an R^2 of exactly 0 (residual == the data itself) that
%   looks like "no fit at all" even when the averaged model is actually a
%   good LOCAL description - it's an artifact of asking the wrong
%   question, not evidence the windowed model is bad.
%
%   This function asks the right question instead: replay the averaged
%   model inside each window using a time index that resets to 0 at the
%   start of EVERY window (matching how runMovingWindowDMD itself fit
%   each window via runDMDOnMatrix), compare to that window's own
%   (locally mean-subtracted) data, and pool the fit/residuals across all
%   windows. This mirrors the exact windowing runMovingWindowDMD used, so
%   pass it the same windowSamples/overlapSamples you used there.
%
%   INPUTS
%   X               : nTime x nChannels matrix (the SAME X you passed
%                     into runMovingWindowDMD).
%   windowSamples   : same value you passed into runMovingWindowDMD.
%   overlapSamples  : same value you passed into runMovingWindowDMD.
%   Eigenvalues     : r x 1, from runMovingWindowDMD (the averaged modes).
%   Eigenvectors    : r x nChannels, from runMovingWindowDMD.
%   ModeAmplitudes  : r x 1, from runMovingWindowDMD.
%
%   OUTPUT
%   fitReport       : same fields as evaluateDMDFit's output
%                     (rmseOverall, r2Overall, rmsePerChannel,
%                     r2PerChannel, nUnstableModes, maxAbsEigenvalue,
%                     residualKSTestH/P, residualADTestH/P), computed by
%                     pooling residuals across all windows, PLUS:
%       .nWindows        - number of windows actually evaluated
%       .r2PerWindow     - nWindows x 1, R^2 computed separately on each
%                           window (use this to see whether the averaged
%                           model fits some windows much better than
%                           others - e.g. nonstationary drift across the
%                           recording).
%
%   USAGE
%       [EigMW, VecMW, AmpMW, ~, ~, ~] = runMovingWindowDMD(Y, EEG.srate, 2500, 1250, 15);
%       reportMW = evaluateMovingWindowDMDFit(Y, 2500, 1250, EigMW, VecMW, AmpMW);
%       % compare apples-to-apples against the full-recording fit:
%       rMax = min(size(Y,2), size(Y,1)-1);
%       [Eig, Vec, Amp, ~, ~, ~] = runDMDOnMatrix(Y, EEG.srate, 35);
%       reportFull = evaluateDMDFit(Y, Eig, Vec, Amp);
%       fprintf('Full DMD   : RMSE=%.4f  R2=%.4f\n', reportFull.rmseOverall, reportFull.r2Overall);
%       fprintf('Moving DMD : RMSE=%.4f  R2=%.4f\n', reportMW.rmseOverall, reportMW.r2Overall);

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('evaluateMovingWindowDMDFit:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

Eigenvalues = Eigenvalues(:);
ModeAmplitudes = ModeAmplitudes(:);
r = numel(Eigenvalues);

if size(Eigenvectors, 1) ~= r || size(Eigenvectors, 2) ~= nChannels
    error('evaluateMovingWindowDMDFit:badEigenvectors', ...
        'Eigenvectors must be r x nChannels (%d x %d); got %dx%d.', ...
        r, nChannels, size(Eigenvectors, 1), size(Eigenvectors, 2));
end
if numel(ModeAmplitudes) ~= r
    error('evaluateMovingWindowDMDFit:badAmplitudes', 'ModeAmplitudes must have r = %d entries.', r);
end

if ~isscalar(windowSamples) || windowSamples ~= round(windowSamples) || windowSamples <= 1
    error('evaluateMovingWindowDMDFit:badWindow', 'windowSamples must be an integer > 1.');
end
if windowSamples > nTime
    error('evaluateMovingWindowDMDFit:windowTooLong', ...
        'windowSamples (%d) cannot exceed size(X,1) (%d).', windowSamples, nTime);
end
if ~isscalar(overlapSamples) || overlapSamples ~= round(overlapSamples) || overlapSamples < 0
    error('evaluateMovingWindowDMDFit:badOverlap', 'overlapSamples must be a nonnegative integer.');
end
if overlapSamples >= windowSamples
    error('evaluateMovingWindowDMDFit:overlapTooLarge', ...
        'overlapSamples (%d) must be smaller than windowSamples (%d).', overlapSamples, windowSamples);
end

stepSamples = windowSamples - overlapSamples;
nWindows = floor((nTime - windowSamples) / stepSamples) + 1;
if nWindows < 1
    error('evaluateMovingWindowDMDFit:noWindows', 'No complete window fits in X with these settings.');
end

% ---- unstable mode check (same meaning as evaluateDMDFit) ----
absEig = abs(Eigenvalues);
nUnstableModes = sum(absEig > 1);
maxAbsEigenvalue = max(absEig);
if nUnstableModes > 0
    warning('evaluateMovingWindowDMDFit:unstableModes', ...
        '%d of %d averaged modes have |eigenvalue| > 1 (max %.4f).', nUnstableModes, r, maxAbsEigenvalue);
end

% ---- local time dynamics, reused for every window (time resets to 0 each time) ----
nVecLocal = (0:windowSamples-1)';                              % windowSamples x 1
timeDynamicsLocal = bsxfun(@power, Eigenvalues, nVecLocal.');   % r x windowSamples
timeDynamicsLocal = bsxfun(@times, timeDynamicsLocal, ModeAmplitudes); % r x windowSamples
XpredLocal = real(timeDynamicsLocal.' * Eigenvectors);          % windowSamples x nChannels, same for every window

% ---- replay across every window, pool residuals ----
allResiduals = zeros(windowSamples * nWindows, nChannels);
ssResPerChannel = zeros(1, nChannels);
ssTotPerChannel = zeros(1, nChannels);
r2PerWindow = zeros(nWindows, 1);

rowCursor = 0;
for k = 1:nWindows
    startIdx = (k - 1) * stepSamples + 1;
    endIdx   = startIdx + windowSamples - 1;
    Xw = X(startIdx:endIdx, :);
    Xm = Xw - mean(Xw, 1);

    residualsW = Xm - XpredLocal;

    allResiduals(rowCursor+1 : rowCursor+windowSamples, :) = residualsW;
    rowCursor = rowCursor + windowSamples;

    ssResPerChannel = ssResPerChannel + sum(residualsW.^2, 1);
    ssTotPerChannel = ssTotPerChannel + sum(Xm.^2, 1);

    ssResW = sum(residualsW(:).^2);
    ssTotW = sum(Xm(:).^2);
    r2PerWindow(k) = 1 - ssResW / ssTotW;
end

rmsePerChannel = sqrt(ssResPerChannel / (windowSamples * nWindows))'; % nChannels x 1
rmseOverall = sqrt(sum(ssResPerChannel) / (windowSamples * nWindows * nChannels));

r2PerChannel = (1 - ssResPerChannel ./ ssTotPerChannel)'; % nChannels x 1
r2Overall = 1 - sum(ssResPerChannel) / sum(ssTotPerChannel);

% ---- goodness-of-fit test on pooled residuals ----
resFlat = allResiduals(:);
resStd = (resFlat - mean(resFlat)) / std(resFlat);

try
    [ksH, ksP] = kstest(resStd);
catch
    ksH = NaN; ksP = NaN;
    warning('evaluateMovingWindowDMDFit:noKSTest', 'kstest unavailable (needs Statistics and Machine Learning Toolbox).');
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
    'nWindows', nWindows, ...
    'r2PerWindow', r2PerWindow);

fprintf('=== Moving-Window DMD Goodness-of-Fit (local replay, %d windows) ===\n', nWindows);
fprintf('  RMSE (overall)        : %.4f\n', rmseOverall);
fprintf('  R^2  (overall)        : %.4f\n', r2Overall);
fprintf('  R^2  (median/min/max across windows) : %.4f / %.4f / %.4f\n', ...
    median(r2PerWindow), min(r2PerWindow), max(r2PerWindow));
fprintf('  Unstable modes        : %d / %d (max |lambda| = %.4f)\n', nUnstableModes, r, maxAbsEigenvalue);
if ~isnan(ksH)
    fprintf('  Residual K-S test     : H=%d (1=non-normal/structured) p=%.4g\n', ksH, ksP);
end
if ~isnan(adH)
    fprintf('  Residual A-D test     : H=%d (1=non-normal/structured) p=%.4g\n', adH, adP);
end
fprintf('\n');

end
