function plotDMDOneStepPrediction(X, Eigenvalues, Eigenvectors, channel, sampleRange)
%PLOTDMDONESTEPPREDICTION Visualize actual vs. one-step-ahead DMD prediction for one channel.
%
%   PLOTDMDONESTEPPREDICTION(X, Eigenvalues, Eigenvectors, channel)
%   PLOTDMDONESTEPPREDICTION(X, Eigenvalues, Eigenvectors, channel, sampleRange)
%
%   Uses the exact same one-step-ahead reconstruction as
%   evaluateDMDOneStepFit (rebuild the state operator A from the modes,
%   then predict x(t+1) from the REAL x(t) at every step - nothing
%   compounds across time) and plots the real signal against what the
%   model predicted, for one channel, so you can see the fit instead of
%   only reading RMSE/R^2 numbers. Draws two stacked subplots: actual vs.
%   predicted on top, and the residual (actual - predicted) underneath.
%
%   INPUTS
%   X               : nTime x nChannels matrix the DMD was fit on (or any
%                     matrix with the same number of channels - e.g. one
%                     segment from windowStimSegments, or Xtest from
%                     splitTrainTestDMD if you want to see held-out
%                     behavior specifically).
%   Eigenvalues     : r x 1 complex (from runDMDOnMatrix or runMovingWindowDMD).
%   Eigenvectors    : r x nChannels complex (from either of those).
%   channel         : scalar integer, which channel (column of X) to plot.
%   sampleRange     : OPTIONAL vector of sample indices (into X) to plot,
%                     e.g. 5000:7000. Defaults to the first 2000 samples
%                     if omitted - plotting the whole recording at once
%                     is unreadable and slow for multi-million-sample
%                     EEG data.
%
%   USAGE
%       % whole-segment model, look at the start of that same segment:
%       plotDMDOneStepPrediction(Xbefore, EigBefore, VecBefore, 1);
%
%       % zoom into a specific stretch:
%       plotDMDOneStepPrediction(Xbefore, EigBefore, VecBefore, 1, 10000:12000);
%
%       % check held-out behavior specifically (model fit on train, shown on test):
%       [Xtrain, ~, Xtest, ~] = splitTrainTestDMD(Y, keptIdx, 0.7);
%       [EigTr, VecTr, ~, ~, ~, ~] = runMovingWindowDMD(Xtrain, EEG.srate, 2500, 1250, 15);
%       plotDMDOneStepPrediction(Xtest, EigTr, VecTr, 1, 1:3000);

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('plotDMDOneStepPrediction:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

Eigenvalues = Eigenvalues(:);
r = numel(Eigenvalues);

if size(Eigenvectors, 1) ~= r || size(Eigenvectors, 2) ~= nChannels
    error('plotDMDOneStepPrediction:badEigenvectors', ...
        'Eigenvectors must be r x nChannels (%d x %d); got %dx%d.', ...
        r, nChannels, size(Eigenvectors, 1), size(Eigenvectors, 2));
end

if ~isscalar(channel) || channel < 1 || channel > nChannels || channel ~= round(channel)
    error('plotDMDOneStepPrediction:badChannel', 'channel must be an integer between 1 and %d.', nChannels);
end

if nTime < 2
    error('plotDMDOneStepPrediction:tooShort', 'X must have at least 2 time samples.');
end

if nargin < 5 || isempty(sampleRange)
    sampleRange = 1:min(2000, nTime - 1);
end
sampleRange = sampleRange(:)';
if any(sampleRange < 1) || any(sampleRange > nTime - 1)
    error('plotDMDOneStepPrediction:badRange', ...
        'sampleRange must stay within 1 to size(X,1)-1 = %d (prediction needs a next sample to compare against).', nTime - 1);
end

% ---- one-step-ahead reconstruction (same math as evaluateDMDOneStepFit) ----
PhiMat = Eigenvectors.';
A = PhiMat * diag(Eigenvalues) * pinv(PhiMat);

Xm = X - mean(X, 1);
Xcurrent = Xm(sampleRange, :);
Xnext    = Xm(sampleRange + 1, :);
Xpred    = real(Xcurrent * A.');

actual    = Xnext(:, channel);
predicted = Xpred(:, channel);
residual  = actual - predicted;

rmseHere = sqrt(mean(residual.^2));
ssRes = sum(residual.^2);
ssTot = sum(actual.^2);
r2Here = 1 - ssRes / ssTot;

figure('Color', 'w');

subplot(2, 1, 1);
plot(sampleRange + 1, actual, 'b', 'LineWidth', 1.1); hold on;
plot(sampleRange + 1, predicted, 'r--', 'LineWidth', 1.1);
xlabel('Sample');
ylabel('Amplitude (mean-subtracted)');
title(sprintf('Channel %d: actual vs. one-step DMD prediction (RMSE=%.4f, R^2=%.4f)', ...
    channel, rmseHere, r2Here));
legend('Actual', 'DMD one-step prediction', 'Location', 'best');
grid on; box off;

subplot(2, 1, 2);
plot(sampleRange + 1, residual, 'Color', [0.5 0.1 0.1]);
yline(0, 'k--');
xlabel('Sample');
ylabel('Residual (actual - predicted)');
title('Residual');
grid on; box off;

end
