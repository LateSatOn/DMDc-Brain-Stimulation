function plotAllChannelsDMDOneStepPrediction(X, Eigenvalues, Eigenvectors, sampleRange)
%PLOTALLCHANNELSDMDONESTEPPREDICTION Visualize actual vs. one-step DMD prediction, all channels stacked.
%
%   PLOTALLCHANNELSDMDONESTEPPREDICTION(X, Eigenvalues, Eigenvectors)
%   PLOTALLCHANNELSDMDONESTEPPREDICTION(X, Eigenvalues, Eigenvectors, sampleRange)
%
%   Same one-step-ahead reconstruction as plotDMDOneStepPrediction/
%   evaluateDMDOneStepFit (rebuild the state operator A from the modes,
%   predict x(t+1) from the REAL x(t) every step), but draws every
%   channel at once, stacked vertically with a constant offset (same
%   style as plotAllCleanedChannels), so you can compare how well the
%   model fits across channels in one view instead of one at a time.
%   Each channel's own RMSE/R^2 for the shown window is printed on its
%   y-axis label.
%
%   INPUTS
%   X               : nTime x nChannels matrix the DMD was fit on (or any
%                     matrix with the same number of channels).
%   Eigenvalues     : r x 1 complex (from runDMDOnMatrix or runMovingWindowDMD).
%   Eigenvectors    : r x nChannels complex (from either of those).
%   sampleRange     : OPTIONAL vector of sample indices (into X) to plot.
%                     Defaults to the first 2000 samples if omitted.
%
%   USAGE
%       plotAllChannelsDMDOneStepPrediction(XbeforeTest, EigBeforeMW, VecBeforeMW);
%       plotAllChannelsDMDOneStepPrediction(XbeforeTest, EigBeforeMW, VecBeforeMW, 1:3000);

if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('plotAllChannelsDMDOneStepPrediction:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

Eigenvalues = Eigenvalues(:);
r = numel(Eigenvalues);

if size(Eigenvectors, 1) ~= r || size(Eigenvectors, 2) ~= nChannels
    error('plotAllChannelsDMDOneStepPrediction:badEigenvectors', ...
        'Eigenvectors must be r x nChannels (%d x %d); got %dx%d.', ...
        r, nChannels, size(Eigenvectors, 1), size(Eigenvectors, 2));
end

if nTime < 2
    error('plotAllChannelsDMDOneStepPrediction:tooShort', 'X must have at least 2 time samples.');
end

if nargin < 4 || isempty(sampleRange)
    sampleRange = 1:min(2000, nTime - 1);
end
sampleRange = sampleRange(:)';
if any(sampleRange < 1) || any(sampleRange > nTime - 1)
    error('plotAllChannelsDMDOneStepPrediction:badRange', ...
        'sampleRange must stay within 1 to size(X,1)-1 = %d.', nTime - 1);
end

% ---- one-step-ahead reconstruction, all channels at once ----
PhiMat = Eigenvectors.';
A = PhiMat * diag(Eigenvalues) * pinv(PhiMat);

Xm = X - mean(X, 1);
Xcurrent = Xm(sampleRange, :);
Xnext    = Xm(sampleRange + 1, :);
Xpred    = real(Xcurrent * A.');

residual = Xnext - Xpred;

rmsePerChannel = sqrt(mean(residual.^2, 1));
ssRes = sum(residual.^2, 1);
ssTot = sum(Xnext.^2, 1);
r2PerChannel = 1 - ssRes ./ ssTot;

% ---- stacked layout (same style as plotAllCleanedChannels) ----
channelStd = std(Xnext, 0, 1);
offsetStep = 8 * median(channelStd);
if offsetStep == 0 || isnan(offsetStep)
    offsetStep = 1;
end

figure('Color', 'w', 'Position', [100 100 1100 800]);
hold on;
yTickPositions = zeros(nChannels, 1);
yTickLabelsText = cell(nChannels, 1);

for ch = 1:nChannels
    offset = (ch - 1) * offsetStep;
    plot(sampleRange + 1, Xnext(:, ch) + offset, 'b', 'LineWidth', 0.75);
    plot(sampleRange + 1, Xpred(:, ch) + offset, 'r--', 'LineWidth', 0.75);
    yTickPositions(ch) = offset;
    yTickLabelsText{ch} = sprintf('Ch %d (R^2=%.2f, RMSE=%.1f)', ch, r2PerChannel(ch), rmsePerChannel(ch));
end

set(gca, 'YTick', yTickPositions, 'YTickLabel', yTickLabelsText, 'FontSize', 8);
xlabel('Sample');
title('All channels: actual (blue) vs. one-step DMD prediction (red dashed)');
legend({'Actual', 'DMD one-step prediction'}, 'Location', 'eastoutside');
grid on; box off;
xlim([sampleRange(1) + 1, sampleRange(end) + 1]);
ylim([-offsetStep, (nChannels) * offsetStep]);

hold off;

end
