function plotAllCleanedChannels(EEG, Xclean, keptSampleIdx)
%PLOTALLCLEANEDCHANNELS Plot every channel from a stimulation-removed
%matrix on one stacked graph.
%
%   PLOTALLCLEANEDCHANNELS(EEG, Xclean, keptSampleIdx) loops through
%   every column of Xclean (the output of removeStimSegments) and plots
%   them all on the same axes, each vertically offset so traces don't
%   overlap - the standard "stacked" multi-channel EEG view.
%
%   INPUTS
%   EEG           : EEGLAB EEG struct the data came from (e.g. from
%                   loadEEGSet). Used to resolve channel labels/type
%                   filtering, same as plotCleanedChannel.
%   Xclean        : nTimeClean x nChannels matrix, as returned by
%                   removeStimSegments(EEG, X, bufferSamples).
%   keptSampleIdx : (optional) second output of removeStimSegments -
%                   the original sample index kept at each row of
%                   Xclean. Same reason as in plotCleanedChannel: without
%                   it, the line is drawn straight across every removed
%                   gap, which looks like a spike that isn't really
%                   there. Pass it whenever you have it.

if ~isfield(EEG, 'data')
    error('plotAllCleanedChannels:badInput', 'EEG must be a valid EEGLAB EEG struct (missing .data).');
end

% ---- Recompute the same channel selection as eegToDMDMatrix, so ----
% ---- labels line up with Xclean's columns.                      ----
nChannelsAll = size(EEG.data, 1);
keepIdx = 1:nChannelsAll;
chanLabelsAll = {};

if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
    if isfield(EEG.chanlocs, 'labels')
        chanLabelsAll = {EEG.chanlocs.labels};
    end
    if isfield(EEG.chanlocs, 'type') && any(~cellfun(@isempty, {EEG.chanlocs.type}))
        types = {EEG.chanlocs.type};
        isEEGType = cellfun(@(t) ischar(t) && strcmpi(t, 'EEG'), types);
        if any(isEEGType)
            keepIdx = find(isEEGType);
        end
    end
end

if isempty(chanLabelsAll)
    chanLabelsAll = arrayfun(@(k) sprintf('Ch%d', k), 1:nChannelsAll, 'UniformOutput', false);
end

chanLabels = chanLabelsAll(keepIdx);
nCh = size(Xclean, 2);

if nCh ~= numel(chanLabels)
    error('plotAllCleanedChannels:sizeMismatch', ...
        ['Xclean has %d columns but %d channels were expected from EEG.chanlocs. ' ...
         'Make sure Xclean came from removeStimSegments(EEG, eegToDMDMatrix(EEG), ...) ' ...
         'using this same EEG struct.'], nCh, numel(chanLabels));
end

% ---- Build the (possibly gap-broken) x-axis once, shared by all channels ----
nTime = size(Xclean, 1);
useKeptIdx = nargin >= 3 && ~isempty(keptSampleIdx);

if useKeptIdx
    keptSampleIdx = double(keptSampleIdx(:));
    if numel(keptSampleIdx) ~= nTime
        error('plotAllCleanedChannels:keptIdxMismatch', ...
            'keptSampleIdx has %d entries but Xclean has %d rows - they must match.', ...
            numel(keptSampleIdx), nTime);
    end
    gapAfter = find(diff(keptSampleIdx) > 1);
    xLabelStr = 'Original sample index (gaps shown as breaks, not joined)';
else
    gapAfter = [];
    xLabelStr = 'Sample index (post-stim-removal, non-uniform time) - gaps NOT shown, pass keptSampleIdx to fix';
end

% ---- Vertical offset so channels don't overlap ----
chanStd = std(Xclean, 0, 1);
offsetStep = 8 * median(chanStd(chanStd > 0));
if ~isfinite(offsetStep) || offsetStep == 0
    offsetStep = 1;
end

figure('Color', 'w');
hold on;
yTickPos = zeros(1, nCh);
for k = 1:nCh
    yOffset = (nCh - k) * offsetStep;
    yTickPos(k) = yOffset;

    yData = Xclean(:, k) + yOffset;
    if useKeptIdx
        xData = keptSampleIdx;
        for g = numel(gapAfter):-1:1
            pos = gapAfter(g);
            xData = [xData(1:pos); NaN; xData(pos+1:end)]; %#ok<AGROW>
            yData = [yData(1:pos); NaN; yData(pos+1:end)]; %#ok<AGROW>
        end
    else
        xData = 1:nTime;
    end

    plot(xData, yData, 'Color', [0.15 0.15 0.15], 'LineWidth', 0.5);
end
hold off;

set(gca, 'YTick', fliplr(yTickPos), 'YTickLabel', fliplr(chanLabels));
xlabel(xLabelStr);
ylabel('Channel');
title('All channels (post-stim-removal, stacked)');
grid on;
axis tight;

end
