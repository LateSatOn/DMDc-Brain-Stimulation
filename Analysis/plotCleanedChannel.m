function plotCleanedChannel(EEG, Xclean, channel, keptSampleIdx)
%PLOTCLEANEDCHANNEL Plot one channel from a stimulation-removed matrix.
%
%   PLOTCLEANEDCHANNEL(EEG, Xclean, channel, keptSampleIdx) plots a
%   single channel's trace from Xclean (the output of
%   removeStimSegments), i.e. AFTER stimulation-locked samples have
%   been cut out.
%
%   INPUTS
%   EEG           : EEGLAB EEG struct the data came from (e.g. from
%                   loadEEGSet). Used only to resolve channel
%                   labels/type filtering - not to re-derive events,
%                   since Xclean's rows no longer line up with
%                   EEG.event(:).latency once samples have been
%                   removed.
%   Xclean        : nTimeClean x nChannels matrix, as returned by
%                   removeStimSegments(EEG, X, bufferSamples), where X
%                   came from eegToDMDMatrix(EEG).
%   channel       : which channel to plot. Either a numeric column index
%                   into Xclean, or a channel label (char/string).
%                   Because eegToDMDMatrix may drop non-EEG channels
%                   (based on EEG.chanlocs(:).type), this function
%                   recomputes that same channel selection internally
%                   so indices/labels line up with Xclean's actual
%                   columns - not necessarily the same indices as raw
%                   EEG.data.
%   keptSampleIdx : the second output of removeStimSegments (the
%                   original sample index kept at each row of Xclean).
%                   IMPORTANT: without this, the plot draws one
%                   continuous line straight across every removed gap,
%                   connecting the last sample before a cut directly to
%                   the first sample after it - this creates a
%                   misleading-looking spike/line at every gap even
%                   though no stimulation data remains there. Passing
%                   keptSampleIdx lets this function insert a break in
%                   the line at each gap instead, so what you see is
%                   the real signal, not a plotting artifact.
%
%   NOTE ON THE X-AXIS: because stimulation segments were cut out of
%   Xclean, its rows are no longer evenly spaced in real time (there are
%   gaps where samples were removed). This plots against ORIGINAL sample
%   index (from keptSampleIdx) when provided, so gap width is visible
%   and to scale; otherwise it falls back to plain 1:nTimeClean.

if ~isfield(EEG, 'data')
    error('plotCleanedChannel:badInput', 'EEG must be a valid EEGLAB EEG struct (missing .data).');
end

% ---- Recompute the same channel selection as eegToDMDMatrix, so ----
% ---- channel indices/labels line up with Xclean's columns.      ----
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

if size(Xclean, 2) ~= numel(chanLabels)
    error('plotCleanedChannel:sizeMismatch', ...
        ['Xclean has %d columns but %d channels were expected from EEG.chanlocs. ' ...
         'Make sure Xclean came from removeStimSegments(EEG, eegToDMDMatrix(EEG), ...) ' ...
         'using this same EEG struct.'], size(Xclean,2), numel(chanLabels));
end

% ---- Resolve channel selector to a column index into Xclean ----
if isnumeric(channel)
    colIdx = channel;
    if ~isscalar(colIdx) || colIdx < 1 || colIdx > numel(chanLabels) || colIdx ~= round(colIdx)
        error('plotCleanedChannel:badChannelIndex', ...
            'channel index must be an integer between 1 and %d (the number of channels in Xclean).', ...
            numel(chanLabels));
    end
    chLabel = chanLabels{colIdx};
else
    channel = char(channel);
    matchIdx = find(strcmpi(chanLabels, channel));
    if isempty(matchIdx)
        error('plotCleanedChannel:channelNotFound', ...
            'Channel label "%s" not found among Xclean''s channels.', channel);
    end
    colIdx = matchIdx(1);
    chLabel = chanLabels{colIdx};
end

% ---- Plot ----
data = Xclean(:, colIdx);

if nargin >= 4 && ~isempty(keptSampleIdx)
    keptSampleIdx = keptSampleIdx(:);
    if numel(keptSampleIdx) ~= numel(data)
        error('plotCleanedChannel:keptIdxMismatch', ...
            'keptSampleIdx has %d entries but Xclean has %d rows - they must match.', ...
            numel(keptSampleIdx), numel(data));
    end

    % Insert a NaN break wherever there's a gap in the original sample
    % index, so the plotted line doesn't join across removed segments.
    gapAfter = find(diff(keptSampleIdx) > 1);
    xPlot = double(keptSampleIdx);
    yPlot = data;
    for k = 1:numel(gapAfter)
        pos = gapAfter(k) + (k - 1);
        xPlot = [xPlot(1:pos); NaN; xPlot(pos+1:end)]; %#ok<AGROW>
        yPlot = [yPlot(1:pos); NaN; yPlot(pos+1:end)]; %#ok<AGROW>
    end
    xLabelStr = 'Original sample index (gaps shown as breaks, not joined)';
else
    xPlot = 1:numel(data);
    yPlot = data;
    xLabelStr = 'Sample index (post-stim-removal, non-uniform time) - gaps NOT shown, pass keptSampleIdx to fix';
end

figure('Color', 'w');
plot(xPlot, yPlot, 'Color', [0.2 0.2 0.2]);
xlabel(xLabelStr);
ylabel('Amplitude (\muV)');
title(sprintf('Channel %s (post-stim-removal)', chLabel));
grid on;

end