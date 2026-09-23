function plotChannelWithStim(EEG, channel)
%PLOTCHANNELWITHSTIM Plot one EEG channel in full, with stimulation
%event markers overlaid as vertical lines.
%
%   PLOTCHANNELWITHSTIM(EEG, channel) plots EEG.data for the selected
%   channel across the full recording, and draws a vertical line at
%   every event latency in EEG.event, labeled with that event's type.
%   Useful for sanity-checking removeStimSegments: e.g. if events look
%   like paired onset/offset markers with a long gap between them
%   (rather than isolated blips), the stimulation period is the whole
%   SPAN between paired events, not just a short buffer after each one
%   individually - which would mean removeStimSegments needs to remove
%   entire onset-to-offset ranges instead of a fixed post-event buffer.
%
%   INPUTS
%   EEG     : EEGLAB EEG struct (e.g. from loadEEGSet).
%   channel : which channel to plot. Either a numeric index into
%             EEG.data's rows, or a channel label (char/string) matched
%             against EEG.chanlocs(:).labels (case-insensitive).
%
%   Boundary events are drawn in a different color from other events so
%   they're easy to tell apart from actual stimulation markers.

if ~isfield(EEG, 'data') || ~isfield(EEG, 'srate')
    error('plotChannelWithStim:badInput', ...
        'EEG must be a valid EEGLAB EEG struct (missing .data or .srate).');
end

nChannels = size(EEG.data, 1);

% ---- Resolve channel selector to a row index ----
if isnumeric(channel)
    chIdx = channel;
    if ~isscalar(chIdx) || chIdx < 1 || chIdx > nChannels || chIdx ~= round(chIdx)
        error('plotChannelWithStim:badChannelIndex', ...
            'channel index must be an integer between 1 and %d.', nChannels);
    end
    chLabel = sprintf('Ch%d', chIdx);
    if isfield(EEG, 'chanlocs') && numel(EEG.chanlocs) >= chIdx && isfield(EEG.chanlocs, 'labels')
        chLabel = EEG.chanlocs(chIdx).labels;
    end
else
    channel = char(channel);
    if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs) || ~isfield(EEG.chanlocs, 'labels')
        error('plotChannelWithStim:noLabels', ...
            'EEG.chanlocs labels are not available - select the channel by numeric index instead.');
    end
    allLabels = {EEG.chanlocs.labels};
    matchIdx = find(strcmpi(allLabels, channel));
    if isempty(matchIdx)
        error('plotChannelWithStim:channelNotFound', ...
            'Channel label "%s" not found in EEG.chanlocs.', channel);
    end
    chIdx = matchIdx(1);
    chLabel = allLabels{chIdx};
end

% ---- Pull the channel trace and build a time axis in seconds ----
data = double(EEG.data(chIdx, :));
nTime = numel(data);
t = (0:nTime-1) / EEG.srate;

figure('Color', 'w');
plot(t, data, 'Color', [0.2 0.2 0.2]);
hold on;

% ---- Overlay event markers ----
if isfield(EEG, 'event') && ~isempty(EEG.event)
    latencies = round([EEG.event.latency]);
    rawType   = {EEG.event.type};
    isBoundary = cellfun(@(x) ischar(x) && strcmpi(x, 'boundary'), rawType);

    for k = 1:numel(latencies)
        idx = latencies(k);
        if idx < 1 || idx > nTime
            continue; % event falls outside this channel's samples
        end
        tk = t(idx);
        if isBoundary(k)
            lineColor = [0.9 0.6 0.1]; % orange = boundary
        else
            lineColor = [0.85 0.1 0.1]; % red = stimulation/other event
        end
        labelText = rawType{k};
        if ~ischar(labelText)
            labelText = num2str(labelText);
        end
        xline(tk, '-', labelText, 'Color', lineColor, ...
            'LabelOrientation', 'aligned', 'LabelVerticalAlignment', 'top');
    end
end

xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title(sprintf('Channel %s (index %d) with event markers', chLabel, chIdx));
grid on;
hold off;

end