function [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, GrowthRates, POD_Mode_Energies] = runMovingWindowDMD(X, srate, windowSamples, overlapSamples, r)
%RUNMOVINGWINDOWDMD Sliding-window DMD with mode-matched moving average.
%
%   [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, ...
%    GrowthRates, POD_Mode_Energies] = RUNMOVINGWINDOWDMD(X, srate, ...
%    windowSamples, overlapSamples, r)
%
%   Runs runDMDOnMatrix independently on a sliding window of X (instead
%   of once on the whole matrix), then averages the per-window DMD
%   results into one set of modes. The idea: a single DMD fit on a long,
%   nonstationary EEG recording is really fitting one linear operator to
%   data whose dynamics drift over time, which biases the estimate.
%   Splitting into short, mostly-stationary windows, fitting DMD on each,
%   and averaging across windows gives a smoother, more robust estimate
%   of the recurring modes while still using the whole recording.
%
%   WINDOWING
%   windowSamples and overlapSamples work exactly like a standard
%   windowed-analysis setup (e.g. spectrogram/pwelch):
%       stepSamples = windowSamples - overlapSamples
%       window k covers samples [(k-1)*stepSamples + 1, ... , + windowSamples - 1]
%   overlapSamples = 0 means back-to-back, non-overlapping windows.
%   Larger overlap -> more windows, more smoothing, more compute.
%
%   MODE MATCHING (why this isn't just "average the eigenvalues")
%   eig() returns eigenvalues/eigenvectors in an arbitrary order that can
%   differ from window to window, so you can't just average
%   window-by-window element-wise - mode 3 in window 1 might correspond
%   to mode 7 in window 2. This function uses window 1 as the reference
%   mode ordering, then for every other window greedily matches each
%   reference mode to whichever of that window's modes has the closest
%   eigenvalue (complex-plane distance), and only THEN averages across
%   windows. This is a greedy nearest-neighbor match, not a globally
%   optimal assignment (e.g. Hungarian algorithm) - fine for well-
%   separated modes, but closely-spaced eigenvalues can occasionally be
%   mismatched. Inspect Eigenvalues' spread across windows if unsure.
%
%   PHASE/SCALE ALIGNMENT (why eigenvector averaging needed a fix)
%   A DMD eigenvector is only defined up to an arbitrary complex scalar -
%   eig() doesn't guarantee the same phase (or even exactly the same
%   scale) convention from window to window, even for "the same"
%   physical mode. Averaging matched eigenvectors directly, without
%   accounting for this, causes partial destructive interference across
%   many windows and distorts the combined mode shapes - this showed up
%   in practice as a systematic (non-random) bias in one-step
%   reconstruction, not just ordinary fit error. Before averaging, every
%   matched eigenvector is now rescaled by a least-squares-optimal
%   complex scalar c = <ref, cur> / <cur, cur> to best align it with
%   window 1's version of that mode, and the corresponding mode
%   amplitude is rescaled by 1/c so the physical contribution
%   (amplitude * eigenvector) is left unchanged by this relabeling.
%
%   INPUTS
%   X                : nTime x nChannels DMD-ready matrix, e.g. the
%                      output of eegToDMDMatrix (optionally after
%                      removeStimSegments/windowStimSegments and/or
%                      buildHankelMatrix).
%   srate            : sampling rate in Hz (e.g. EEG.srate). Used to
%                      convert eigenvalues to physical frequencies/growth
%                      rates the same way runDMDOnMatrix does.
%   windowSamples    : positive integer, length of each DMD window in
%                      samples. Must be <= size(X,1).
%   overlapSamples   : nonnegative integer, number of samples consecutive
%                      windows share. Must be < windowSamples.
%   r                : positive integer, number of DMD modes to keep per
%                      window (passed straight through to
%                      runDMDOnMatrix). Must be <= min(nChannels,
%                      windowSamples-1).
%
%   OUTPUTS (same convention/order as runDMDOnMatrix, but averaged
%   across all windows after mode-matching)
%   Eigenvalues        : r x 1 complex, averaged discrete-time DMD
%                        eigenvalues.
%   Eigenvectors       : r x nChannels complex, averaged DMD modes.
%   ModeAmplitudes     : r x 1 complex, averaged initial mode amplitudes.
%   ModeFrequencies    : r x 1 real, averaged mode frequencies (Hz).
%   GrowthRates        : r x 1 real, averaged mode growth/decay rates
%                        (1/s).
%   POD_Mode_Energies  : nChannels x 1 real, averaged POD energy
%                        spectrum (no mode-matching needed here - SVD
%                        singular values are already ordered largest
%                        first in every window).
%
%   USAGE
%       [eVals, eVecs, amps, freqs, growth, podE] = ...
%           runMovingWindowDMD(X, EEG.srate, 1000, 500, 10);
%       figure; plot(real(eVals), imag(eVals), 'o'); hold on;
%       theta = linspace(0, 2*pi, 200); plot(cos(theta), sin(theta), 'k--');
%       axis equal; title('Averaged DMD eigenvalues (unit circle)');

% ---- validate inputs ----
if ~ismatrix(X) || ~isnumeric(X) || isempty(X)
    error('runMovingWindowDMD:badX', 'X must be a nonempty numeric nTime x nChannels matrix.');
end
[nTime, nChannels] = size(X);

if ~isscalar(srate) || ~isnumeric(srate) || srate <= 0
    error('runMovingWindowDMD:badSrate', 'srate must be a positive scalar.');
end

if ~isscalar(windowSamples) || windowSamples ~= round(windowSamples) || windowSamples <= 1
    error('runMovingWindowDMD:badWindow', 'windowSamples must be an integer > 1.');
end
if windowSamples > nTime
    error('runMovingWindowDMD:windowTooLong', ...
        'windowSamples (%d) cannot exceed size(X,1) (%d).', windowSamples, nTime);
end

if ~isscalar(overlapSamples) || overlapSamples ~= round(overlapSamples) || overlapSamples < 0
    error('runMovingWindowDMD:badOverlap', 'overlapSamples must be a nonnegative integer.');
end
if overlapSamples >= windowSamples
    error('runMovingWindowDMD:overlapTooLarge', ...
        'overlapSamples (%d) must be smaller than windowSamples (%d).', overlapSamples, windowSamples);
end

if ~isscalar(r) || r ~= round(r) || r <= 0
    error('runMovingWindowDMD:badR', 'r must be a positive integer.');
end
if r > min(nChannels, windowSamples - 1)
    error('runMovingWindowDMD:rTooLarge', ...
        'r (%d) must be <= min(nChannels, windowSamples-1) = %d.', r, min(nChannels, windowSamples - 1));
end

% ---- build window start indices ----
stepSamples = windowSamples - overlapSamples;
nWindows = floor((nTime - windowSamples) / stepSamples) + 1;
if nWindows < 1
    error('runMovingWindowDMD:noWindows', ...
        'No complete window fits in X with these settings.');
end

% ---- run DMD independently on every window ----
allEig    = cell(nWindows, 1);
allVec    = cell(nWindows, 1);
allAmp    = cell(nWindows, 1);
allFreq   = cell(nWindows, 1);
allGrowth = cell(nWindows, 1);
allPodE   = cell(nWindows, 1);

for k = 1:nWindows
    startIdx = (k - 1) * stepSamples + 1;
    endIdx   = startIdx + windowSamples - 1;
    Xw = X(startIdx:endIdx, :);

    [eVals, eVecs, amps, freqs, growth, podE] = runDMDOnMatrix(Xw, srate, r);

    allEig{k}    = eVals;    % r x 1
    allVec{k}    = eVecs;    % r x nChannels
    allAmp{k}    = amps;     % r x 1
    allFreq{k}   = freqs;    % r x 1
    allGrowth{k} = growth;   % r x 1
    allPodE{k}   = podE;     % nCh x 1 (length can vary only if nChannels changes, which it doesn't here)
end

% ---- mode-match every window to window 1's ordering ----
refEig = allEig{1};

matchedEig    = zeros(r, nWindows);
matchedVec    = zeros(r, nChannels, nWindows);
matchedAmp    = zeros(r, nWindows);
matchedFreq   = zeros(r, nWindows);
matchedGrowth = zeros(r, nWindows);

matchedEig(:, 1)      = allEig{1};
matchedVec(:, :, 1)   = allVec{1};
matchedAmp(:, 1)      = allAmp{1};
matchedFreq(:, 1)     = allFreq{1};
matchedGrowth(:, 1)   = allGrowth{1};

refVecForAlign = allVec{1}; % window 1's eigenvectors, r x nChannels - the phase/scale reference

for k = 2:nWindows
    assignment = matchModesGreedy(refEig, allEig{k});
    curVec = allVec{k}(assignment, :); % r x nChannels
    curAmp = allAmp{k}(assignment);    % r x 1

    % DMD eigenvectors are only defined up to an arbitrary complex scalar
    % (eig() doesn't guarantee the same phase/scale convention window to
    % window), so before averaging, align each matched mode to window 1's
    % version via a best-fit complex scalar c (least-squares: c = <ref,
    % cur> / <cur, cur>). Without this, averaging unaligned eigenvectors
    % across many windows causes partial destructive interference and
    % distorts the combined mode shapes - which showed up as a systematic
    % bias in one-step reconstruction, not just noise. The amplitude gets
    % the inverse scaling (1/c) so the physical mode contribution
    % (amplitude * eigenvector) is unchanged by this relabeling.
    for i = 1:r
        refVec_i = refVecForAlign(i, :);
        curVec_i = curVec(i, :);
        denom = curVec_i * curVec_i';
        if denom > eps
            c = (refVec_i * curVec_i') / denom;
            curVec(i, :) = c * curVec_i;
            curAmp(i) = curAmp(i) / c;
        end
    end

    matchedEig(:, k)    = allEig{k}(assignment);
    matchedVec(:, :, k) = curVec;
    matchedAmp(:, k)    = curAmp;
    matchedFreq(:, k)   = allFreq{k}(assignment);
    matchedGrowth(:, k) = allGrowth{k}(assignment);
end

% ---- average across windows ----
Eigenvalues     = mean(matchedEig, 2);
Eigenvectors    = mean(matchedVec, 3);
ModeAmplitudes  = mean(matchedAmp, 2);
ModeFrequencies = mean(matchedFreq, 2);
GrowthRates     = mean(matchedGrowth, 2);

podLengths = cellfun(@numel, allPodE);
minPodLen = min(podLengths);
if any(podLengths ~= minPodLen)
    warning('runMovingWindowDMD:podLengthMismatch', ...
        'POD_Mode_Energies length differed across windows; truncating all to %d entries.', minPodLen);
end
podMat = zeros(minPodLen, nWindows);
for k = 1:nWindows
    podMat(:, k) = allPodE{k}(1:minPodLen);
end
POD_Mode_Energies = mean(podMat, 2);

fprintf(['runMovingWindowDMD: %d window(s) of %d samples (step=%d, overlap=%d), ' ...
    'r=%d modes averaged per window.\n'], nWindows, windowSamples, stepSamples, overlapSamples, r);

end

function assignment = matchModesGreedy(refEig, curEig)
%MATCHMODESGREEDY Greedy nearest-neighbor match of curEig onto refEig's order.
%   assignment(i) = index into curEig that best matches refEig(i), with
%   each curEig index used at most once. Processes reference modes in
%   order and always takes the globally closest remaining pair first, so
%   the easiest/most unambiguous matches are locked in before harder
%   ones - not a true optimal assignment (Hungarian algorithm), but good
%   enough when modes are reasonably well separated in the complex plane.
r = numel(refEig);
assignment = zeros(r, 1);
usedCols = false(1, r);

distMat = abs(refEig(:) - curEig(:).'); % r x r

for n = 1:r
    distMat(:, usedCols) = Inf;
    rowsAssigned = assignment ~= 0;
    distMat(rowsAssigned, :) = Inf;

    [~, linIdx] = min(distMat(:));
    [i, j] = ind2sub(size(distMat), linIdx);

    assignment(i) = j;
    usedCols(j) = true;
end
end