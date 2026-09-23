function [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, GrowthRates, POD_Mode_Energies] = runDMDOnMatrix(X, srate, r)
%RUNDMDONMATRIX Run Dynamic Mode Decomposition on a [time x channels] matrix.
%
%   [Eigenvalues, Eigenvectors, ModeAmplitudes, ModeFrequencies, ...
%       GrowthRates, POD_Mode_Energies] = RUNDMDONMATRIX(X, srate, r)
%
%   Self-contained DMD implementation for the EEG pipeline - no external
%   add-on/toolbox dependency. Takes X exactly as produced by
%   eegToDMDMatrix (optionally passed through removeStimSegments and/or
%   buildHankelMatrix first), the recording's sample rate, and the
%   number of modes to keep.
%
%   INPUTS
%   X     : nTime x nChannels matrix (time down the rows).
%   srate : sample rate in Hz (e.g. EEG.srate). Converted internally to
%           dt = 1/srate.
%   r     : number of DMD modes to keep. Must satisfy
%           1 <= r <= min(nChannels, nTime-1) (the rank limit of the
%           underlying SVD). If you don't know a good r yet, run this
%           once with r = min(nChannels, nTime-1) just to get
%           POD_Mode_Energies back, plot stem(POD_Mode_Energies) /
%           stem(cumsum(POD_Mode_Energies)/sum(POD_Mode_Energies)),
%           pick r from the elbow/cumulative energy, then rerun with
%           that r.
%
%   OUTPUTS
%   Eigenvalues       : r x 1 complex DMD eigenvalues. Each mode's
%                       eigenvector is multiplied by exp(Eigenvalue*dt)
%                       per time step (z-transform).
%   Eigenvectors      : r x nChannels complex mode shapes - row k is
%                       the shape of mode k across channels.
%   ModeAmplitudes    : r x 1 complex amplitudes. Rank modes by
%                       abs(ModeAmplitudes) to find the dominant ones.
%   ModeFrequencies   : r x 1 real two-sided frequencies (Hz), one per
%                       mode.
%   GrowthRates       : r x 1 real growth/decay rates (1/s). Negative =
%                       decaying, ~0 = sustained oscillation, positive =
%                       growing (often a sign of something off for a
%                       finite real-world recording, e.g. numerical
%                       rank issues or non-stationarity, rather than a
%                       genuinely growing process).
%   POD_Mode_Energies : full-rank singular-value-squared energy
%                       spectrum (length min(nChannels, nTime-1)), for
%                       picking r as described above.
%
%   METHOD (exact DMD, per Tu et al. / Schmid): mean-subtract each
%   channel over time, split into two time-shifted snapshot sets,
%   economy-SVD the first set, project onto the truncated POD basis,
%   compute the eigendecomposition of the projected linear operator,
%   and reconstruct the full-state DMD modes from that.

if ndims(X) ~= 2
    error('runDMDOnMatrix:badX', 'X must be a 2D [time x channels] matrix.');
end
if ~isscalar(srate) || srate <= 0
    error('runDMDOnMatrix:badSrate', 'srate must be a positive scalar (Hz).');
end

nTime = size(X, 1);
nCh   = size(X, 2);
rMax  = min(nCh, nTime - 1);

if ~isscalar(r) || r < 1 || r ~= round(r)
    error('runDMDOnMatrix:badR', 'r must be a positive integer.');
end
if r > rMax
    error('runDMDOnMatrix:rTooLarge', ...
        'r (%d) exceeds the rank limit for this X: min(nChannels, nTime-1) = %d.', r, rMax);
end

dt = 1 / srate;

% ---- Remove each channel's time-mean (required so modes capture ----
% ---- oscillations around the mean, not a shifted centroid).      ----
Xm = X - mean(X, 1);

% ---- Orient as channels x time and split into two snapshot sets ----
Xt = Xm.';            % nCh x nTime
X1 = Xt(:, 1:end-1);  % nCh x (nTime-1)
X2 = Xt(:, 2:end);    % nCh x (nTime-1)

% ---- Economy SVD of X1. 'econ' (not the legacy 0 flag) is required ----
% ---- here because X1 is typically WIDE (few channels, many time    ----
% ---- samples) - svd(X1,0) only economizes for TALL matrices and    ----
% ---- would otherwise attempt a full SVD and run out of memory.     ----
[U, S, V] = svd(X1, 'econ');

POD_Mode_Energies = diag(S).^2;

% ---- Truncate to rank r ----
U = U(:, 1:r);
V = V(:, 1:r);
S = S(1:r, 1:r);

% ---- Projected linear operator and its eigendecomposition ----
Atilde = U' * X2 * V / S;
[eVecs, eValsDiag] = eig(Atilde);
Eigenvalues = diag(eValsDiag);

% ---- Reconstruct full-state DMD eigenvectors (mode shapes) ----
Eigenvectors = X2 * V / S * eVecs;   % nCh x r

% ---- Mode amplitudes from the first snapshot ----
ModeAmplitudes = Eigenvectors \ X1(:, 1);

% ---- Frequencies and growth rates ----
fNY = 1 / (2 * dt);
ModeFrequencies = (angle(Eigenvalues) / pi) * fNY;
GrowthRates = log(abs(Eigenvalues)) / dt;

% ---- Return mode shapes as r x nChannels (row k = mode k's shape) ----
Eigenvectors = Eigenvectors.';

end