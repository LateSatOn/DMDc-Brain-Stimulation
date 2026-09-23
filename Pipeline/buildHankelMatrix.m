function H = buildHankelMatrix(X, d)
%BUILDHANKELMATRIX Time-delay (Hankel) embedding of a DMD-ready matrix.
%
%   H = BUILDHANKELMATRIX(X, d) builds a time-delay embedding of X,
%   stacking d delayed copies of the channels side by side. This is the
%   same construction as the buildHankel helper in your original DMD
%   script, pulled out as its own function.
%
%   INPUTS
%   X : nTime x nCh matrix (time down the rows), e.g. from
%       eegToDMDMatrix / removeStimSegments.
%   d : number of delay steps to embed. Must be a positive integer,
%       and d < nTime (needs at least one row left after embedding).
%
%   OUTPUT
%   H : (nTime-d) x (nCh*d) matrix. Still a plain 2D [time x channels]
%       matrix - just with nCh*d "channels" now (d delayed copies of
%       the original nCh channels stacked column-wise) - so it plugs
%       straight into runDMDOnMatrix exactly like a non-embedded X does:
%
%           H = buildHankelMatrix(X, d);
%           r = min(size(H,2), size(H,1)-1);   % or your chosen rank
%           [Eigenvalues, Eigenvectors, ModeAmplitudes, ...
%               ModeFrequencies, GrowthRates, POD_Mode_Energies] = ...
%               runDMDOnMatrix(H, EEG.srate, r);
%
%       Keep in mind r is now capped by nCh*d (as well as nTime-d-1),
%       which grows with d - and Eigenvectors will come back shaped
%       [r, nCh*d], i.e. r mode shapes each spanning all d delayed
%       copies of every channel, not just [r, nCh]. If you want the
%       mode shape back in plain per-channel form, take just the first
%       nCh columns of each row (the most recent/undelayed copy):
%       Eigenvectors(:, 1:nCh).

if ndims(X) ~= 2
    error('buildHankelMatrix:badX', 'X must be a 2D [time x channels] matrix.');
end
if ~isscalar(d) || d < 1 || d ~= round(d)
    error('buildHankelMatrix:badD', 'd must be a positive integer.');
end

[nTime, nCh] = size(X);

if d >= nTime
    error('buildHankelMatrix:dTooLarge', ...
        'd (%d) must be less than nTime (%d) - need at least one row left after embedding.', d, nTime);
end

H = zeros(nTime - d, nCh * d);
for i = 1:d
    H(:, (i-1)*nCh+1 : i*nCh) = X(i : end-d+i-1, :);
end

end
