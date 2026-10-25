function sweepTable = evaluateDMDHorizonSweep(X, Eigenvalues, Eigenvectors, horizons, keptSampleIdx)
%EVALUATEDMDHORIZONSWEEP Table of RMSE/R^2 vs. prediction horizon.
%
%   sweepTable = EVALUATEDMDHORIZONSWEEP(X, Eigenvalues, Eigenvectors, horizons)
%   sweepTable = EVALUATEDMDHORIZONSWEEP(X, Eigenvalues, Eigenvectors, horizons, keptSampleIdx)
%
%   Calls evaluateDMDHorizonFit once per entry in `horizons` and collects
%   the results into one table, so you can see exactly how fast the
%   model's predictive power degrades as you ask it to look farther into
%   the future - e.g. horizons = [1 5 10 25 50 100] shows you RMSE/R^2 at
%   each of those jump sizes side by side.
%
%   INPUTS
%   X, Eigenvalues, Eigenvectors, keptSampleIdx : same as evaluateDMDHorizonFit.
%   horizons        : vector of positive integers, e.g. [1 5 10 25 50 100].
%
%   OUTPUT
%   sweepTable      : table with columns horizon, rmseOverall, r2Overall,
%                     residualKSTestH, residualADTestH, nSamplesEvaluated,
%                     nExcludedGapTransitions - one row per horizon.
%
%   USAGE
%       sweepTable = evaluateDMDHorizonSweep(XbeforeTest, EigBeforeMW, VecBeforeMW, ...
%           [1 5 10 25 50 100 250 500], []);
%       disp(sweepTable)

if ~isvector(horizons) || isempty(horizons)
    error('evaluateDMDHorizonSweep:badHorizons', 'horizons must be a nonempty vector of positive integers.');
end
horizons = sort(horizons(:));

if nargin < 5
    keptSampleIdx = [];
end

nH = numel(horizons);
horizon          = horizons;
rmseOverall      = zeros(nH, 1);
r2Overall        = zeros(nH, 1);
residualKSTestH  = zeros(nH, 1);
residualADTestH  = zeros(nH, 1);
nSamplesEvaluated = zeros(nH, 1);
nExcludedGapTransitions = zeros(nH, 1);

for i = 1:nH
    rep = evaluateDMDHorizonFit(X, Eigenvalues, Eigenvectors, horizons(i), keptSampleIdx);
    rmseOverall(i)            = rep.rmseOverall;
    r2Overall(i)               = rep.r2Overall;
    residualKSTestH(i)         = rep.residualKSTestH;
    residualADTestH(i)         = rep.residualADTestH;
    nSamplesEvaluated(i)       = rep.nSamplesEvaluated;
    nExcludedGapTransitions(i) = rep.nExcludedGapTransitions;
end

sweepTable = table(horizon, rmseOverall, r2Overall, residualKSTestH, residualADTestH, ...
    nSamplesEvaluated, nExcludedGapTransitions);

fprintf('evaluateDMDHorizonSweep: evaluated %d horizons (%s).\n', nH, mat2str(horizon'));

end
