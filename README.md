# DMDc-Brain-Stimulation

## Pipeline overview

1. **`loadEEGSet`** — load a subject/session's `.set`/`.fdt` pair into an `EEG` struct.
2. (optional) **`EEGLAB`** for filtering
3. **`eegToDMDMatrix`** — convert the `EEG` struct into a `[time x channels]` matrix ready for DMD.
4. **`removeStimSegments`** — cut stimulation periods out of the DMD matrix (concatenates what's left into one continuous matrix).
5. **`windowStimSegments`** — alternative to `removeStimSegments`: splits the recording into separate before/after-stimulation segments instead of concatenating, so each segment can be modeled on its own.
6. (optional) **`buildHankelMatrix`** — time-delay (Hankel) embedding, for use before `runDMDOnMatrix` or `runMovingWindowDMD`.
7. **`runDMDOnMatrix`** — run Dynamic Mode Decomposition once on a DMD-ready matrix.
8. **`runMovingWindowDMD`** — run DMD on a sliding window of the data and average the matched, phase-aligned modes across windows, for a more robust natural-dynamics estimate than a single global fit on a long, nonstationary recording.
9. **`combineSegmentDMDs`** — run `runMovingWindowDMD` on every segment from `windowStimSegments` and combine the per-segment models (same mode-matching/phase-alignment approach) into one overall natural-dynamics model.
10. **`splitTrainTestDMD`** — time-based train/test split of a DMD-ready matrix, used to evaluate a model on held-out data rather than the data it was fit on.
11. **`findSurroundingSegments`** — given `windowStimSegments` output and a target sample, finds the segment ending right before it and the segment starting right after it (used to compare DMD models before vs. after a specific stimulation event).

## Evaluation tools

Goodness-of-fit for a DMD model: RMSE, R², and Kolmogorov-Smirnov/Anderson-Darling tests on the residuals (is the leftover error noise-like, or still structured?).

1. **`evaluateDMDFit`** — open-loop (free-running) fit over the whole matrix a DMD model was fit on.
2. **`evaluateMovingWindowDMDFit`** — same as above but for `runMovingWindowDMD` output, replaying the model locally inside each window instead of over the whole recording.
3. **`evaluateDMDOneStepFit`** — one-step-ahead fit, re-anchored on the real data every step (the most commonly used evaluator).
4. **`evaluateDMDHorizonFit`** — generalizes the one-step evaluator to an arbitrary prediction horizon `h`, still re-anchored on real data every `h` steps.
5. **`evaluateDMDHorizonSweep`** — runs `evaluateDMDHorizonFit` across a vector of horizons and returns one table, to see how fast predictive power decays with distance.
6. **`evaluateSegmentDMDs`** — held-out goodness-of-fit table across every segment from `windowStimSegments`, with segments named by their nearest stimulation event where possible.
7. **`compareBeforeAfterStimDMD`** — repeats a before-stim vs. after-stim held-out moving-window DMD comparison across every stimulation event, with before/after RMSE, R², and the per-event change (`deltaR2`, `deltaRMSE`).

## Analysis tools

1. **`plotChannelWithStim`** — visualize a raw channel with event markers (pre-removal sanity check).
2. **`plotCleanedChannel`** — visualize one channel after stimulation removal.
3. **`plotAllCleanedChannels`** — visualize every channel, stacked, after stimulation removal.
4. **`plotDMDOneStepPrediction`** — visualize actual vs. one-step DMD prediction (plus residual) for one channel.
5. **`plotAllChannelsDMDOneStepPrediction`** — same as above, all channels stacked, with each channel's R²/RMSE on its y-axis label.

## Full pipeline script

**`runFullDMDPipelineComparison.m`** — a single parameterized script (not a function) that runs the whole pipeline end to end: load → convert → segment → Hankel-embed → `combineSegmentDMDs` → held-out one-step and horizon-sweep evaluation → full per-segment table → visual checks → saves everything to a tagged `.mat` file. Edit the settings block at the top and rerun unchanged on a different recording (e.g. a different sample rate) for an apples-to-apples comparison between two saved results files.

## Requirements
 
- MATLAB
- [EEGLAB](https://sccn.ucsd.edu/eeglab/index.php) on the MATLAB path (for `pop_loadset`)

---
