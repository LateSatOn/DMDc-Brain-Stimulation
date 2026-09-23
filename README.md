# DMDc-Brain-Stimulation

## Pipeline overview

1. **`loadEEGSet`** — load a subject/session's `.set`/`.fdt` pair into an `EEG` struct.
2. (optional) **`EEGLAB`** for filtering
3. **`eegToDMDMatrix`** — convert the `EEG` struct into a `[time x channels]` matrix ready for DMD.
4. **`removeStimSegments`** — cut stimulation periods out of the DMD matrix.
5. (optional) **`buildHankelMatrix`** — time-delay (Hankel) embedding, for use before `runDMDOnMatrix`.
6. **`runDMDOnMatrix`** — run Dynamic Mode Decomposition on a DMD-ready matrix.

## Analysis tools

1. **`plotChannelWithStim`** — visualize a raw channel with event markers (pre-removal sanity check).
2. **`plotCleanedChannel`** — visualize one channel after stimulation removal.
3. **`plotAllCleanedChannels`** — visualize every channel, stacked, after stimulation removal.

## Requirements
 
- MATLAB
- [EEGLAB](https://sccn.ucsd.edu/eeglab/index.php) on the MATLAB path (for `pop_loadset`)

---