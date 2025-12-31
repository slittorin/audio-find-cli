# Audio Similarity Finder

Python 3.11 CLI to find WAV files that sound similar to a query clip. It:
- Reads search roots from a text file (one directory per line)
- Builds/updates an index under `./index/` (manifest + per-file feature NPZ)
- Extracts segmented log-mel + chroma embeddings for key/length tolerance
- Searches with cosine similarity (centroid prefilter + window-level refine)

## Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Prepare paths file
`paths.txt` example:
```
C:\samples\drums
Z:\libraries\pads
# comment lines are ignored
```

## Update the index
```bash
python -m audio_find_cli.cli update --paths paths.txt --index-dir index
# or after install: audio-find update --paths paths.txt --index-dir index
```

## Query similar files
```bash
python -m audio_find_cli.cli query --paths paths.txt --query C:\audio\snippet.wav --top-k 10 --filter-k 50
# or: audio-find query --paths paths.txt --query C:\audio\snippet.wav --top-k 10 --filter-k 50
```
- `--filter-k` controls how many top centroid candidates are refined with window-level matching. Lower values are faster but may miss good matches; higher values are slower but more thorough. Use `0` to refine all indexed files (best recall, slowest), a small number like `20-50` for quick interactive searches, and larger values when your library is noisy or you want stronger recall.

## Notes
- Index updates in-place: new/changed files are re-extracted; missing files are dropped.
- Features are segmented (default 3s window, 1.5s hop) with log-mel + chroma stats and L2 normalization.
- Defaults balance speed/quality on CPU; adjust constants in `src/audio_find_cli/cli.py` for larger windows or more mels.
