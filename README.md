# Audio Similarity Finder

Python CLI to find WAV files that sound similar to a query clip. It:
- Reads search roots from a text file (one directory per line)
- Builds/updates an index under `./index/` (manifest + per-file feature NPZ)
- Extracts segmented log-mel + chroma embeddings for key/length tolerance
- Searches with cosine similarity (centroid prefilter + window-level refine)

## Versions:
0.1.0 - First version from Chat--filter-GPT.\
0.2.0 - Vibe coding to allow browsing through sounds and play them, informative text and progress-bars and management of corrupt or empyt wav-files.
0.3.0 - Vibe coding to add option to exclude patterns when creating result.

## Install (dev)

### Python
Install Python 3.11

### Dev
- Install VS Code
- Install Git for Windows: https://gitforwindows.org/
- Install GitHub CLI: https://cli.github.com/

### Extensions
Add the following extensions in VS Code:
- Python + Pylance (Microsoft)
- ESLint (for Node/TS)
- Prettier (formatting)
- GitHub Pull Requests and Issues

### Setup Git
Inside project-folder:
```
git config --global user.name "MYUSERNAMEINGIT"
git config --global user.email "MYEMAILINGIT"
git init
git add .
git commit -m "Initial commit"
gh auth login
gh repo create audio-find-cli --source=. --private --push
```

### Setup PowerShell
Run the following PowerShell-command: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted`

### Setup the environment
In PowerShell run:
```
python -m venv .venv
.\.venv\Scripts\activate
```

### Build
In PowerShell run:
```
pip install -e .
```

## Usage
Usage: audio-find-cli [OPTIONS] COMMAND [ARGS]

Commands:
```
update              Create/update index based on wav-files from path-file.
query               Perform query on index based on wav-file, list top-k. Possibility also to browse through the result and play sounds.
```

### Update
Usage: audio-find-cli update [OPTIONS]

Options:
```
--paths              PATH     Text file with directories to scan (one per line). [default: paths.txt]
                              Can include commented lines '#'.
--index-dir          PATH     Index directory (manifest + features). [default: index]
--help       -h               Show help and exit.  
```

### Query
Usage: audio-find-cli query [OPTIONS] QUERY_PATH

Arguments:
```
query_path           PATH     Query WAV file. [required] 
```

Options:
```
--index-dir          PATH     Index directory (manifest + features). [default: index]
--top-k              INTEGER  Number of top results to show. [default: 10]
--filter-k           INTEGER  Candidates to refine after centroid filter (0 = all). [default: 50]
                              Controls how many top centroid candidates are refined with window-level matching. Lower values are faster but may miss good matches; higher values are slower but more thorough. Use `0` to refine all indexed files (best recall, slowest), a small number like `20-50` for quick interactive searches, and larger values when your library is noisy or you want stronger recall.
--name-pattern       TEXT     Filter indexed paths by name (supports wildcards like * and ?). If no wildcard is provided, matches as a substring.
--exclude-pattern    TEXT     Exclude indexed paths by name (supports wildcards like * and ?). If no wildcard is provided, matches as a substring.
--browse                      Interactive way to listen to results: use arrow keys to move and spacebar to play/stop the current WAV.
--help       -h               Show help and exit.  
```

## Example query of similar files
```
audio_find_cli query --top-k 10 --filter-k 70 C:\audio\snippet.wav 
```

## Notes
- Unreadable wav-files will be appended in to `unreadable.txt` in `index` directory.
- Index updates in-place: new/changed files are re-extracted; missing files are dropped.
- Features are segmented (default 3s window, 1.5s hop) with log-mel + chroma stats and L2 normalization.
- Defaults balance speed/quality on CPU; adjust constants in `src/audio_find_cli/cli.py` for larger windows or more mels.
