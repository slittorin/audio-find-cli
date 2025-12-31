"""
CLI entrypoint and shared core for audio similarity finder.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import librosa
import typer
import click


# Defaults tuned for balance of speed/quality on CPU
SAMPLE_RATE = 22050
WINDOW_SEC = 3.0
HOP_SEC = 1.5
N_MELS = 64
N_CHROMA = 12


def load_paths_file(paths_file: Path) -> List[Path]:
    """Read newline-separated directories and expand environment variables."""
    paths = []
    with paths_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(os.path.expandvars(line)).expanduser())
    return paths


def iter_wavs(directories: Iterable[Path]) -> Iterable[Path]:
    exts = {".wav", ".wave"}
    for d in directories:
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.suffix.lower() in exts and path.is_file():
                yield path


def file_sig(path: Path) -> Tuple[float, int]:
    stat = path.stat()
    return stat.st_mtime, stat.st_size


def hash_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_segments(
    audio: np.ndarray,
    sr: int,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
    n_mels: int = N_MELS,
    n_chroma: int = N_CHROMA,
) -> np.ndarray:
    """Return array of shape (n_segments, feat_dim)."""
    window = int(window_sec * sr)
    hop = int(hop_sec * sr)
    if len(audio) < window:
        # pad short audio
        audio = np.pad(audio, (0, window - len(audio)))
    segments = []
    for start in range(0, len(audio) - window + 1, hop):
        clip = audio[start : start + window]
        mel = librosa.feature.melspectrogram(
            y=clip, sr=sr, n_mels=n_mels, hop_length=512, n_fft=2048
        )
        logmel = librosa.power_to_db(mel + 1e-9)
        mel_stat = np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)])
        chroma = librosa.feature.chroma_stft(y=clip, sr=sr, n_chroma=n_chroma)
        chroma_stat = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])
        feat = np.concatenate([mel_stat, chroma_stat])
        # L2 normalize per segment
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        segments.append(feat)
    return np.stack(segments, axis=0)


def extract_features(path: Path) -> Dict:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    segments = extract_segments(audio, sr)
    centroid = segments.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    return {
        "segments": segments.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "sr": sr,
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
    }


@dataclass
class ManifestEntry:
    id: str
    path: str
    mtime: float
    size: int
    feature_file: str
    n_segments: int


def load_manifest(path: Path) -> Dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: ManifestEntry(**v) for k, v in data.items()}


def save_manifest(path: Path, manifest: Dict[str, ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in manifest.items()}, f, indent=2)


app = typer.Typer(
    add_help_option=False,
    add_completion=False,
    help="Find similar WAV files.",
)


def _print_help(ctx: click.Context) -> None:
    command = ctx.command
    if isinstance(command, click.Group):
        parts = [command.get_help(ctx).rstrip()]
        for name, sub in sorted(command.commands.items()):
            sub_ctx = click.Context(sub, parent=ctx)
            parts.append(sub.get_help(sub_ctx).rstrip())
        typer.echo("\n\n".join(parts))
    else:
        typer.echo(command.get_help(ctx))


def _help_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    if value and not ctx.resilient_parsing:
        _print_help(ctx)
        raise typer.Exit()
    return value


@app.callback(invoke_without_command=True)
def _root(
    _show_help: bool = typer.Option(
        False,
        "--help",
        "-h",
        is_eager=True,
        callback=_help_callback,
        help="Show this message and exit.",
    ),
) -> None:
    if _show_help:
        return


@app.command()
def update(
    paths: Path = typer.Option(
        Path("paths.txt"),
        "--paths",
        help="Text file with directories to scan (one per line).",
    ),
    index_dir: Path = typer.Option(
        Path("index"),
        "--index-dir",
        help="Index directory (manifest + features).",
    ),
    _show_help: bool = typer.Option(
        False,
        "--help",
        "-h",
        is_eager=True,
        callback=_help_callback,
        help="Show this message and exit.",
    ),
) -> None:
    t0 = time.time()
    from . import update as update_mod

    manifest = update_mod.update_index(paths, index_dir)
    print(f"Indexed {len(manifest)} files in {(time.time() - t0):.1f}s")


@app.command()
def query(
    paths: Path = typer.Option(
        Path("paths.txt"),
        "--paths",
        help="Text file with directories to scan (one per line).",
    ),
    query: Path = typer.Option(
        ...,
        "--query",
        help="Query WAV file.",
    ),
    index_dir: Path = typer.Option(
        Path("index"),
        "--index-dir",
        help="Index directory (manifest + features).",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Number of top results to show.",
    ),
    filter_k: int = typer.Option(
        50,
        "--filter-k",
        help="Candidates to refine after centroid filter (0 = all).",
    ),
    _show_help: bool = typer.Option(
        False,
        "--help",
        "-h",
        is_eager=True,
        callback=_help_callback,
        help="Show this message and exit.",
    ),
) -> None:
    t0 = time.time()
    from . import query as query_mod
    from . import update as update_mod

    manifest = update_mod.update_index(paths, index_dir)
    if not manifest:
        print("Index is empty; add WAVs to the directories and rerun.")
        sys.exit(1)
    results = query_mod.query(
        query_path=query,
        manifest=manifest,
        index_dir=index_dir,
        top_k=top_k,
        filter_k=filter_k,
    )
    print(f"Top {len(results)} similar files:")
    for rank, (entry, score) in enumerate(results, 1):
        print(f"{rank:2d}. {entry.path}  score={score:.3f}")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
