"""
CLI entrypoint and shared core for audio similarity finder.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import librosa
import typer
import click
import sounddevice as sd
import soundfile as sf
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style


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
        chroma = librosa.feature.chroma_stft(
            y=clip, sr=sr, n_chroma=n_chroma, tuning=0.0
        )
        chroma_stat = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])
        feat = np.concatenate([mel_stat, chroma_stat])
        # L2 normalize per segment
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        segments.append(feat)
    return np.stack(segments, axis=0)


def extract_features(path: Path) -> Dict:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="PySoundFile failed. Trying audioread instead.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="librosa.core.audio.__audioread_load",
            category=FutureWarning,
        )
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    if not np.any(audio):
        print(f"Warning: silent audio detected in {path}", file=sys.stderr)
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


def _strip_usage(help_text: str) -> str:
    lines = help_text.splitlines()
    if lines and lines[0].startswith("Usage:"):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).rstrip()


def _print_help(ctx: click.Context) -> None:
    command = ctx.command
    if isinstance(command, click.Group):
        parts = [command.get_help(ctx).rstrip()]
        for name, sub in sorted(command.commands.items()):
            sub_ctx = sub.make_context(name, [], parent=ctx, resilient_parsing=True)
            parts.append(_strip_usage(sub.get_help(sub_ctx)))
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

    manifest = update_mod.update_index(paths, index_dir, progress_label="Indexing")
    print(f"Indexed {len(manifest)} files in {(time.time() - t0):.1f}s")


@app.command()
def query(
    query_path: Path = typer.Argument(
        ...,
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
    browse: bool = typer.Option(
        False,
        "--browse",
        help="Interactively browse results with arrow keys.",
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

    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        print("Index manifest not found. Run 'update' to build it first.")
        sys.exit(1)
    manifest = load_manifest(manifest_path)
    if not manifest:
        print("Index is empty; add WAVs to the directories and rerun.")
        sys.exit(1)
    results = query_mod.query(
        query_path=query_path,
        manifest=manifest,
        index_dir=index_dir,
        top_k=top_k,
        filter_k=filter_k,
    )
    print(f"Top {len(results)} similar files:")
    for rank, (entry, score) in enumerate(results, 1):
        print(f"{rank:2d}. {entry.path}  score={score:.3f}")
    if browse and results:
        _browse_results(results)


def _browse_results(results: List[Tuple["ManifestEntry", float]]) -> None:
    idx = 0
    playing = False
    offset = 0

    def stop_playback() -> None:
        nonlocal playing
        sd.stop()
        playing = False

    def is_playing() -> bool:
        try:
            stream = sd.get_stream()
        except Exception:  # noqa: BLE001
            return playing
        return bool(getattr(stream, "active", False))

    def _ensure_visible(height: int) -> None:
        nonlocal offset
        if idx < offset:
            offset = idx
        elif idx >= offset + height:
            offset = max(0, idx - height + 1)

    def _render() -> FormattedText:
        size = app.output.get_size()
        list_height = max(1, size.rows - 3)
        _ensure_visible(list_height)
        lines: FormattedText = []
        header = (
            "Arrows: move | Space: play/stop | Q/Esc: quit"
            f"  [{idx + 1}/{len(results)}]"
        )
        lines.append(("", header))
        lines.append(("", "\n"))
        end = min(len(results), offset + list_height)
        for i in range(offset, end):
            entry, score = results[i]
            prefix = "> " if i == idx else "  "
            line = f"{prefix}{i + 1:3d}. {entry.path}  score={score:.3f}"
            if i == idx:
                lines.append(("class:selected", line))
            else:
                lines.append(("", line))
            if i < end - 1:
                lines.append(("", "\n"))
        return lines

    def _play_selected() -> None:
        nonlocal playing
        entry, _ = results[idx]
        stop_playback()
        try:
            data, sr = sf.read(entry.path, dtype="float32", always_2d=False)
            sd.play(data, sr, blocking=False)
            playing = True
        except Exception as exc:  # noqa: BLE001
            playing = False
            status_text.text = FormattedText(
                [("class:error", f"Failed to play: {entry.path} ({exc})")]
            )
            app.invalidate()

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("left")
    def _on_up(event: object) -> None:
        nonlocal idx
        if idx > 0:
            idx -= 1
            stop_playback()
            app.invalidate()

    @kb.add("down")
    @kb.add("right")
    def _on_down(event: object) -> None:
        nonlocal idx
        if idx < len(results) - 1:
            idx += 1
            stop_playback()
            app.invalidate()

    @kb.add("pageup")
    def _on_page_up(event: object) -> None:
        nonlocal idx
        size = app.output.get_size()
        page = max(1, size.rows - 3)
        idx = max(0, idx - page)
        stop_playback()
        app.invalidate()

    @kb.add("pagedown")
    def _on_page_down(event: object) -> None:
        nonlocal idx
        size = app.output.get_size()
        page = max(1, size.rows - 3)
        idx = min(len(results) - 1, idx + page)
        stop_playback()
        app.invalidate()

    @kb.add(" ")
    def _on_space(event: object) -> None:
        if is_playing():
            stop_playback()
        else:
            _play_selected()
        app.invalidate()

    @kb.add("q")
    @kb.add("escape")
    def _on_quit(event: object) -> None:
        stop_playback()
        event.app.exit()

    list_text = FormattedTextControl(text=_render, focusable=False)
    list_window = Window(content=list_text, wrap_lines=False)
    status_text = FormattedTextControl(text=FormattedText([]), focusable=False)
    status_window = Window(content=status_text, height=1)

    style = Style.from_dict(
        {
            "selected": "reverse",
            "error": "fg:#ff5f5f",
        }
    )

    app = Application(
        layout=Layout(HSplit([list_window, status_window])),
        key_bindings=kb,
        full_screen=True,
        style=style,
    )
    app.run()


def run() -> None:
    app()


if __name__ == "__main__":
    run()
