from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from .cli import (
    ManifestEntry,
    extract_features,
    file_sig,
    hash_path,
    iter_wavs,
    load_manifest,
    load_paths_file,
    save_manifest,
)


def update_index(paths_file: Path, index_dir: Path) -> Dict[str, ManifestEntry]:
    index_dir.mkdir(parents=True, exist_ok=True)
    features_dir = index_dir / "features"
    features_dir.mkdir(exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    target_dirs = load_paths_file(paths_file)
    wav_files = list(iter_wavs(target_dirs))
    wav_set = {str(p.resolve()) for p in wav_files}

    # Remove stale entries
    stale_keys = [k for k, v in manifest.items() if v.path not in wav_set]
    for k in stale_keys:
        mf = index_dir / manifest[k].feature_file
        if mf.exists():
            mf.unlink(missing_ok=True)
        manifest.pop(k, None)

    for path in tqdm(wav_files, desc="Indexing"):
        real_path = path.resolve()
        sig = file_sig(real_path)
        key = str(real_path)
        needs_update = False
        if key not in manifest:
            needs_update = True
        else:
            entry = manifest[key]
            if entry.mtime != sig[0] or entry.size != sig[1]:
                needs_update = True
        if not needs_update:
            continue

        feats = extract_features(real_path)
        file_id = hash_path(real_path)
        feat_file = features_dir / f"{file_id}.npz"
        np.savez_compressed(
            feat_file,
            segments=feats["segments"],
            centroid=feats["centroid"],
            sr=feats["sr"],
            window_sec=feats["window_sec"],
            hop_sec=feats["hop_sec"],
        )
        manifest[key] = ManifestEntry(
            id=file_id,
            path=str(real_path),
            mtime=sig[0],
            size=sig[1],
            feature_file=str(feat_file.relative_to(index_dir)),
            n_segments=feats["segments"].shape[0],
        )

    save_manifest(manifest_path, manifest)
    return manifest
