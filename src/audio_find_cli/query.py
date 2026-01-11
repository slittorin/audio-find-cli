from __future__ import annotations

from pathlib import Path
import fnmatch
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .cli import ManifestEntry, cosine_sim, extract_features


def load_features(index_dir: Path, entry: ManifestEntry) -> Dict[str, np.ndarray]:
    feat_path = index_dir / entry.feature_file
    data = np.load(feat_path)
    return {"segments": data["segments"], "centroid": data["centroid"]}


def score_query(
    query_feats: Dict[str, np.ndarray],
    candidates: List[Tuple[ManifestEntry, Dict[str, np.ndarray]]],
    candidate_filter_weight: float = 0.7,
) -> List[Tuple[ManifestEntry, float]]:
    results = []
    q_centroid = query_feats["centroid"]
    q_segments = query_feats["segments"]
    for entry, feats in candidates:
        c_centroid = feats["centroid"]
        base = cosine_sim(q_centroid, c_centroid)
        # Window-level refine: average of best matches per query segment
        seg_scores = []
        segs = feats["segments"]
        for q in q_segments:
            # cosine vs all segments of candidate
            dots = segs @ q
            norms = np.linalg.norm(segs, axis=1) * np.linalg.norm(q)
            norms = np.where(norms == 0, 1e-9, norms)
            sims = dots / norms
            seg_scores.append(np.max(sims))
        refine = float(np.mean(seg_scores)) if seg_scores else base
        score = candidate_filter_weight * base + (1 - candidate_filter_weight) * refine
        results.append((entry, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def query(
    query_path: Path,
    manifest: Dict[str, ManifestEntry],
    index_dir: Path,
    top_k: int,
    filter_k: int,
    name_pattern: str | None = None,
    name_exclude_pattern: str | None = None,
    show_progress: bool = True,
) -> List[Tuple[ManifestEntry, float]]:
    if show_progress:
        print("Loading query and scanning index...")
    q_feats = extract_features(query_path)
    # coarse filter by centroid
    centroids = []
    entries = list(manifest.values())
    if name_pattern:
        pattern = name_pattern
        if not any(ch in pattern for ch in "*?[]"):
            pattern = f"*{pattern}*"
        entries = [
            entry
            for entry in entries
            if fnmatch.fnmatch(str(entry.path), pattern)
        ]
    if name_exclude_pattern:
        pattern = name_exclude_pattern
        if not any(ch in pattern for ch in "*?[]"):
            pattern = f"*{pattern}*"
        entries = [
            entry
            for entry in entries
            if not fnmatch.fnmatch(str(entry.path), pattern)
        ]
    iterator = (
        tqdm(entries, desc="Searching index", unit="file")
        if show_progress
        else entries
    )
    for entry in iterator:
        feat = load_features(index_dir, entry)
        centroids.append((entry, cosine_sim(q_feats["centroid"], feat["centroid"])))
    centroids.sort(key=lambda x: x[1], reverse=True)
    if filter_k > 0:
        candidates_entries = [e for e, _ in centroids[:filter_k]]
    else:
        candidates_entries = [e for e, _ in centroids]
    candidates = [(e, load_features(index_dir, e)) for e in candidates_entries]
    scored = score_query(q_feats, candidates)
    return scored[:top_k]
