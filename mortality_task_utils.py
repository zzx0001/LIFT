import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from masld_task_utils import fit_static_stats, process_static_by_train_stats

# Adjust these paths for your local preprocessed-npz layout
DATA_PREPROC_DIR = Path("<PATH_TO_MIMICIV_MORTALITY_NPZ>")
EICU_DIR = Path("<PATH_TO_EICU_MORTALITY_NPZ>")

# variant -> (subdir, filename); subdir=None means DATA_PREPROC_DIR
VARIANT_NPZ = {
    "basic":                (None,     "mimiciv_mimic_basic_48h.npz"),
    "regular":              (None,     "mimiciv_mimic_regular_48h.npz"),
    "full":                 (None,     "mimiciv_mimic_full_48h.npz"),
    "eicu_hospital_basic":  (EICU_DIR, "eicu_hospital_basic_48h.npz"),
}


def load_mortality_source(variant="regular", npz_dir=None):
    """Read preprocessed npz. Returns a `source` dict consumed by
    `make_mortality_prepare_fn(...)`'s inner function."""
    if variant not in VARIANT_NPZ:
        raise ValueError(f"variant must be one of {list(VARIANT_NPZ)}, got {variant!r}")
    subdir, fname = VARIANT_NPZ[variant]
    if npz_dir is not None:
        npz_path = Path(npz_dir) / fname
    else:
        npz_path = (subdir if subdir is not None else DATA_PREPROC_DIR) / fname
    if not npz_path.exists():
        raise FileNotFoundError(f"Preprocessed npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {
        "variant": variant,
        "npz_path": str(npz_path),
        "X_ts": np.asarray(data["X_ts"]),
        "X_static": np.asarray(data["X_static"]),
        "y": np.asarray(data["y"]),
        "ts_feature_names": [str(s) for s in data["ts_feature_names"]],
        "static_feature_names": [str(s) for s in data["static_feature_names"]],
    }


def _fit_ts_stats(X_train_ts, eps=1e-6):
    """Channel-wise mean/std over (N, T) for each F. NaN-robust."""
    F = X_train_ts.shape[-1]
    flat = X_train_ts.reshape(-1, F).astype(np.float64)
    flat[~np.isfinite(flat)] = np.nan
    mean = np.nanmean(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(flat, axis=0)
    std = np.where(np.isfinite(std), std, 1.0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_ts_stats(X, mean, std, fill_value=0.0):
    """Z-score then mean-impute NaN."""
    X = X.astype(np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    X = (X - mean) / std
    # NaN after normalize → fill with 0 (= train mean)
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        X[nan_mask] = fill_value
    return X.astype(np.float32)


def _broadcast_concat(X_ts, X_static):
    """(N, T, F_ts) + (N, F_st) → (N, T, F_ts + F_st)."""
    if X_static is None or X_static.shape[1] == 0:
        return X_ts
    Xs_rep = np.repeat(X_static[:, None, :], X_ts.shape[1], axis=1)
    return np.concatenate([X_ts, Xs_rep.astype(np.float32)], axis=-1)


def prepare_mortality_split(source, seed, val_frac=0.1, test_frac=0.1):
    """Stratified split + train-stats normalize + broadcast-concat.

    Returns a `prepared` dict with the keys plugin_8 / plugin_9 runners read:
      X_train_combined / X_val_combined / X_test_combined  (N, T, F_ts + F_st)
      y_train / y_val / y_test
      n_steps              = T
      n_features_ts        = F_ts (used as plugin_input_features in adapter)
      n_features_total     = F_ts + F_st (input dim seen by backbone)
    plus auxiliary fields useful for logging / reproducibility.
    """
    X_ts = source["X_ts"]
    X_static = source["X_static"]
    y = source["y"].astype(np.int64)
    N, T, F_ts = X_ts.shape
    F_st = X_static.shape[1] if X_static.ndim == 2 else 0

    # Two-step stratified split: 80% train / 10% val / 10% test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.arange(N), y))
    rel_val = val_frac / (1.0 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    rel_train, rel_val_inner = next(sss2.split(np.arange(len(trainval_idx)), y[trainval_idx]))
    train_idx = trainval_idx[rel_train]
    val_idx = trainval_idx[rel_val_inner]

    # TS normalize by train stats (channel-wise z-score, NaN-fill with mean)
    ts_mean, ts_std = _fit_ts_stats(X_ts[train_idx])
    X_train_ts = _apply_ts_stats(X_ts[train_idx], ts_mean, ts_std)
    X_val_ts = _apply_ts_stats(X_ts[val_idx], ts_mean, ts_std)
    X_test_ts = _apply_ts_stats(X_ts[test_idx], ts_mean, ts_std)

    # Static normalize by train stats (reuse liver helpers)
    if F_st > 0:
        st_mean, st_std = fit_static_stats(X_static[train_idx])
        X_train_st = process_static_by_train_stats(X_static[train_idx], st_mean, st_std)
        X_val_st = process_static_by_train_stats(X_static[val_idx], st_mean, st_std)
        X_test_st = process_static_by_train_stats(X_static[test_idx], st_mean, st_std)
    else:
        st_mean, st_std = None, None
        X_train_st = np.zeros((len(train_idx), 0), dtype=np.float32)
        X_val_st = np.zeros((len(val_idx), 0), dtype=np.float32)
        X_test_st = np.zeros((len(test_idx), 0), dtype=np.float32)

    # Broadcast-concat static into ts feature dim
    X_train_combined = _broadcast_concat(X_train_ts, X_train_st)
    X_val_combined = _broadcast_concat(X_val_ts, X_val_st)
    X_test_combined = _broadcast_concat(X_test_ts, X_test_st)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    return {
        # required by runners
        "X_train_combined": X_train_combined,
        "X_val_combined": X_val_combined,
        "X_test_combined": X_test_combined,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "n_steps": T,
        "n_features_ts": F_ts,
        "n_features_total": F_ts + F_st,
        # extras used by some code paths (run_liver_lift_plugin_models reads these via .get)
        "X_train_ts_norm": X_train_ts,
        "X_val_ts_norm": X_val_ts,
        "X_test_ts_norm": X_test_ts,
        "X_train_static_norm": X_train_st,
        "X_val_static_norm": X_val_st,
        "X_test_static_norm": X_test_st,
        "n_features_static": F_st,
        # metadata
        "variant": source["variant"],
        "feature_cols": source["ts_feature_names"],
        "static_cols": source["static_feature_names"],
        "ts_stats": {"mean": ts_mean, "std": ts_std},
        "static_stats": (
            {"mean": st_mean, "std": st_std, "static_cols": source["static_feature_names"]}
            if st_mean is not None else None
        ),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }


def make_mortality_prepare_fn(val_frac=0.1, test_frac=0.1):
    """Factory matching the (source, seed) signature plugin runners expect."""
    def _prepare(source, seed):
        return prepare_mortality_split(source, seed, val_frac=val_frac, test_frac=test_frac)
    return _prepare


def make_mortality_plugin_config_fn(num_atoms=24, emb_dim=32, grid_size=48,
                                    delta_t_mode="time_step",
                                    use_ftm=True, use_density_comp=True,
                                    plugin_dropout=None):
    """Mortality plug-in config: start from `default_lift_plugin_config(48)`
    auto-scaled periods (trend 24-48h, event 1-24h, beta_init scaled to
    window). Only override generic knobs (no period overrides, unlike liver
    runner which hardcodes for window=12)."""
    sys.path.insert(0, str(CODE_DIR / "lift_plugin_8"))
    from liver_lift_plugin_runner import default_lift_plugin_config

    def _cfg(prepared):
        cfg = default_lift_plugin_config(window_size=prepared["n_steps"])
        cfg.update({
            "num_atoms": int(num_atoms),
            "emb_dim": int(emb_dim),
            "grid_size": int(grid_size),
            "pooling": "max",
            "attn_hidden_dim": 128,
            "delta_t_mode": delta_t_mode,
            "trust_temperature_init": 3.0,
        })
        cfg["use_ftm"] = bool(use_ftm)
        cfg["use_density_comp"] = bool(use_density_comp)
        if plugin_dropout is not None:
            cfg["dropout"] = plugin_dropout
            cfg["head_dropout"] = plugin_dropout
        return cfg
    return _cfg


if __name__ == "__main__":
    # quick smoke check
    for variant in ("basic", "regular", "full"):
        try:
            src = load_mortality_source(variant)
            print(f"[{variant}] X_ts {src['X_ts'].shape}  X_static {src['X_static'].shape}  "
                  f"y mean {src['y'].mean():.3f}  feats {len(src['ts_feature_names'])}+"
                  f"{len(src['static_feature_names'])}")
            prep = prepare_mortality_split(src, seed=0)
            print(f"           train {prep['n_train']}  val {prep['n_val']}  test {prep['n_test']}  "
                  f"combined dim {prep['X_train_combined'].shape[-1]}")
        except FileNotFoundError as e:
            print(f"[{variant}] MISSING: {e}")
