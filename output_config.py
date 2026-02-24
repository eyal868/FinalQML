"""
output_config.py — Central output path management for FinalQML.

All scripts import from here to get consistent dual-save behavior:
  1. Repo:    outputs/{category}/...          (CSVs tracked in git, PNGs gitignored)
  2. Desktop: ~/Desktop/FinalQML_Outputs/YYYY-MM-DD/{experiment}_{HHMMSS}/...

Usage:
    from output_config import get_run_dirs, save_file, save_run_info

    repo_dir, desktop_dir = get_run_dirs("qaoa_unweighted/N12")
    df.to_csv(repo_dir / "qaoa_sweep_N12_p1to10.csv", index=False)
    # Desktop copy is handled automatically by save_file(), or manually:
    shutil.copy2(repo_dir / "file.csv", desktop_dir / "file.csv")
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =========================================================================
# PATHS
# =========================================================================

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

# Repo-local output directory (CSVs committed, PNGs gitignored)
REPO_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Desktop mirror with date/timestamp organization
DESKTOP_OUTPUT_ROOT = Path.home() / "Desktop" / "FinalQML_Outputs"

# =========================================================================
# CORE FUNCTIONS
# =========================================================================

def get_run_dirs(experiment_name: str, timestamp: bool = True):
    """
    Create and return (repo_dir, desktop_dir) for a given experiment.

    Args:
        experiment_name: e.g. "qaoa_unweighted/N12", "spectral_gap", "figures"
        timestamp: If True, Desktop subfolder gets HHMMSS suffix.

    Returns:
        (repo_dir, desktop_dir) as Path objects, both already created.
    """
    # Repo path: outputs/{experiment_name}/
    repo_dir = REPO_OUTPUT_DIR / experiment_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Desktop path: ~/Desktop/FinalQML_Outputs/YYYY-MM-DD/{experiment}_{HHMMSS}/
    date_str = datetime.now().strftime("%Y-%m-%d")
    if timestamp:
        time_str = datetime.now().strftime("%H%M%S")
        desktop_experiment = f"{experiment_name.replace('/', '_')}_{time_str}"
    else:
        desktop_experiment = experiment_name.replace("/", "_")

    desktop_dir = DESKTOP_OUTPUT_ROOT / date_str / desktop_experiment
    desktop_dir.mkdir(parents=True, exist_ok=True)

    return repo_dir, desktop_dir


def save_file(source_path, experiment_name: str, filename: str = None,
              timestamp: bool = True, _desktop_dir: Path = None):
    """
    Copy an already-saved repo file to the Desktop mirror.

    Args:
        source_path: Path to the file in the repo outputs/ directory.
        experiment_name: e.g. "qaoa_unweighted/N12"
        filename: Override destination filename (default: same as source).
        timestamp: Whether Desktop subfolder gets HHMMSS suffix.
        _desktop_dir: If provided, use this Desktop dir instead of creating a new one.

    Returns:
        Path to the Desktop copy.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        print(f"⚠️  Source file not found, skipping Desktop copy: {source_path}")
        return None

    if _desktop_dir is not None:
        desktop_dir = _desktop_dir
    else:
        _, desktop_dir = get_run_dirs(experiment_name, timestamp=timestamp)

    dest_filename = filename or source_path.name
    dest_path = desktop_dir / dest_filename
    shutil.copy2(source_path, dest_path)
    return dest_path


def save_dual(data_or_fig, repo_path, experiment_name: str,
              _desktop_dir: Path = None, **kwargs):
    """
    Save a DataFrame (to_csv) or matplotlib Figure (savefig) to both locations.

    Args:
        data_or_fig: pandas DataFrame or matplotlib Figure.
        repo_path: Full path in the repo outputs/ directory.
        experiment_name: For Desktop mirror organization.
        _desktop_dir: If provided, reuse this Desktop dir.
        **kwargs: Passed to to_csv() or savefig().

    Returns:
        (repo_path, desktop_path)
    """
    import matplotlib.figure

    repo_path = Path(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    if _desktop_dir is None:
        _, _desktop_dir = get_run_dirs(experiment_name, timestamp=True)

    desktop_path = _desktop_dir / repo_path.name

    if isinstance(data_or_fig, matplotlib.figure.Figure):
        csv_kwargs = {}
        fig_kwargs = {"dpi": 300, "bbox_inches": "tight"}
        fig_kwargs.update(kwargs)
        data_or_fig.savefig(repo_path, **fig_kwargs)
        data_or_fig.savefig(desktop_path, **fig_kwargs)
    else:
        # Assume pandas DataFrame
        csv_kwargs = {"index": False}
        csv_kwargs.update(kwargs)
        data_or_fig.to_csv(repo_path, **csv_kwargs)
        data_or_fig.to_csv(desktop_path, **csv_kwargs)

    return repo_path, desktop_path


def save_run_info(desktop_dir, experiment_name: str = "",
                  extra_info: dict = None):
    """
    Write run_info.json to a Desktop run folder for reproducibility.

    Includes: git commit, timestamp, CLI args, experiment name, and any extra info.
    """
    desktop_dir = Path(desktop_dir)
    desktop_dir.mkdir(parents=True, exist_ok=True)

    # Get git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"

    # Get git dirty status
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = bool(git_status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_dirty = None

    info = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_hash,
        "git_dirty": git_dirty,
        "cli_args": sys.argv,
        "python_version": sys.version,
    }

    if extra_info:
        info["notes"] = extra_info

    info_path = desktop_dir / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"📋 Run info saved: {info_path}")
    return info_path
