#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-Port Decision Analyzer — pre-trial-style

* plots every 5-s **post-trial** trajectory on an arena image
* returns a per-trial DataFrame + a per-session stats dict
* makes diagnostic plots (can be disabled)

Assumes
  1) BehaviorCSVReader & CSVDataReader are import-able helpers
  2) Body part “Nose”
  3) video recorded at `dlc_fps` Hz  (default 25)
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Sequence
from PIL import Image
import sys
from BehaviorCSVReader import BehaviorCSVReader  
from CSVDataReader     import CSVDataReader 
from scipy.stats import friedmanchisquare     

# helper function to create a spatial area around the WP coordinates
def _in_circle(pt: np.ndarray, centre: Tuple[float, float], r: float) -> bool:
    """Euclidean point-in-circle test."""
    return np.hypot(pt[0] - centre[0], pt[1] - centre[1]) <= r


class WaterPortDecisionAnalyzer:
    """Analysis of the first decisions from mice after trial initiation"""


    SIDE_LABEL = {0: "left", 1: "right"} # codes within Sensor df

    def __init__(
        self,
        low_port: Tuple[float, float],
        high_port: Tuple[float, float],
        radius: float,
        *,
        dlc_fps: int = 25,
        window_ms: int = 5_000,
    ):
        # arena geometry --------------------------------------------------
        self.low_port  = low_port
        self.high_port = high_port
        self.radius    = radius

        # timing ----------------------------------------------------------
        self.dlc_fps   = dlc_fps
        self.dt_ms     = 1000.0 / dlc_fps
        self.window_ms = window_ms


        self._correct_map = {0: "low", 1: "high"}


    def analyse_session(
        self,
        beh_csv: Path | str,
        dlc_csv: Path | str,
        *,
        bg_img: Optional[Path | str] = None,
        plot: bool = True,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        """
        Parameters
        ----------
        beh_csv : behaviour CSV (sensor events)
        dlc_csv : DLC coordinates CSV
        bg_img  : optional background image
        plot    : if True make trajectory/count plots
        """
        beh_csv = Path(beh_csv)
        dlc_csv = Path(dlc_csv)

        # 1) read out the sensor df, get trial events
        br = BehaviorCSVReader(beh_csv.parent, beh_csv.name)
        _, info = br.read_behavior_csv()
        trials: Sequence[Tuple[float, int]] = info.get("TrialTimes", [])
        if not trials:
            print("no trials in", beh_csv)
            return pd.DataFrame(), {}

        # 2) read out DLC dataframe, reading out spatial information of nose 
        cdr   = CSVDataReader(dlc_csv)
        parts = cdr.get_data_from_csv(exclude_list=["bodyparts"], omit_prediction=False)
        if "Nose" not in parts:
            print("Nose missing in", dlc_csv)
            return pd.DataFrame(), {}

        arr          = parts["Nose"] # change if desired
        frame_times  = np.arange(arr.shape[0]) * self.dt_ms

        # 3) trial loop --------------------------------------------------
        rows:  List[Dict]                                   = []
        paths: List[Tuple[np.ndarray, List[int], int]]      = []

        for idx, (start_ms, side) in enumerate(trials):
            coords, _           = self._window(arr, frame_times, start_ms)
            first, corr, alts, switch_idx = self._analyse_trial(coords, side)

            rows.append(
                dict(TrialIdx      = idx,
                     TrialSide     = self.SIDE_LABEL.get(side, "?"),
                     FirstPort     = first,
                     Correct       = corr,
                     Alternations  = alts)
            )
            paths.append((coords, switch_idx, side))

        df    = pd.DataFrame(rows)
        stats = self._session_stats(df)

        if plot:
            mouse, sess = dlc_csv.name[4], dlc_csv.name[6:12]
            self._plot_session(paths, mouse, sess, bg_img)
            #self._plot_counts(stats, mouse, sess)


        return df, stats

    # ─────────────────────────────────────────────────────────────────
    # internals used within analysis
    # ─────────────────────────────────────────────────────────────────
    def _window(
        self,
        arr: np.ndarray,
        frame_times: np.ndarray,
        start_ms: float
    ) -> tuple[np.ndarray, np.ndarray]:
        m = (frame_times >= start_ms) & (frame_times < start_ms + self.window_ms)
        return arr[m, :2], frame_times[m]

    def _analyse_trial(
        self,
        coords: np.ndarray,
        side: int
    ) -> tuple[Optional[str], Optional[bool], int, List[int]]:
        first: Optional[str]   = None
        visits: List[str]      = []
        switch_pts: List[int]  = []

        for idx, pt in enumerate(coords):
            lab = None
            if _in_circle(pt, self.low_port,  self.radius): lab = "low" # if first visited port low, variable is assigned to "low"
            elif _in_circle(pt, self.high_port, self.radius): lab = "high"

            if lab and (not visits or lab != visits[-1]):
                visits.append(lab)
                if len(visits) > 1:
                    switch_pts.append(idx)
            if first is None and lab:
                first = lab

        correct = first == self._correct_map.get(side) if first else None # matching correct side with side first visited
        return first, correct, max(0, len(visits) - 1), switch_pts

    def _session_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        left_df,  right_df = df[df.TrialSide == "left"], df[df.TrialSide == "right"]
        c_left,   ic_left  = left_df.Correct.sum(),  left_df.Correct.eq(False).sum()
        c_right,  ic_right = right_df.Correct.sum(), right_df.Correct.eq(False).sum()

        n_left,   n_right  = c_left + ic_left, c_right + ic_right
        pct_left, pct_right = (
            (c_left / n_left)  * 100 if n_left  else np.nan,
            (c_right / n_right) * 100 if n_right else np.nan,
        )

        fp_low, fp_high = (df.FirstPort == "low").sum(), (df.FirstPort == "high").sum()
        bias = fp_high / (fp_low + fp_high) if (fp_low + fp_high) else np.nan

        return dict(
            Trials       = len(df), # all trials
            Correct_L    = pct_left,
            Incorrect_L  = ic_left,
            Correct_R    = pct_right,
            Incorrect_R  = ic_right,
            TotalAlt     = df.Alternations.sum(),
            PctCorrect   = df.Correct.mean() * 100 if len(df) else np.nan,
            MeanAlt      = df.Alternations.mean() if len(df) else np.nan,
            MedianAlt    = df.Alternations.median() if len(df) else np.nan,
            BiasRight_FP = bias
        )

    # ─────────────────────────────────────────────────────────────────
    # plotting helpers
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _add_circle(ax, centre, radius, label, color="red"):
        ax.add_patch(
            plt.Circle(centre, radius, color=color, alpha=.3, label=f"{label} port")
        )

    def _plot_session(
        self,
        paths: Sequence[Tuple[np.ndarray, List[int], int]],
        mouse: str,
        session: str,
        bg_img: Optional[Path | str] = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        width, height = 1200, 600

        # background
        if bg_img is not None and Path(bg_img).is_file():
            img = Image.open(bg_img).resize((width, height), Image.Resampling.LANCZOS)
            ax.imshow(img, origin="lower")
        elif bg_img:
            print("Background image not found:", bg_img)

        ax.set_xlim(0, width); ax.set_ylim(0, height)

        # port circles
        self._add_circle(ax, self.low_port,  self.radius, "low",  "red")
        self._add_circle(ax, self.high_port, self.radius, "high", "yellow")

        # trajectories
        for coords, switch_idx, side in paths:
            col = "red" if side == 0 else "yellow"
            ax.plot(coords[:, 0], coords[:, 1], lw=1, alpha=.25, color=col)
            for idx in switch_idx:
                ax.plot(coords[idx, 0], coords[idx, 1],
                        'green', markersize=3, zorder=5)

        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        plt.tight_layout(); plt.show()


        plt.show()

import os 

def run_waterport_decision_analysis(base_path, low, high, radius, bg_img, out_dir):
    analyzer = WaterPortDecisionAnalyzer(low, high, radius)

    session_stats = []
    summary_frames = []

    # Helper functions
    def find_top_folders(base_path):
        # Collect only valid top folders and sort them for predictable order
        top_folders = [Path(r) / 'top' for r, ds, _ in os.walk(base_path) if 'top' in ds]
        return sorted(top_folders)  # ensures consistent and expected processing order
    def match_beh(folder: Path):
        for f in folder.parent.iterdir():
            if f.suffix == '.csv' and not f.name.startswith('._'):
                return f
        return None

    # base_path = Path(base_path)

    for top in find_top_folders(base_path):
        print('\n', top)
        beh = match_beh(top)
        if beh is None:
            print('⚠ No behavior CSV in', top.parent)
            continue

        for dlc in top.glob('*filtered.csv'):
            if dlc.name.startswith('._'):
                continue
            try:
                df, stats = analyzer.analyse_session(beh, dlc, bg_img=bg_img)
                if not df.empty:
                    df.insert(0, 'Mouse', dlc.name[4])
                    df.insert(1, 'Session', dlc.name[6:12])
                    summary_frames.append(df)
                    stats.update({'Mouse': dlc.name[4], 'Session': dlc.name[6:12]})
                    session_stats.append(stats)
            except Exception as e:
                print('Error processing', dlc.name, '\n', e)

    if summary_frames:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.concat(summary_frames, ignore_index=True).to_csv(out_dir / 'waterport_decision_summary.csv', index=False)
        stats_df = pd.DataFrame(session_stats)
        stats_df.to_csv(out_dir / 'waterport_session_stats.csv', index=False)
        print('✔ Saved summary & stats to', out_dir)

        # ---- development plots ----
        topickle(stats_df)
        plot_development(stats_df)
        plot_group_summary(stats_df)
        plot_plots_group_summary(stats_df)

        for mouse_id in stats_df["Mouse"].unique():
            df_mouse = stats_df[stats_df["Mouse"] == mouse_id]
            analyzer._plot_trials_vs_accuracy(df_mouse, mouse=mouse_id)
    else:
        print('No valid sessions processed.')

def topickle(stats_df):
    stats_df.to_pickle("/Volumes/Expansion/New_cleaned_data/statsdf/allmice/waterport_session_stats.pkl")
    
def plot_development(stats_df):
    # 1) Accuracy & alternations
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(stats_df['PctCorrect'], 'o-', label='% Correct', color='tab:blue')
    ax1.set_ylabel('% Correct')
    ax1.set_ylim(0, 100)

    ax1.set_xticks(stats_df.index)
    ax1.set_xticklabels(stats_df['Session'], rotation=45)
    ax1.set_title('Performance over sessions')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.92))
    plt.show()

    # 2) Bias plot
    plt.figure(figsize=(8, 4))
    plt.plot(stats_df['BiasRight_FP'] * 100, 's-', color='purple', label='Right ward bias')
    plt.axhline(50, ls='--', color='gray', alpha=.6)
    plt.ylabel('Right side trial proportion (%)')
    plt.xticks(stats_df.index, stats_df['Session'], rotation=45)
    plt.ylim(0, 100)
    plt.title('Right / Left bias over sessions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

def plot_group_summary(stats_df):
    # Convert session to datetime
    stats_df["SessionDate"] = pd.to_datetime(stats_df["Session"], format="%y%m%d", errors="coerce")

    # Filter out weekends or irrelevant entries
    excluded_dates = [pd.Timestamp("2025-04-26"), pd.Timestamp("2025-04-27")]
    df_filtered = stats_df[
        (~stats_df["SessionDate"].isin(excluded_dates)) &
        (stats_df["PctCorrect"].notna())
    ]

    # Create a categorical string label for x-axis to avoid time-based spacing
    df_filtered["SessionLabel"] = df_filtered["SessionDate"].dt.strftime("%Y-%m-%d")
    df_sorted = df_filtered.sort_values(["SessionDate", "Mouse"])

    # Plot
    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=df_sorted,
        x="SessionLabel",  # categorical x-axis
        y="PctCorrect",
        hue="Mouse",
        marker="o",
        palette="tab10",
        linewidth=1.5
    )
    plt.title("Correct First Decisions Over Time")
    plt.xlabel("Session Date")
    plt.ylabel("% Correct")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.legend(title="Mouse ID")
    plt.show()
    
def plot_plots_group_summary(stats_df):
    # Convert session to datetime
    stats_df["SessionDate"] = pd.to_datetime(stats_df["Session"], format="%y%m%d", errors="coerce")

    # Filter out weekends or irrelevant entries
    excluded_dates = [pd.Timestamp("2025-04-26"), pd.Timestamp("2025-04-27")]
    df_filtered = stats_df[
        (~stats_df["SessionDate"].isin(excluded_dates)) &
        (stats_df["Trials"].notna())
    ]

    # Create a categorical string label for x-axis to avoid time-based spacing
    df_filtered["SessionLabel"] = df_filtered["SessionDate"].dt.strftime("%Y-%m-%d")
    df_sorted = df_filtered.sort_values(["SessionDate", "Mouse"])

    # Plot
    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=df_sorted,
        x="SessionLabel",  # categorical x-axis
        y="Trials",
        hue="Mouse",
        errorbar=("se", 2),
        marker="o",
        palette="tab10",
        linewidth=1.5
    )
        
    mean_per_session = df_sorted.groupby("SessionLabel")["Trials"].mean().reset_index()
    sns.lineplot(
        data=mean_per_session,
        x="SessionLabel",
        y="Trials",
        color="black",
        linestyle="--",
        linewidth=2,
        label="Session Mean"
    )
    plt.title("Initiated trials")
    plt.xlabel("Session Date")
    plt.ylabel("Trials")
    plt.xticks(rotation=45)
    plt.ylim(0, 170)
    plt.tight_layout()
    plt.legend(title="Mouse ID")
    plt.show()
    
stats_df = pd.read_pickle('/Volumes/Expansion/New_cleaned_data/statsdf/allmice/waterport_session_stats.pkl')

stats_df['TotalCorrect'] = (
    ((stats_df['PctCorrect'] / 100) * stats_df['Trials'])
    .round()
    .where(stats_df['PctCorrect'].notna() & stats_df['Trials'].notna())
    .astype('Int64')  # nullable integer type
)
print(stats_df['TotalCorrect'])
#stat, p = friedmanchisquare(stats_df['TotalCorrect'])
#print(f"Friedman χ² = {stat:.3f}, p = {p:.4f}")

def plot_correct_group_summary(stats_df):
    # Convert session to datetime
    stats_df["SessionDate"] = pd.to_datetime(stats_df["Session"], format="%y%m%d", errors="coerce")

    # Filter out weekends or irrelevant entries
    excluded_dates = [pd.Timestamp("2025-04-26"), pd.Timestamp("2025-04-27")]
    df_filtered = stats_df[
        (~stats_df["SessionDate"].isin(excluded_dates)) &
        (stats_df["TotalCorrect"].notna())
    ]

    df_filtered["SessionLabel"] = df_filtered["SessionDate"].dt.strftime("%Y-%m-%d")
    df_sorted = df_filtered.sort_values(["SessionDate", "Mouse"])

    # Plot
    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=df_sorted,
        x="SessionLabel",  # date as x-axis 
        y="TotalCorrect",
        hue="Mouse",
        errorbar=("se", 2),
        marker="o",
        palette="tab10",
        linewidth=1.5
    )
    mean_per_session2 = df_sorted.groupby("SessionLabel")["TotalCorrect"].mean().reset_index()
    sns.lineplot(
        data=mean_per_session2,
        x="SessionLabel",
        y="TotalCorrect",
        color="black",
        linestyle="--",
        linewidth=2,
        label="Session Mean"
        )
    plt.title("Correct trials")
    plt.xlabel("Session Date")
    plt.ylabel("Correct Trials")
    plt.xticks(rotation=45)
    plt.ylim(0, 170)
    plt.tight_layout()
    plt.legend(title="Mouse ID")
    plt.show()

#plot_plots_group_summary(stats_df)   
#plot_correct_group_summary(stats_df)  
#graph = plot_correct_group_summary(stats_df)       
base_path = ''
run_waterport_decision_analysis(
        base_path=Path(base_path),
        low=(158.0, 270.0),
        high=(1055.9, 270.1),
        radius=50.0,
        bg_img=Path("/Volumes/Expansion/New_cleaned_data/box.jpg"), # put the related background image in here
        out_dir=Path("/Volumes/Expansion/decisionchange") # select output path 
        )
