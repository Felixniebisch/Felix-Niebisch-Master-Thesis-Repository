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
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
from BehaviorCSVReader import BehaviorCSVReader  
from CSVDataReader     import CSVDataReader      

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
            self._plot_counts(stats, mouse, sess)

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
            ax.imshow(img)
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
        ax.set_title(f"Trajectories — Mouse 00{mouse}, Session {session}")
        ax.legend(loc="upper right")
        plt.tight_layout(); plt.show()

    @staticmethod
    def _plot_counts(stats: Dict[str, float], mouse: str, session: str):
        bias = stats["BiasRight_FP"]
        bias_txt = f"Right ward bias = {bias*100:.1f}%" if not np.isnan(bias) else "Right ward bias = n/a"

        # % correct per side
        plt.figure(figsize=(4, 4))
        plt.bar(["Corr%L", "Corr%R"], [stats["Correct_L"], stats["Correct_R"]],
                color="green")
        plt.ylim(0, 100); plt.ylabel("%")
        plt.title(f"% correct per side — Mouse 00{mouse}, Sess {session}\n{bias_txt}")
        plt.tight_layout(); plt.show()

        # raw incorrect & alternations
        plt.figure(figsize=(4, 4))
        plt.bar(["Inc-L", "Inc-R", "Alt"],
                [stats["Incorrect_L"], stats["Incorrect_R"], stats["TotalAlt"]],
                color=["red", "red", "steelblue"])
        ymax = max(stats["Incorrect_L"], stats["Incorrect_R"], stats["TotalAlt"]) * 1.1 + 1
        plt.ylim(0, ymax); plt.ylabel("Count")
        plt.title(f"Errors & alternations — Mouse 00{mouse}, Sess {session}")
        plt.tight_layout(); plt.show()