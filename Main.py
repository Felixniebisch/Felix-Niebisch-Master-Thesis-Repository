import sys
import os
sys.path.append('/Users/Master_thesis_30_ECTS/behavior-box-repo_copy/behavior_box_analysis/Felix_scripts/') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from AnalyzeExploration import AnalyzeExploration
from ExplorationAnalyzer import ExplorationAnalyzerTemporal
from HoldingTimeAnalysis import HoldingTimeAnalysis
from Plotpretrajectories import PretrialTrajectoryAnalyzer
from WaterPortDecisionAnalyzer import WaterPortDecisionAnalyzer
from pathlib import Path

base_path = ''  #input point of the data

'''
assumes a folder structure that contains: 
    - session wise data
    - sensor data + top-folder
    - top-folder containing filtered csv's
    - something like this: /Volumes/Expansion/Mouse2024006/Mouse2024006-240624-100847/top/MO406-240624-0936-topDLC_resnet50_ObjectInteractionOct12shuffle1_200000_filtered.csv
'''


def find_top_folders(base_path):
    top_folders = []
    for root, dirs, _ in os.walk(base_path):
        if "top" in dirs:
            top_folders.append(os.path.join(root, "top"))
    return top_folders


def process_and_plot_variance(base_path):
    variances = []
    file_names = []

    top_folders = find_top_folders(base_path)

    for folder in top_folders:
        print(f"Processing folder: {folder}")

        for file in os.listdir(folder):
            if file.startswith("._") or not file.endswith("filtered.csv"):
                continue

            csv_path = os.path.join(folder, file)
            print(f"Processing CSV: {csv_path}")

            exploration = AnalyzeExploration(csv_path, '') # add path of the background image here
            binary_grid = exploration.makegrids(csv_path)

            if binary_grid is not None:
                coords = np.argwhere(binary_grid.to_numpy() > 0)
                if coords.size == 0:
                    print(f"No visits recorded in {csv_path}")
                    continue

                width, height = binary_grid.shape[::-1]
                temporal_analyzer = ExplorationAnalyzerTemporal(width, height, grid_size=10)
                temporal_analyzer.update_grid(coords[:, 1], coords[:, 0])
                session_variance = temporal_analyzer.compute_spatial_distribution_variance()

                if session_variance is not None:
                    variances.append(session_variance)
                    file_names.append(file[6:12])  # or adapt if this slicing is date-specific
                    print(f"Variance for {file[:12]}: {session_variance}")

    if not variances:
        print("No variance data collected.")
        return


    sorted_indices = np.argsort(file_names)
    file_names = [file_names[i] for i in sorted_indices]
    variances = [variances[i] for i in sorted_indices]

    # Stats
    variances_array = np.array(variances)
    mean_variance = np.mean(variances_array)
    std_variance = np.std(variances_array)

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.plot(file_names, variances, linestyle='-', linewidth=2, color='#1f77b4', label='Session Variance')
    plt.fill_between(file_names,
                     mean_variance - std_variance,
                     mean_variance + std_variance,
                     color='#1f77b4',
                     alpha=0.2,
                     label='Mean ± 1 Std Dev')
    plt.axhline(mean_variance, color='gray', linestyle='--', linewidth=1.5, label='Overall Mean')

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Session Date", fontsize=12)
    plt.ylabel("Spatial Variance", fontsize=12)
    plt.title("Spatial Distribution Variance Across Sessions", fontsize=14)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.show()


def process_visitedsquares_csvs_in_top_folders(base_path):
    visited_counts = []
    visited_counts_corr = []
    file_names = []
    top_folders = find_top_folders(base_path)

    for folder in top_folders:
        print(f"Processing folder: {folder}")

        for file in os.listdir(folder):
            if file.startswith("._"):
                continue
            if file.endswith("filtered.csv"):
                csv_path = os.path.join(folder, file)

                print(f"Processing CSV: {csv_path}")

                exploration = AnalyzeExploration(csv_path, '/Volumes/Expansion/Felix/box.png')
                binary_grid = exploration.makegrids(csv_path)
                exploration.countvisited(binary_grid)
                exploration.countnotvisited(binary_grid)

                if exploration.squaresvisitedcount > 0:
                    visited_counts.append(exploration.squaresvisitedcount)

                    file = file[:12]
                    file_names.append(file[6:])

                    print(f"Visited squares: {exploration.squaresvisitedcount}")
                    print(f"Not visited squares: {exploration.squaresnotvisitedcount}")
                else:
                    print(f"Not sufficient data in {csv_path}")


    plt.figure(figsize=(10, 5))
    plt.plot(file_names, visited_counts, marker="o", linestyle="-", color="b", label="Visited Squares")
    plt.xticks(rotation=45, ha="right")  # Rotate file names for better visibility
    plt.xlabel("Dates")
    plt.ylabel("Visited Squares Count")
    plt.title("Visited Squares Per Day")
    plt.legend()
    plt.grid(False)
    plt.show()


def process_sensor_data_in_parent_folders(base_path):
    analyzer = HoldingTimeAnalysis(threshold=30) # threshold needed to hold in order to initiate a trial 
    analyzer.process_all_subfolders(base_path)
 
def Plotpretrialanalysis(base_path, top_img_path, dlc_fps):
    # Initialize the analyzer
    preanalyzer = PretrialTrajectoryAnalyzer(
        base_path=base_path,
        top_img_path=top_img_path,
        dlc_fps=dlc_fps
    )
    
    # Run the full pretrial analysis
    preanalyzer.process_all_sessions()                


def run_waterport_decision_analysis(base_path, low, high, radius, bg_img, out_dir):
    analyzer = WaterPortDecisionAnalyzer(low, high, radius)

    session_stats = []
    summary_frames = []

    # Helper functions
    def find_top_folders(base_path):
        return [Path(r) / 'top' for r, ds, _ in os.walk(base_path) if 'top' in ds]

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

    if summary_frames: # optional, save the whole df as csv 
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.concat(summary_frames, ignore_index=True).to_csv(out_dir / 'waterport_decision_summary.csv', index=False)
        stats_df = pd.DataFrame(session_stats)
        stats_df.to_csv(out_dir / 'waterport_session_stats.csv', index=False)
        print('✔ Saved summary and stats to', out_dir)

        # ---- development plots ----
        plot_development(stats_df)
    else:
        print('No valid sessions processed.')

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
    
def run_full_analysis(base_path):
    process_and_plot_variance(base_path)
    process_visitedsquares_csvs_in_top_folders(base_path)
    process_sensor_data_in_parent_folders(base_path)
    Plotpretrialanalysis(base_path, "/Volumes/Expansion/Felix/box.png", 25)

    run_waterport_decision_analysis(
        base_path=Path(base_path),
        low=(145.0, 230.0),
        high=(1050.0, 260.0),
        radius=25.0,
        bg_img=Path(""), # add background image
        out_dir=Path("") # add optional output path for waterportdecision class
    )
    
    
    
if __name__ == "__main__":
    run_full_analysis(base_path)