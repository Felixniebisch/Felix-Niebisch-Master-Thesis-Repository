import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter
import pandas as pd
import os

from PIL import Image

def draw_smoothed_heatmaps_with_background(base_path, background_image_path, output_folder=None):
    def find_top_folders(base_path):
        top_folders = []
        for root, dirs, _ in os.walk(base_path):
            if "top" in dirs:
                top_folders.append(os.path.join(root, "top"))
        return top_folders

    top_folders = find_top_folders(base_path)

    for folder in top_folders:
        for file in os.listdir(folder):
            if file.endswith("filtered.csv") and not file.startswith("._"):
                csv_path = os.path.join(folder, file)
                print(f"Processing: {csv_path}")

                try:
                    df = pd.read_csv(csv_path, header=[1, 2])
                    if len(df) < 5000:
                        continue

                    valid = df[("Nose", "likelihood")] > 0.01
                    df = df.loc[valid, ("Nose", ["x", "y"])].iloc[:50000]
                    x = df[("Nose", "x")].astype(int)
                    y = df[("Nose", "y")].astype(int)
                    w, h = int(x.max()) + 1, int(y.max()) + 1

                    heatgrid = np.zeros((h, w))
                    for xi, yi in zip(x, y):
                        heatgrid[yi, xi] += 1

                    heatgrid_smooth = gaussian_filter(heatgrid, sigma=40)

                    fig, ax = plt.subplots(figsize=(6, 6))

                    # Load and display background image
                    bg_img = Image.open(background_image_path).resize((w, h), Image.Resampling.LANCZOS)
                    ax.imshow(bg_img, extent=[0, w, h, 0])

                    # Overlay the heatmap
                    im = ax.imshow(
                        heatgrid_smooth,
                        cmap='inferno',
                        interpolation='bilinear',
                        origin='upper',
                        norm=PowerNorm(gamma=0.5),
                        alpha=0.6
                    )

                    # Optional circular overlays
                    circle_left = Circle((158, 270), 30, linestyle='--', edgecolor='red', facecolor='none', linewidth=2)
                    circle_right = Circle((1055.9, 270), 30, linestyle='--', edgecolor='yellow', facecolor='none', linewidth=2)
                    ax.add_patch(circle_left)
                    ax.add_patch(circle_right)

                    ax.set_title(f"{file[:12]}")
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if output_folder:
                        os.makedirs(output_folder, exist_ok=True)
                        save_path = os.path.join(output_folder, f"{file[:12]}_heatmap_with_bg.png")
                        plt.savefig(save_path, dpi=200)
                        plt.close(fig)
                    else:
                        plt.show()

                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")

# Run with background image
draw_smoothed_heatmaps_with_background(
    base_path="/Volumes/Expansion/Usable_data copy/M1797/",
    background_image_path="/Volumes/Expansion/New_cleaned_data/box.jpg",
)