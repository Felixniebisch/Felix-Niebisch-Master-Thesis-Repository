import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

from ExplorationAnalyzer import ExplorationAnalyzer



class AnalyzeExploration:
    def __init__(self, session_path, background_image_path=None):
        self.session_path = session_path
        self.background_image_path = background_image_path
        
        self.squaresvisited = []
        self.squaresvisitedcount = 0
        self.squaresnotvisited = []
        self.squaresnotvisitedcount = 0

    def makegrids(self, session_path=None):
        if session_path is None:
            session_path = self.session_path

        df = pd.read_csv(session_path, header=[1, 2])

        if len(df) < 5000: # input for the required size of grid
            print(f"Skipping {session_path}: insufficient data ({len(df)} frames).")
            return None  

        valid_rows = df[("Nose", "likelihood")] > 0.01
        df_filtered = df.loc[valid_rows, ("Nose", ["x", "y"])].iloc[:50000]
        df_x, df_y = df_filtered[("Nose", "x")], df_filtered[("Nose", "y")]

        width = int(df_x.max()) + 1
        height = int(df_y.max()) + 1

        grid = ExplorationAnalyzer(width=width, height=height)
        #coarse_grid = ExplorationAnalyzerTemporal(width=width, height=height)
        #coarse_grid.update_chunked_grid(df_x.values, df_y.values)

        for x, y in zip(df_x, df_y):
            grid.update_grid(x, y)
            grid.get_frequency(x, y)

        self._plot_with_background(grid.heatgrid, width, height, "Visit Frequency Heatmap (nose)", cmap='afmhot', norm=LogNorm(vmin=1, vmax=np.max(grid.heatgrid)))
        # plot the mouse name with it
        
        #variance_map = coarse_grid.compute_normalized_variance_map()
        #self._plot_with_background(variance_map, width, height, "Exploratory Behavior Variance Heatmap", cmap='viridis')
        #coarse_grid.compute_distribution_variances_over_time()
        #coarse_grid.plot_variance_over_time()
        #coarse_grid.show_variance_heatmap()
        binary_grid = grid.get_grid()
        binary_grid_df = pd.DataFrame(binary_grid)

        return binary_grid_df 

    def _plot_with_background(self, data, width, height, title, cmap, norm=None):
        fig, ax = plt.subplots(figsize=(8, 6))  
    
        if self.background_image_path:
            bg_img = Image.open(self.background_image_path).resize((width, height), Image.Resampling.LANCZOS)
            ax.imshow(bg_img, extent=[0, width, height, 0])
    
        im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest", origin="upper", extent=[0, width, height, 0], alpha=0.7)
        fig.colorbar(im, ax=ax, label=title)
    
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        fig.tight_layout()
        plt.show()
        plt.close(fig) 

    def countvisited(self, binary_grid):
        if binary_grid is None:
            print("No valid grid provided.")
            return [], 0

        self.squaresvisited = list(zip(*((binary_grid == 1).to_numpy().nonzero())))
        self.squaresvisitedcount = len(self.squaresvisited)

        return self.squaresvisited, self.squaresvisitedcount

    def countnotvisited(self, binary_grid):
        if binary_grid is None:
            print("No valid grid provided.")
            return [], 0

        self.squaresnotvisited = list(zip(*((binary_grid == 0).to_numpy().nonzero())))
        self.squaresnotvisitedcount = len(self.squaresnotvisited)

        return self.squaresnotvisited, self.squaresnotvisitedcount
