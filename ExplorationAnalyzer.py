import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd 
from MiceLearningCurveMultipleSessions import MiceLearningCurveMultipleSessions
from PIL import Image
import pylab as pyl
import numpy as np
from CSVDataReader import CSVDataReader
from BehaviorCSVReader import BehaviorCSVReader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#self, path, file_behavior, session_path, mouse, img,

class ExplorationAnalyzer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Create a grid with one cell per pixel
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.heatgrid = np.zeros((self.height, self.width), dtype=int)
        #initialize a lists of visited and not visited squares
        self.visited_squares = []
        self.not_visited_squares = []


    def map_to_grid(self, x, y):
        """
        Direct pixel-to-grid mapping.
        """
        row = int(y)  # y → row
        col = int(x)  # x → col
    
        # Clamp to image bounds just in case
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)

        return row, col

    def update_grid(self, x, y):
        """
        Marks the cell corresponding to (x, y) as visited.

        """
        row, col = self.map_to_grid(x, y)
        self.grid[row, col] = 1  # Binary presence
        
    def get_frequency(self, x, y):
        """ marks the frequncy with which places have been visited by incrementing the visit counter by 1 for each visit"""
        
        row, col = self.map_to_grid(x, y)
        self.heatgrid[row, col] += 1  # Binary presence
        #print(f"Updated heatgrid[{row}, {col}] = {self.heatgrid[row, col]}")
   
    def show_frequency_heatmap(self):
        print("max value in heatgrid:", np.max(self.heatgrid))
        print("nonzero values:", np.count_nonzero(self.heatgrid))
    
        if np.max(self.heatgrid) == 0:
            print("Heatgrid is empty — nothing to display.")
            return
    
        plt.figure(figsize=(12, 10), dpi=120)
        plt.imshow(
            self.heatgrid,
            cmap='afmhot',
            norm=LogNorm(vmin=1, vmax=np.max(self.heatgrid)),
            interpolation="none",  # 👈 show each pixel clearly
            origin="upper"
        )
        plt.colorbar(label="Visit Frequency")
        plt.title("Mouse Visit Frequency Heatmap for the nose")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.tight_layout()
        plt.show()

    def get_grid(self):
        

        return self.grid
    
    def show_smoothed_heatmap(data):
        fig, ax = plt.subplots(figsize=(6, 6))
    
        # Display heatmap with smooth interpolation
        im = ax.imshow(
            data,
            cmap='inferno',  # or 'magma', 'plasma', etc.
            interpolation='bilinear',  # smooth look
            origin='upper'
        )
    
        # Add a circular overlay (example: two dashed circles)
        circle1 = plt.Circle((100, 100), 20, color='white', linestyle='--', fill=False, linewidth=2)
        circle2 = plt.Circle((200, 100), 20, color='white', linestyle='--', fill=False, linewidth=2)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
    
        fig.colorbar(im, ax=ax, label="Visit Intensity")
        ax.set_title("Smoothed Mouse Visit Heatmap")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

class ExplorationAnalyzerTemporal:
    def __init__(self, width2, height2, grid_size=10):
        self.width2 = width2
        self.height2 = height2
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

    def _map_to_coarse_grid(self, x, y):
        col_bin = int((x / self.width2) * self.grid_size)
        row_bin = int((y / self.height2) * self.grid_size)
        col_bin = min(max(col_bin, 0), self.grid_size - 1)
        row_bin = min(max(row_bin, 0), self.grid_size - 1)
        return row_bin, col_bin

    def update_grid(self, x_coords, y_coords):
        for x, y in zip(x_coords, y_coords):
            row_bin, col_bin = self._map_to_coarse_grid(x, y)
            self.grid[row_bin, col_bin] += 1

    def compute_spatial_distribution_variance(self):
        total_visits = np.sum(self.grid)
        if total_visits == 0:
            print("No visits recorded, variance undefined.")
            return None

        normalized_distribution = self.grid.flatten() / total_visits
        variance = np.var(normalized_distribution)
        return variance
    
    def plot_visit_distribution(self):
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(self.grid.flatten())), self.grid.flatten())
        plt.title("Visit Distribution Across Grid Cells")
        plt.xlabel("Grid Cells")
        plt.ylabel("Number of Visits")
        plt.tight_layout()
        plt.show()
        plt.close()



