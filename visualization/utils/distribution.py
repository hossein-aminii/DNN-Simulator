import numpy as np
import matplotlib.pyplot as plt


class DistributionVisualizer:
    def __init__(
        self,
        weights: np.ndarray,
        title: str = "Weight Distribution",
        xlabel: str = "Values",
        ylabel: str = "Frequency",
        color: str = "blue",
        edge_color: str = "black",
        bins: int = 50,
    ):
        """
        Initializes the visualizer with LSTM weights.

        :param weights: NumPy array of LSTM weights.
        """
        if not isinstance(weights, np.ndarray):
            raise ValueError("Weights must be a NumPy array")
        self.weights = weights

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color = color
        self.edge_color = edge_color
        self.bins = bins

    def plot_weight_distribution(self):
        """
        Plots the distribution of the LSTM weights.
        """
        # Flatten the weights array
        flattened_weights = self.weights.flatten()

        # Create a histogram of the weights
        plt.figure(figsize=(10, 6))
        plt.hist(flattened_weights, bins=self.bins, color=self.color, edgecolor=self.edge_color)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(True)
        plt.show()
