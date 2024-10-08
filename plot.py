import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional
import colorsys

def load_csv() -> Optional[pd.DataFrame]:
    """
    Prompts the user to select a CSV file and loads it into a pandas DataFrame.
    Returns None if loading fails or no file is selected.
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return None

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the CSV file: {str(e)}")
        return None

def generate_distinct_colors(num_classes: int) -> List[Tuple[float, float, float]]:
    """
    Generates a list of distinct RGB colors by subdividing the HSV hue space.
    
    Args:
        num_classes (int): The number of distinct colors to generate.
    
    Returns:
        List[Tuple[float, float, float]]: A list of RGB color tuples.
    """
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    colors = [colorsys.hsv_to_rgb(h, 0.7, 0.9) for h in hues]
    return colors

def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the feature columns of the DataFrame to the range [0, 1].
    
    Args:
        features (pd.DataFrame): The feature columns.
    
    Returns:
        pd.DataFrame: Normalized feature columns.
    """
    mins = features.min()
    maxs = features.max()
    return (features - mins) / (maxs - mins)

def compute_points(row: pd.Series, radii: np.ndarray) -> np.ndarray:
    """
    Computes the (x, y) coordinates for a data point based on its normalized feature values.
    
    Args:
        row (pd.Series): A row of normalized feature values.
        radii (np.ndarray): An array of radii for the concentric circles.
    
    Returns:
        np.ndarray: An array of (x, y) coordinates.
    """
    points = []
    for j, radius in enumerate(radii):
        angle = -2 * np.pi * row.iloc[j] + np.pi / 2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append((x, y))
    return np.array(points)

def plot_concentric_circles(data: pd.DataFrame, class_column: str = 'class') -> None:
    """
    Plots data points on concentric circles, coloring them based on their class labels.
    Allows zooming and panning through the dimensions while keeping the concentric circles equidistant.
    
    Args:
        data (pd.DataFrame): The input data with features and a class label column.
        class_column (str): The name of the column containing class labels.
    """
    data.columns = map(str.lower, data.columns)
    class_column = class_column.lower()
    if class_column not in data.columns:
        raise ValueError(f"Column '{class_column}' not found in the CSV file.")
    
    features = data.drop(columns=[class_column])
    classes = data[class_column]

    normalized_data = normalize_features(features)

    num_attributes = len(features.columns)
    unique_classes = classes.unique()
    num_classes = len(unique_classes)
    colors = generate_distinct_colors(num_classes)

    radii = np.linspace(1, num_attributes, num_attributes)  # Create a circle for each attribute

    class_to_color = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='lightgrey')

    plt.subplots_adjust(top=0.99, bottom=0.15)

    def update_plot(zoom_factor=1.0, x_pan=0.0, y_pan=0.0):
        """
        Updates the plot based on the current zoom factor and pan positions.
        
        Args:
            zoom_factor (float): Factor by which to zoom the plot.
            x_pan (float): Horizontal pan adjustment.
            y_pan (float): Vertical pan adjustment.
        """
        ax.clear()

        for idx, row in normalized_data.iterrows():
            points = compute_points(row, radii)
            cls = classes.iloc[idx]
            color = class_to_color[cls]
            
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.3, zorder=1, label=f'{cls}')
            ax.plot(points[:, 0], points[:, 1], 'o', color=color, alpha=0.6, zorder=1)

        for radius in radii:
            circle = plt.Circle((0, 0), radius, color='black', alpha=0.6, fill=False)
            ax.add_artist(circle)

        for j, radius in enumerate(radii):
            angle = np.pi / 2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            ax.text(x, y, features.columns[j], horizontalalignment='center', verticalalignment='center')

        ax.set_xlim((-5 + x_pan) * zoom_factor, (5 + x_pan) * zoom_factor)
        ax.set_ylim((-5 + y_pan) * zoom_factor, (5 + y_pan) * zoom_factor)
        ax.set_aspect('equal')
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        
        fig.suptitle("Concentric Circles Plot", y=0.95, fontsize=16)  # Move title to the very top
        plt.axis('off')
        fig.canvas.draw_idle()

    # Set up zoom and pan sliders
    zoom_ax = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgrey')
    zoom_slider = Slider(zoom_ax, 'Zoom', 0.5, 10.0, valinit=1.0)

    hpan_ax = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgrey')
    hpan_slider = Slider(hpan_ax, 'H Pan', -num_attributes, num_attributes, valinit=0.0)

    vpan_ax = plt.axes([0.2, 0.09, 0.65, 0.03], facecolor='lightgrey')
    vpan_slider = Slider(vpan_ax, 'V Pan', -num_attributes, num_attributes, valinit=0.0)

    zoom_slider.on_changed(lambda val: update_plot(zoom_factor=val, x_pan=hpan_slider.val, y_pan=vpan_slider.val))
    hpan_slider.on_changed(lambda val: update_plot(zoom_factor=zoom_slider.val, x_pan=val, y_pan=vpan_slider.val))
    vpan_slider.on_changed(lambda val: update_plot(zoom_factor=zoom_slider.val, x_pan=hpan_slider.val, y_pan=val))

    update_plot()

    plt.show()

def main() -> None:
    """
    Main function to load data and plot the concentric circles.
    """
    data = load_csv()
    if data is not None:
        plot_concentric_circles(data)

if __name__ == "__main__":
    main()
