import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    root.withdraw()  # Hide the main tkinter window

    # Ask the user to select a CSV file
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return None

    # Load the CSV file
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
    
    Args:
        data (pd.DataFrame): The input data with features and a class label column.
        class_column (str): The name of the column containing class labels.
    """
    # Ensure the class column is treated in a case-insensitive manner
    data.columns = map(str.lower, data.columns)
    class_column = class_column.lower()
    if class_column not in data.columns:
        raise ValueError(f"Column '{class_column}' not found in the CSV file.")
    
    # Separate the features and the class labels
    features = data.drop(columns=[class_column])
    classes = data[class_column]

    # Normalize the data
    normalized_data = normalize_features(features)

    # Determine the number of attributes and classes
    num_attributes = len(features.columns)
    unique_classes = classes.unique()
    num_classes = len(unique_classes)
    colors = generate_distinct_colors(num_classes)

    # Setup radii for concentric circles
    radii = np.linspace(1, 4, num_attributes)

    # Create a mapping from class label to color
    class_to_color = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='lightgrey')

    for idx, row in normalized_data.iterrows():
        points = compute_points(row, radii)
        cls = classes.iloc[idx]
        color = class_to_color[cls]
        
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.3, zorder=1)
        ax.plot(points[:, 0], points[:, 1], 'o', color=color, alpha=0.6, zorder=1)

    # Draw the concentric circles
    for radius in radii:
        circle = plt.Circle((0, 0), radius, color='black', alpha=0.6, fill=False)
        ax.add_artist(circle)

    # Add labels for each attribute on the circles
    for j, radius in enumerate(radii):
        angle = np.pi / 2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ax.text(x, y, features.columns[j], horizontalalignment='center', verticalalignment='center')

    # Add legend
    for cls, color in class_to_color.items():
        ax.plot([], [], color=color, label=cls, linewidth=5)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    plt.legend(loc='upper right')
    plt.title("Concentric Circles Plot", pad=20)
    plt.axis('off')
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
