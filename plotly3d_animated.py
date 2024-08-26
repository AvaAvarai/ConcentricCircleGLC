import pandas as pd
import numpy as np
import plotly.graph_objs as go
import tkinter as tk
from tkinter import filedialog, messagebox
import colorsys
from typing import List, Optional

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

def generate_hue_shifted_colors(num_classes: int, num_attributes: int) -> List[List[str]]:
    """
    Generates a list of distinct RGB hex colors with a hue shift from the innermost to the outermost sphere.
    
    Args:
        num_classes (int): The number of distinct colors to generate.
        num_attributes (int): The number of attributes (spheres) to generate a hue shift across.
    
    Returns:
        List[List[str]]: A list of lists where each sublist contains hex color strings for a specific hue shift.
    """
    base_hues = np.linspace(0, 1, num_classes, endpoint=False)
    color_sets = []

    for i in range(num_attributes):
        shift_amount = i / num_attributes
        shifted_colors = [colorsys.hsv_to_rgb((h + shift_amount) % 1, 0.7, 0.9) for h in base_hues]
        hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in shifted_colors]
        color_sets.append(hex_colors)

    return color_sets

def distribute_points_on_sphere(n: int) -> np.ndarray:
    """
    Distributes `n` points evenly on the surface of a unit sphere using the Fibonacci lattice.
    
    Args:
        n (int): The number of points to distribute.
    
    Returns:
        np.ndarray: Array of shape (n, 3) containing the (x, y, z) coordinates of points.
    """
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.vstack([x, y, z]).T

def plot_3d_spheres(data: pd.DataFrame, class_column: str = 'class', grid_size=100) -> None:
    """
    Plots data points on expanding 3D spheres using Plotly, coloring them based on their class labels with hue shift.
    
    Args:
        data (pd.DataFrame): The input data with features and a class label column.
        class_column (str): The name of the column containing class labels.
        grid_size (int): The number of points in the spherical lattice.
    """
    data.columns = map(str.lower, data.columns)
    class_column = class_column.lower()
    if class_column not in data.columns:
        raise ValueError(f"Column '{class_column}' not found in the CSV file.")
    
    features = data.drop(columns=[class_column])
    classes = data[class_column]

    mins = features.min()
    maxs = features.max()
    normalized_data = (features - mins) / (maxs - mins)

    num_attributes = len(features.columns)
    unique_classes = classes.unique()
    num_classes = len(unique_classes)
    color_sets = generate_hue_shifted_colors(num_classes, num_attributes)

    radii = np.linspace(1, num_attributes, num_attributes)

    frames = []

    # Create frames for each sphere
    for frame_idx, radius in enumerate(radii):
        scatter_data = []

        # Group by class and create one trace per class
        for cls_idx, cls in enumerate(unique_classes):
            cls_data = normalized_data[classes == cls]
            x_points = []
            y_points = []
            z_points = []
            for _, row in cls_data.iterrows():
                for j in range(frame_idx + 1):  # Only include up to the current sphere
                    point_idx = int(row.iloc[j] * (grid_size - 1))  # Map normalized value to a point on the lattice
                    x, y, z = distribute_points_on_sphere(grid_size)[point_idx] * radii[j]
                    x_points.append(x)
                    y_points.append(y)
                    z_points.append(z)

            scatter_data.append(go.Scatter3d(
                x=x_points, y=y_points, z=z_points,
                mode='markers+lines',
                marker=dict(size=4, color=color_sets[j][cls_idx], opacity=0.8),
                line=dict(color=color_sets[j][cls_idx], width=2),
                name=f'Class {cls}'
            ))

        frames.append(go.Frame(data=scatter_data, name=str(frame_idx)))

    # Initial plot setup with the first frame
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title="3D Expanding Spheres Plot with Spherical Lattice and Hue Shift",
            scene=dict(
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis'),
                zaxis=dict(title='Z Axis'),
                aspectmode='cube'
            ),
            legend=dict(x=0.7, y=0.9),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[dict(type="buttons", showactive=False,
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, dict(frame=dict(duration=500, redraw=True), 
                                                             fromcurrent=True)])])]
        ),
        frames=frames
    )

    fig.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_3d_spheres(data)
