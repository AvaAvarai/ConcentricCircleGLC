import pandas as pd
import numpy as np
import plotly.graph_objs as go
import tkinter as tk
from tkinter import filedialog, messagebox
import colorsys
from typing import List, Optional, Dict

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

def generate_distinct_colors(num_classes: int) -> List[str]:
    """
    Generates a list of distinct RGB hex colors by subdividing the HSV hue space.
    
    Args:
        num_classes (int): The number of distinct colors to generate.
    
    Returns:
        List[str]: A list of hex color strings.
    """
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    colors = [colorsys.hsv_to_rgb(h, 0.7, 0.9) for h in hues]
    return ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in colors]

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

def compute_class_distribution_histogram(class_counts: Dict[str, int], total: int, class_totals: Dict[str, int]) -> str:
    """
    Computes the class distribution relative to the total count at a point and returns it as a string.
    
    Args:
        class_counts (Dict[str, int]): Dictionary with class counts at the point.
        total (int): Total number of cases passing through the point.
        class_totals (Dict[str, int]): Total count for each class.
    
    Returns:
        str: The class distribution histogram represented as a string.
    """
    histogram = '\n'.join(
        [f'{cls}: {count} ({count / total:.2%} of point, {count / class_totals[cls]:.2%} of class)' 
         for cls, count in class_counts.items()]
    )
    return f'Class Distribution:\n{histogram}'

def plot_3d_spheres(data: pd.DataFrame, class_column: str = 'class', grid_size=100) -> None:
    """
    Plots data points on expanding 3D spheres using Plotly, coloring them based on their class labels.
    
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
    colors = generate_distinct_colors(num_classes)

    class_to_color = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

    radii = np.linspace(1, num_attributes, num_attributes)

    scatter_data = []

    # Track the points and the number of cases passing through each lattice point
    point_class_distribution = {}
    class_totals = classes.value_counts().to_dict()

    # Group by class and create one trace per class
    for cls in unique_classes:
        cls_data = normalized_data[classes == cls]
        x_points = []
        y_points = []
        z_points = []
        hover_texts = []
        for _, row in cls_data.iterrows():
            for j, radius in enumerate(radii):
                # Use the Fibonacci lattice to place data points on the sphere
                point_idx = int(row.iloc[j] * (grid_size - 1))  # Map normalized value to a point on the lattice
                x, y, z = distribute_points_on_sphere(grid_size)[point_idx] * radius
                point = (x, y, z)

                # Track the class distribution for the lattice point
                if point not in point_class_distribution:
                    point_class_distribution[point] = {cls: 1}
                else:
                    if cls in point_class_distribution[point]:
                        point_class_distribution[point][cls] += 1
                    else:
                        point_class_distribution[point][cls] = 1

                x_points.append(x)
                y_points.append(y)
                z_points.append(z)

        # Compute hover text for each point
        for point in zip(x_points, y_points, z_points):
            total_cases = sum(point_class_distribution[point].values())
            histogram = compute_class_distribution_histogram(point_class_distribution[point], total_cases, class_totals)
            hover_texts.append(histogram)

        scatter_data.append(go.Scatter3d(
            x=x_points, y=y_points, z=z_points,
            mode='markers+lines',
            marker=dict(size=4, color=class_to_color[cls], opacity=0.8),
            line=dict(color=class_to_color[cls], width=2),
            text=hover_texts,  # Attach the histogram to the hover text
            hoverinfo='text',
            name=f'Class {cls}'
        ))

    # Create the layout for the 3D plot
    layout = go.Layout(
        title="3D Expanding Spheres Plot with Class Distribution on Hover",
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis'),
            aspectmode='cube'
        ),
        legend=dict(x=0.7, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode='closest'  # Ensure hover is focused on the closest point
    )

    # Create the figure and display it
    fig = go.Figure(data=scatter_data, layout=layout)
    fig.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_3d_spheres(data)
