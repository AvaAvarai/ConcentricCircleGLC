import pandas as pd
import numpy as np
import plotly.graph_objs as go
import tkinter as tk
from tkinter import filedialog, messagebox
import colorsys
from typing import List, Tuple, Optional

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

def plot_3d_spheres(data: pd.DataFrame, class_column: str = 'class') -> None:
    """
    Plots data points on expanding 3D spheres using Plotly, coloring them based on their class labels.
    
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

    mins = features.min()
    maxs = features.max()
    normalized_data = (features - mins) / (maxs - mins)

    num_attributes = len(features.columns)
    unique_classes = classes.unique()
    num_classes = len(unique_classes)
    colors = generate_distinct_colors(num_classes)

    class_to_color = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

    radii = np.linspace(1, num_attributes, num_attributes)

    # Prepare data for Plotly with a single trace per class
    scatter_data = []
    for cls in unique_classes:
        cls_data = normalized_data[classes == cls]
        x_points = []
        y_points = []
        z_points = []
        for _, row in cls_data.iterrows():
            points = []
            for j, radius in enumerate(radii):
                theta = 2 * np.pi * row.iloc[j]
                phi = np.arccos(2 * row.iloc[j] - 1)
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                points.append((x, y, z))

            points = np.array(points)
            x_points.extend(points[:, 0])
            y_points.extend(points[:, 1])
            z_points.extend(points[:, 2])

        scatter_data.append(go.Scatter3d(
            x=x_points, y=y_points, z=z_points,
            mode='markers+lines',
            marker=dict(size=4, color=class_to_color[cls], opacity=0.8),
            line=dict(color=class_to_color[cls], width=2),
            name=f'Class {cls}'
        ))

    # Plot reference spheres
    for radius in radii:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

        scatter_data.append(go.Surface(
            x=x, y=y, z=z,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
            hoverinfo='skip'
        ))

    # Create the layout for the 3D plot
    layout = go.Layout(
        title="3D Expanding Spheres Plot",
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis'),
            aspectmode='cube'
        ),
        legend=dict(x=0.7, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Create the figure and display it
    fig = go.Figure(data=scatter_data, layout=layout)
    fig.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_3d_spheres(data)
