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

    # Track non-pure points for flashing
    non_pure_points = []
    for point, distribution in point_class_distribution.items():
        if len(distribution) > 1:  # More than one class at this point
            non_pure_points.append(point)

    # Flashing effect for non-pure points
    frames = []
    for i in range(2):  # Two frames to toggle visibility
        frame_data = []
        for trace in scatter_data:
            if trace.name.startswith('Class'):
                new_colors = []
                for x, y, z in zip(trace.x, trace.y, trace.z):
                    if i % 2 == 0 and (x, y, z) in non_pure_points:
                        new_colors.append('yellow')  # Highlight non-pure points
                    else:
                        new_colors.append(trace.marker.color)
                
                frame_data.append(go.Scatter3d(
                    x=trace.x, y=trace.y, z=trace.z,
                    mode='markers+lines',
                    marker=dict(size=trace.marker.size, color=new_colors, opacity=trace.marker.opacity),
                    line=dict(color=trace.line.color, width=trace.line.width),
                    text=trace.text,
                    hoverinfo=trace.hoverinfo,
                    name=trace.name
                ))
        frames.append(go.Frame(data=frame_data))

    # Plot reference spheres (optional, for visual reference)
    sphere_traces = []
    sphere_opacity = 0.05  # Default opacity for spheres

    for radius in radii:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

        sphere_traces.append(go.Surface(
            x=x, y=y, z=z,
            opacity=sphere_opacity,  # Use the default opacity
            showscale=False,
            colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
            hoverinfo='skip',  # Disable hover when spheres are toggled off
            visible=True  # Default visibility
        ))

    scatter_data.extend(sphere_traces)

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
        hovermode='closest',  # Ensure hover is focused on the closest point
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Toggle Spheres",
                        "method": "update",
                        "args": [
                            {"visible": [False if trace in sphere_traces else True for trace in scatter_data]},
                            {"title": "Spheres Off"}
                        ],
                        "args2": [
                            {"visible": [True] * len(scatter_data)},
                            {"title": "Spheres On"}
                        ]
                    },
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Flash Non-Pure Points",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "left",
                "y": 0.05,
                "yanchor": "bottom"
            }
        ]
    )

    # Create the figure and display it
    fig = go.Figure(data=scatter_data, layout=layout, frames=frames)
    fig.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_3d_spheres(data)
