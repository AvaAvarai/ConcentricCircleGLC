import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
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

def plot_3d_spheres(data, class_column='class', grid_size=100):
    data.columns = map(str.lower, data.columns)
    if class_column.lower() not in data.columns:
        raise ValueError(f"Column '{class_column}' not found in the CSV file.")
    
    features = data.drop(columns=[class_column.lower()])
    classes = data[class_column.lower()]

    mins = features.min()
    maxs = features.max()
    normalized_data = (features - mins) / (maxs - mins)

    num_attributes = len(features.columns)
    unique_classes = classes.unique()
    num_classes = len(unique_classes)
    colors = generate_distinct_colors(num_classes)

    class_to_color = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

    radii = np.linspace(1, num_attributes, num_attributes)

    # Generate a full lattice (grid) for each sphere
    lattice_points = distribute_points_on_sphere(grid_size)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', facecolor='lightgrey')

    # Plot the lattice grid on each sphere
    for j, radius in enumerate(radii):
        lattice = lattice_points * radius
        ax.scatter(lattice[:, 0], lattice[:, 1], lattice[:, 2], color='lightgrey', alpha=0.2)

    # Plot points on the surface of spheres based on data and connect them
    for i, row in normalized_data.iterrows():
        points = []
        for j, radius in enumerate(radii):
            # Use the Fibonacci lattice to place data points on the sphere
            point_idx = int(row.iloc[j] * (grid_size - 1))  # Map normalized value to a point on the lattice
            x, y, z = lattice_points[point_idx] * radius
            points.append((x, y, z))

        points = np.array(points)
        color = class_to_color[classes.iloc[i]]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.8)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.8)

    # Add reference spheres for visual reference
    for radius in radii:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightgrey', alpha=0.05, zorder=-1)

    for j, radius in enumerate(radii):
        ax.text(radius, 0, 0, features.columns[j], horizontalalignment='center', verticalalignment='center', color='black')

    axcolor = 'lightgoldenrodyellow'
    ax_rot_x = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
    ax_rot_y = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)

    rot_x_slider = Slider(ax_rot_x, 'Rotate X', 0, 360, valinit=0)
    rot_y_slider = Slider(ax_rot_y, 'Rotate Y', 0, 360, valinit=0)

    def update(val):
        ax.view_init(elev=rot_y_slider.val, azim=rot_x_slider.val)
        plt.draw()

    rot_x_slider.on_changed(update)
    rot_y_slider.on_changed(update)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[cls], markersize=10, label=cls) for cls in unique_classes]
    ax.legend(handles=legend_elements, loc='upper right', title="Classes")

    plt.title("3D Expanding Spheres Plot with Spherical Lattice", pad=20)
    plt.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_3d_spheres(data)
