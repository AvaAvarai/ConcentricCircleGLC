import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog, messagebox

def load_csv():
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

def plot_3d_spheres(data, class_column='class'):
    # Ensure the class column is treated in a case-insensitive manner
    data.columns = map(str.lower, data.columns)
    if class_column.lower() not in data.columns:
        raise ValueError(f"Column '{class_column}' not found in the CSV file.")
    
    # Separate the features and the class labels
    features = data.drop(columns=[class_column.lower()])
    classes = data[class_column.lower()]

    # Normalize the data
    mins = features.min()
    maxs = features.max()
    normalized_data = (features - mins) / (maxs - mins)

    # Determine the number of attributes and classes
    num_attributes = len(features.columns)
    unique_classes = classes.unique()
    colors = plt.cm.get_cmap('tab10', len(unique_classes))

    # Setup radii for expanding spheres
    radii = np.linspace(1, 4, num_attributes)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', facecolor='lightgrey')

    # Plot points on the surface of spheres and connect them
    for i, row in normalized_data.iterrows():
        points = []
        for j, radius in enumerate(radii):
            theta = 2 * np.pi * row.iloc[j]  # Full circle [0, 2pi] for azimuthal angle
            phi = np.arccos(2 * row.iloc[j] - 1)  # Full sphere [0, pi] for polar angle
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            points.append((x, y, z))

        points = np.array(points)
        class_index = np.where(unique_classes == classes.iloc[i])[0][0]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors(class_index), alpha=0.3)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors(class_index), alpha=0.6)

    # Plot transparent spheres for visual reference
    for radius in radii:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightgrey', alpha=0.1, zorder=-1)

    # Add labels for each attribute on the spheres
    for j, radius in enumerate(radii):
        ax.text(radius, 0, 0, features.columns[j], horizontalalignment='center', verticalalignment='center', color='black')

    # Add sliders for rotating the 3D plot
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

    plt.title("3D Expanding Spheres Plot", pad=20)
    plt.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_3d_spheres(data)
