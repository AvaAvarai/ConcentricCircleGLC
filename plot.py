import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

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

def plot_concentric_circles(data, class_column='class'):
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

    # Setup radii for concentric circles
    radii = np.linspace(1, 4, num_attributes)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='lightgrey')

    for i, row in normalized_data.iterrows():
        points = []
        for j, radius in enumerate(radii):
            angle = -2 * np.pi * row.iloc[j] + np.pi / 2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append((x, y))

        points = np.array(points)
        class_index = np.where(unique_classes == classes.iloc[i])[0][0]
        ax.plot(points[:, 0], points[:, 1], color=colors(class_index), alpha=0.3, zorder=1)
        ax.plot(points[:, 0], points[:, 1], 'o', color=colors(class_index), alpha=0.6, zorder=1)

    # Draw the circles
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
    for i, class_label in enumerate(unique_classes):
        ax.plot([], [], color=colors(i), label=class_label, linewidth=5)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    plt.legend(loc='upper right')
    plt.title("Concentric Circles Plot", pad=20)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    data = load_csv()
    if data is not None:
        plot_concentric_circles(data)
