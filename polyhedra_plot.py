import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import torch

def plot_polyhedron_2d(A, b, xlim=(-10, 10), ylim=(-10, 10), ax=None, color='lightblue', points=None):
    """Plots a 2D polyhedron with optional labeled points."""
    if A.shape[1] != 2:
        raise ValueError("Function only works for 2D polyhedra")
    
    # Create a grid of points to test
    x_range = np.linspace(xlim[0], xlim[1], 200)
    y_range = np.linspace(ylim[0], ylim[1], 200)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Test which points satisfy all inequalities (with some numerical tolerance)
    tol = 1e-10
    inequalities = np.all(A @ grid_points.T <= b[:, np.newaxis] + tol, axis=0)
    feasible_points = grid_points[inequalities]
    
    # Create a new figure and axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # If no feasible points, return empty plot with a message
    if len(feasible_points) == 0:
        ax.text(0.5, 0.5, "No feasible region found", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax
    
    # Compute the convex hull of the feasible points
    try:
        hull = ConvexHull(feasible_points)
        vertices = feasible_points[hull.vertices]
        
        # Plot the polyhedron
        poly = Polygon(vertices, closed=True, alpha=0.5, color=color, edgecolor='blue')
        ax.add_patch(poly)
        
        # Plot the inequality lines
        for i in range(len(b)+1):
            if np.abs(A[i, 1]) < 1e-10:  # Almost vertical line
                if A[i, 0] > 0:
                    x_val = b[i] / A[i, 0]
                    ax.axvline(x=x_val, color='red', linestyle='--', alpha=0.5)
            else:
                x_vals = np.array([xlim[0], xlim[1]])
                y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                ax.plot(x_vals, y_vals, 'r--', alpha=0.5)
    
    except Exception as e:
        # Fallback: just scatter plot the points
        ax.scatter(feasible_points[:, 0], feasible_points[:, 1], s=1, color=color, alpha=0.5)
    
    # Plot the specified points with labels
    if points is not None:
        for point in points:
            x, y = point[0], point[1]
            label = point[2] if len(point) > 2 else ""
            
            # Plot the point
            ax.plot(x, y, 'ko', markersize=6)
            
            # Add label with slight offset
            if label:
                ax.text(x + 0.1, y + 0.1, label, fontsize=8, ha='left', va='bottom')
    
    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    return ax

def create_square(size=1.0):
    A = np.array([
        [1, 0],   # x ≤ size
        [-1, 0],  # x ≥ -size
        [0, 1],   # y ≤ size
        [0, -1]   # y ≥ -size
    ])
    b = np.array([size, size, size, size])
    return A, b

def create_regular_polygon(n_sides, radius=1.0):
    A = np.zeros((n_sides, 2))
    b = np.zeros(n_sides)
    
    for i in range(n_sides):
        # Calculate the outward normal for each side
        angle = i * 2 * np.pi / n_sides
        # The normal points outward perpendicular to the side
        nx = np.cos(angle)
        ny = np.sin(angle)
        
        # Store the normalized normal vector
        A[i] = [nx, ny]
        
        # Set the right-hand side to maintain distance from origin
        b[i] = radius
    
    return A, b

def generate_random_polytope(n_dim=2, num_halfspaces=8, size_range=(0.5, 2.0)):
    # First, let's generate a box centered at the origin to ensure boundedness
    A_box = np.zeros((2*n_dim, n_dim))
    b_box = np.zeros(2*n_dim)
    
    # For each dimension, add two constraints: x_i ≤ size and -x_i ≤ size
    base_size = np.random.uniform(*size_range)
    for i in range(n_dim):
        # Positive direction: unit vector
        A_box[2*i, :] = np.zeros(n_dim)
        A_box[2*i, i] = 1.0
        b_box[2*i] = base_size
        
        # Negative direction: negative unit vector
        A_box[2*i+1, :] = np.zeros(n_dim)
        A_box[2*i+1, i] = -1.0
        b_box[2*i+1] = base_size
    
    # Now add random half-spaces that cut through the box
    num_additional = max(0, num_halfspaces - 2*n_dim)
    if num_additional > 0:
        A_additional = np.random.normal(0, 1, (num_additional, n_dim))
        
        # Normalize each row to get unit normals
        row_norms = np.sqrt(np.sum(A_additional**2, axis=1))
        A_additional = A_additional / row_norms[:, np.newaxis]
        
        # Generate right-hand sides that ensure the origin is still feasible
        # by making each constraint pass near but not through the origin
        b_additional = np.random.uniform(0.1, base_size, num_additional)
        
        # Combine the box constraints with the additional constraints
        A = np.vstack((A_box, A_additional))
        b = np.hstack((b_box, b_additional))
    else:
        A = A_box
        b = b_box
    
    return A, b


def compute_and_show_center(A, b, title, color):
    """Helper function to compute and display the analytic center of a polyhedron."""
    # Convert to PyTorch tensors
    A_torch = torch.tensor(A, dtype=torch.float32)
    b_torch = torch.tensor(b, dtype=torch.float32)
    
    # Find interior point and analytic center
    interior_point = find_interior_point(A_torch, b_torch)
    center = find_analytic_center(A_torch, b_torch, verbose=False, lr = 1e-5)
    
    # Plot the polyhedron with both points
    fig, ax = plt.subplots(figsize=(8, 6))
    points = [
        (center[0].item(), center[1].item(), "C")
    ]
    plot_polyhedron_2d(A, b, xlim=(-2, 2), ylim=(-2, 2), ax=ax, color=color, points=points)
    plt.title(title)
    plt.show()
    
    return center




