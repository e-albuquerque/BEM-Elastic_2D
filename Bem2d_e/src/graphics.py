import matplotlib.pyplot as plt
import matplotlib.tri as triang
from matplotlib import cm
import numpy as np


def show_problem(nodes, normal, coord, bcs, tri):
    """
   Visualizes the mesh and boundary conditions of a 2D problem.

    This function generates a plot that displays:

    - Mesh: The underlying triangular mesh structure used for the problem.
    - Known Displacement Nodes: Nodes where the displacement is prescribed
       (Dirichlet boundary conditions). Marked as red dots.
    - Known Traction Nodes: Nodes where the traction is prescribed (Neumann boundary
       conditions). Marked as blue dots.
    - Normal Vectors: Arrows indicating the direction of the outward normal
       vectors at nodes with known traction (blue arrows) or displacement (red arrows).
       The length of the arrow is scaled by the magnitude of the displacement/traction.

    Args:
        node_med (np.ndarray): A 2D array containing the coordinates of the midpoint
                              of each element in the mesh. Shape: (num_elements, 2).
        normal (np.ndarray): A 2D array containing the outward normal vectors at
                             each element midpoint. Shape: (num_elements, 2).
        coord (np.ndarray): A 2D array containing the coordinates of all nodes in
                            the mesh. Shape: (num_nodes, 2).
        bcs (np.ndarray): A 2D array containing the boundary condition information
                          for each element. Each row has the following structure:
                          [element_index, bc_type n, bc_value n, bc_type t, bc_value t].
                          - bc_type: 0 for Dirichlet (known displacement), 1 for
                                     Neumann (known traction).
                          - bc_value: The prescribed displacement or traction value.
        tri (np.ndarray): A 2D array defining the connectivity of the mesh
                          (element-node relationships). Each row contains the
                          indices of the three nodes forming a triangle.
                          Shape: (num_triangles, 3).
    """

    # Extract indices of elements with known displacement and traction conditions
    indknown_ux = np.where(bcs[:, 0] == 0)[0]  # Dirichlet BC (known displacement)
    indknown_tx = np.where(bcs[:, 0] == 1)[0]  # Neumann BC (known traction)
    indknown_uy = np.where(bcs[:, 2] == 0)[0]  # Dirichlet BC (known displacement)
    indknown_ty = np.where(bcs[:, 2] == 1)[0]  # Neumann BC (known traction)

    # Determine plot limits to provide a 15% margin around the geometry
    x_min, x_max = np.min(coord[:, 0]), np.max(coord[:, 0])
    y_min, y_max = np.min(coord[:, 1]), np.max(coord[:, 1])
    delta_x, delta_y = x_max - x_min, y_max - y_min

    # Get the current axes for plotting
    ax = plt.gca()
    plt.grid(True)  # Turn on gridlines for better visualization

    # Plot the triangular mesh
    plt.triplot(coord[:, 0], coord[:, 1], tri, color="black", linewidth=0.4)

    # Plot normal vectors (scaled by BC value) for elements with known traction (blue)
    plt.quiver(nodes[indknown_tx, 0], nodes[indknown_tx, 1],
               normal[indknown_tx, 0] * bcs[indknown_tx, 1], normal[indknown_tx, 1] * bcs[indknown_tx, 1],
               color="blue", angles='xy', scale_units='xy', scale=10, width=0.004, headaxislength=0)

    # Plot normal vectors (scaled by BC value) for elements with known displacement (red)
    plt.quiver(nodes[indknown_ux, 0], nodes[indknown_ux, 1],
               normal[indknown_ux, 0] * bcs[indknown_ux, 1], normal[indknown_ux, 1] * bcs[indknown_ux, 1],
               color="red", angles='xy', scale_units='xy', scale=10, width=0.004, headaxislength=0)

  # Plot normal vectors (scaled by BC value) for elements with known traction (blue)
    plt.quiver(nodes[indknown_ty, 0], nodes[indknown_ty, 1],
               -normal[indknown_ty, 1] * bcs[indknown_ty, 3], normal[indknown_ty, 0] * bcs[indknown_ty, 3],
               color="blue", angles='xy', scale_units='xy', scale=10, width=0.004, headaxislength=0)

    # Plot normal vectors (scaled by BC value) for elements with known displacement (red)
    plt.quiver(nodes[indknown_uy, 0], nodes[indknown_uy, 1],
               -normal[indknown_uy, 1] * bcs[indknown_uy, 3], normal[indknown_uy, 0] * bcs[indknown_uy, 3],
               color="red", angles='xy', scale_units='xy', scale=10, width=0.004, headaxislength=0)


    # Plot nodes with known displacement (red dots)
    plt.plot(nodes[indknown_ux, 0], nodes[indknown_ux, 1], 'rx', markersize=6)

    # Plot nodes with known traction (blue dots)
    plt.plot(nodes[indknown_tx, 0], nodes[indknown_tx, 1], 'bx', markersize=6)

    # Plot nodes with known displacement (red dots)
    plt.plot(nodes[indknown_uy, 0], nodes[indknown_uy, 1], 'rP', markersize=4)

    # Plot nodes with known traction (blue dots)
    plt.plot(nodes[indknown_ty, 0], nodes[indknown_ty, 1], 'bP', markersize=4)

    # Set plot aspect ratio and adjust limits for a better visual representation
    plt.axis("equal")
    ax.set_xlim(x_min - 0.15 * delta_x, x_max + 0.15 * delta_x)
    ax.set_ylim(y_min - 0.15 * delta_y, y_max + 0.15 * delta_y)

    # Show the plot
    plt.show()
    return

def inpoly(points, polygon_vertices, polygon_edges):
    """
    Determines if a set of points are inside a polygon with potential holes.

    Args:
        points: (np.ndarray) An array of shape (n, 2) where each row is a point to test.
        polygon_vertices: (np.ndarray) An array of shape (m, 2) where each row is a polygon vertex.
        polygon_edges: (np.ndarray) An array of shape (p, 2) where each row contains indices of two vertices forming an edge.

    Returns:
        np.ndarray: A boolean array of shape (n,) where True indicates a point is inside the polygon.
    """

    reltol = 1e-12  # Relative tolerance for numerical comparisons

    num_points = points.shape[0]
    num_edges = polygon_edges.shape[0]

    # Optimize for longer dimension
    x_range, y_range = np.max(points, 0) - np.min(points, 0)
    if x_range > y_range:
        points = points[:, [1, 0]]  # Swap coordinates if x range is larger
        polygon_vertices = polygon_vertices[:, [1, 0]]

    # Calculate polygon bounding box
    min_coords = np.min(polygon_vertices, 0)
    max_coords = np.max(polygon_vertices, 0)
    tol = reltol * min(max_coords - min_coords)
    tol = max(tol, reltol)  # Ensure a minimum tolerance

    # Sort points by y-coordinate for efficient iteration
    if num_points > 1:
        sort_indices = np.argsort(points[:, 1])
        points_sorted = points[sort_indices]
    else:
        points_sorted = points
        sort_indices = np.array([0])

    inside = np.zeros(num_points, dtype=bool)

    for k in range(num_edges):
        # Get vertices of the current edge
        v1, v2 = polygon_vertices[polygon_edges[k]]

        # Ensure v1 is the lower vertex (smaller y-coordinate)
        if v1[1] > v2[1]:
            v1, v2 = v2, v1

        # Skip edge if entirely above or below all points
        if v2[1] < points_sorted[0, 1] or v1[1] > points_sorted[-1, 1]:
            continue

        # Find the index of the first point above the lower endpoint of the edge
        idx = np.searchsorted(points_sorted[:, 1], v1[1], side="left")

        for j in range(idx, num_points):
            p = points_sorted[j]
            # If point is below the upper endpoint, stop checking this edge
            if p[1] > v2[1]:
                break

            # If point is to the left of the edge's bounding box, skip
            if p[0] < min(v1[0], v2[0]):
                continue

            # Point is on the right side; check for intersection
            if p[0] <= max(v1[0], v2[0]):
                # Check for intersection with the edge (cross product)
                if (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0]) < tol:
                    inside[sort_indices[j]] = not inside[sort_indices[j]] # Toggle inside/outside status
            # If point is to the right of the edge's bounding box, and below the line
            # it's inside if the polygon is oriented clockwise
            elif v1[1] != v2[1] and (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0]) < -tol:
                inside[sort_indices[j]] = not inside[sort_indices[j]] # Toggle inside/outside status
    return inside

def show_results(node_all, bound_nodes, node_int, nodes, elem, coord, u_t, uint_t,title_fig):

    """Plots the displacement map of a 2D body with internal nodes and boundary elements."""

    # Create mesh with middle points of elements and internal points
    coord_tri2 = np.concatenate((nodes[:, 0:2], coord[node_int, 0:2]), axis=0)
    trimesh = triang.Triangulation(coord_tri2[:, 0], coord_tri2[:, 1])

    # Prepare colors and triangle centroids
    cor = np.concatenate((u_t, uint_t))
    centroid = (coord_tri2[trimesh.triangles[:, 0], :] +
                coord_tri2[trimesh.triangles[:, 1], :] +
                coord_tri2[trimesh.triangles[:, 2], :]) / 3.0

    # Filter out triangles outside the domain
    ind = inpoly(centroid[:, 0:2], coord[bound_nodes, 0:2], elem)
    valid_triangles = trimesh.triangles[ind, :]

    # Plot
    fig, ax = plt.subplots()  # Create a figure and axis

    # Plot mesh outline
    ax.triplot(coord_tri2[:, 0], coord_tri2[:, 1], valid_triangles,
               color=(0.0, 0., 0.), linewidth=0.4)

    # Contour plot with colorbar
    contourf = ax.tricontourf(coord_tri2[:, 0], coord_tri2[:, 1], valid_triangles, cor, cmap=cm.jet)
    fig.colorbar(contourf, ax=ax, label=title_fig)  # Add colorbar

    ax.set_title(title_fig)
    ax.set_aspect("equal")
    plt.show()

def compute_bounds(elem, nodes_all):
    """
    Renumbers nodes in boundary elements to use local node numbers.

    Args:
        elem: A NumPy array of shape (nelem, 2) where each row represents a boundary
              element with global node indices.
        nodes_all: A list or array containing all unique global node indices.

    Returns:
        elem_local: A NumPy array of shape (nelem, 2) where each row represents a
                    boundary element with local node indices.
    """

    nelem = elem.shape[0]  # Number of boundary elements

    # Initialize arrays to store local node numbers and updated elements
    elem_local = np.zeros((nelem, 2), dtype=int)
    nodes_local = np.zeros(len(nodes_all))

    # Extract unique node indices from elements to get the order in which they appear
    seq_nodes = elem[:, 0]
    nnodes_local = len(seq_nodes)

    # Assign local node numbers based on the order they appear in the elements
    for t in range(nnodes_local):
        inode_global = seq_nodes[t]
        nodes_local[inode_global] = t  # Local node number is the index in the sequence

    # Update elements with local node numbers
    for t in range(nelem):
        inode1 = elem[t, 0]   # Global node index of the first node in the element
        inode2 = elem[t, 1]   # Global node index of the second node in the element

        inode1_local = nodes_local[inode1]  # Get local node index for the first node
        inode2_local = nodes_local[inode2]  # Get local node index for the second node

        elem_local[t] = [inode1_local, inode2_local]  # Store updated element with local indices

    return elem_local