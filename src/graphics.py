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