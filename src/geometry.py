def compute_inodes(file_name, bound_cond):
    """
    This function reads a mesh file and extracts information about nodes, elements,
    and boundary conditions.

    Args:
      file_name: Name of the mesh file.
      bound_cond: Dictionary containing boundary conditions for each edge.

    Returns:
      A dictionary containing information about nodes, elements, and boundary
      conditions.
    """
    mesh = meshio.read(file_name + '.msh')
    coordinates = mesh.points
    line_elements = mesh.cells_dict['line']
    triangle_elements = mesh.cells_dict['triangle']
    segments = mesh.cell_data_dict['gmsh:physical']['line']

    bc_info = {}
    for key, value in bound_cond.items():
        segment = mesh.field_data[key][0]
        bc_info[key] = {
            'type_n': value['type_n'],
            'value_n': value['value_n'],
            'segment': segment,
            'type_t': value['type_t'],
            'value_t': value['value_t']
        }

    boundary_nodes = np.unique(line_elements)
    all_nodes = np.unique(triangle_elements)
    internal_nodes = [node for node in all_nodes if node not in boundary_nodes]

    return {
        'boundary_nodes': boundary_nodes,
        'internal_nodes': internal_nodes,
        'all_nodes': all_nodes,
        'coordinates': coordinates,
        'line_elements': line_elements,
        'triangle_elements': triangle_elements,
        'segments': segments,
        'bc_info': bc_info
    }




def comp_node_and_normal(elem, nodes):
    """
    Calculates the midpoint and outward normal vector for each boundary element.

    This function takes element connectivity information (`elem`) and node coordinates (`nodes`)
    to determine the geometric center of each boundary element and the unit normal vector
    pointing outward from that element.

    Args:
        elem (np.ndarray): A 2D array defining the connectivity of the boundary elements.
                          Each row contains the indices of the two nodes that form an element.
                          Shape: (num_elements, 2).
        nodes (np.ndarray): A 2D array containing the coordinates (x, y) of each node in the mesh.
                           Shape: (num_nodes, 2).

    Returns:
        node_des (np.ndarray): A 2D array containing the coordinates (x, y) of the discontinuous nodes
                              of each boundary element. Shape: (2*num_elements, 2).
        normal (np.ndarray): A 2D array containing the unit outward normal vectors for each
                             boundary element. Shape: (num_elements, 2).
    """

    num_elements = elem.shape[0]

    # Initialize arrays to store results
    node_des = np.zeros((2*num_elements, 2))  # Midpoint coordinates
    normal = np.zeros((2*num_elements, 2))    # Outward normal vectors

    # Iterate over each boundary element
    for i in range(num_elements):
        # Get node indices for the current element
        node1, node2 = elem[i]

        # Get coordinates of the two nodes forming the element
        x1, y1 = nodes[node1][0:2]
        x2, y2 = nodes[node2][0:2]


        # Calculate the midpoint coordinates
        node_des[2*i,:] = [x1+(x2 - x1) / 4, y1+(y2 - y1) / 4]

        # Calculate the midpoint coordinates
        node_des[2*i+1,:] = [x1+3*(x2 - x1) / 4, y1+3*(y2 - y1) / 4]

        # Calculate the length of the element
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the unit tangent vector (from node1 to node2)
        tangent_x = (x2 - x1) / L
        tangent_y = (y2 - y1) / L

        # Calculate the outward unit normal vector (rotate tangent 90 degrees counterclockwise)
        normal[2*i,:] = [tangent_y, -tangent_x]
        normal[2*i+1,:] = [tangent_y, -tangent_x]

    return node_des, normal


