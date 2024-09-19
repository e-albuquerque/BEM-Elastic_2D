def mount_bcs(segments, bc_info):
    """
    Constructs a boundary condition matrix from segment information and boundary condition data.

    This function takes a list of segment indices (`segments`) and a dictionary of boundary condition
    information (`bc_info`) to create a matrix (`bcs`) that stores the type and value of the boundary
    condition for each element.

    Args:
        segments (list or np.ndarray): A list or array of integers representing the indices of the boundary
                                       elements (segments) in the problem.
        bc_info (dict): A dictionary containing boundary condition information. The keys are typically
                        strings representing boundary condition names, and the values are dictionaries
                        with the following structure:
                        {
                            'type_n': bc_type,     # The type of boundary condition in n (0 for Dirichlet, 1 for Neumann)
                            'value_n': bc_value    # The value of the boundary condition in n (displacement or traction)
                            'segment': segment_index,  # The index of the segment this BC applies to
                            'type': bc_type,          # The type of boundary condition in t (0 for Dirichlet, 1 for Neumann)
                            'value': bc_value         # The value of the boundary condition in t (displacement or traction)
                        }

    Returns:
        bcs (np.ndarray): A 2D NumPy array where each row corresponds to a boundary element. The first
                         and third column contains the boundary condition type (0 or 1), and the second
                         and fourth column contains the boundary condition value.
    """

    num_elements = len(segments)

    # Initialize boundary condition matrix
    bcs = np.zeros((2*num_elements, 4), dtype=np.float64)

    # Populate boundary condition matrix (bcs)
    for i, segment_index in enumerate(segments):
        # Find the boundary condition data corresponding to the current segment
        for bc_name, bc_data in bc_info.items():
            if bc_data['segment'] == segment_index:
                tp_n=bc_data['type_n']
                vl_n= bc_data['value_n']
                tp_t=bc_data['type_t']
                vl_t= bc_data['value_t']
                bcs[2*i:2*i+2, 0] = np.array([tp_n,tp_n])  # Store boundary condition type
                bcs[2*i:2*i+2, 1] = np.array([vl_n,vl_n]) # Store boundary condition value
                bcs[2*i:2*i+2, 2] = np.array([tp_t,tp_t])  # Store boundary condition type
                bcs[2*i:2*i+2, 3] = np.array([vl_t,vl_t]) # Store boundary condition value
                break  # Move on to the next segment once the BC is found
    return bcs


