def compute_fund_solutions(r1, r2, r, nx, ny, E, nu):
  """
  Computes the fundamental solutions for plane elasticity.

  Args:
    r1: x-coordinate of the distance vector.
    r2: y-coordinate of the distance vector.
    r: Magnitude of the distance vector.
    nx: x-component of the outward normal vector.
    ny: y-component of the outward normal vector.
    E: Young's modulus.
    nu: Poisson's ratio.

  Returns:
    A tuple containing the displacement and traction fundamental solution matrices.
  """
  mu = E / (2 * (1 + nu))
  # Components of the unity vector in the radial direction
  rd1 = r1 / r
  rd2 = r2 / r
  # Plane elasticity fundamental solutions
  c1 = 4 * np.pi * (1 - nu)
  c2 = (3 - 4 * nu) * np.log(1 / r)
  c3 = rd1 * nx + rd2 * ny

  u11 = (c2 + rd1 ** 2) / (2 * c1 * mu)
  u22 = (c2 + rd2 ** 2) / (2 * c1 * mu)
  u12 = (rd1 * rd2) / (2 * c1 * mu)
  u21 = u12

  t11 = -(c3 * ((1 - 2 * nu) + 2 * rd1 ** 2)) / (c1 * r)
  t22 = -(c3 * ((1 - 2 * nu) + 2 * rd2 ** 2)) / (c1 * r)
  t12 = -((c3 * 2 * rd1 * rd2) - (1 - 2 * nu) * (rd1 * ny - rd2 * nx)) / (c1 * r)
  t21 = -((c3 * 2 * rd1 * rd2) - (1 - 2 * nu) * (rd2 * nx - rd1 * ny)) / (c1 * r)

  # Assembly of matrices that contain fundamental solutions
  uast = np.array([[u11, u12], [u21, u22]])
  tast = np.array([[t11, t12], [t21, t22]])
  return uast, tast

def compute_gh_sing(E,nu,x1,x2,y1,y2,n1,n2,xidd):
  """
  Compute g and h elementary matrices when the source point
  belongs to the element.

  Args:
    E: Young's modulus.
    nu: Poisson's ratio.
    x1, x2: x-coordinates of the element nodes.
    y1, y2: y-coordinates of the element nodes.
    n1, n2: x and y components of the outward normal vector.
    xidd: Local coordinate of the source point.

  Returns:
    g_el: Elementary G matrix.
    h_el: Elementary H matrix.
  """

  xid = -0.5  # Local coordinate of node 1
  mu = E / (2 * (1 + nu))
  c1 = 4 * np.pi * (1 - nu)
  L = 2 * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

  # Initialize H matrix components
  h11 = 0.5
  h22 = 0.5
  h13 = 0.0
  h24 = 0.0

  # Compute h12, h21, h14, h23
  fin_part = (-2 + np.log(np.abs((1 - xid) / (1 + xid)))) # finite part
  coef = 2 * (1 - 2 * nu) * ((x2 - x1) * n2 - (y2 - y1) * n1) / (L * c1)
  h12 = coef * fin_part
  h21 = -h12
  h14 = coef * 2
  h23 = -h14

  # Compute G matrix components
  g11s = -1.781167572309405 + np.log(L / 2)
  g11ns = 4 * (x2 - x1) ** 2 / L ** 2
  g13s = 0.042791644191678246 + np.log(L / 2)
  g13ns = g11ns
  g13 = g13s + g13ns
  g22s = g11s
  g22ns = 4 * (y2 - y1) ** 2 / L ** 2
  g24s = g13s
  g24ns = g22ns

  g11 = -L * (3 - 4 * nu) / (4 * c1 * mu) * g11s + L / (4 * c1 * mu) * g11ns
  g22 = -L * (3 - 4 * nu) / (4 * c1 * mu) * g22s + L / (4 * c1 * mu) * g22ns
  g24 = -L * (3 - 4 * nu) / (4 * c1 * mu) * g24s + L / (4 * c1 * mu) * g24ns
  g13 = -L * (3 - 4 * nu) / (4 * c1 * mu) * g13s + L / (4 * c1 * mu) * g13ns
  g14 = (x2 - x1) * (y2 - y1) / (c1 * L * mu)
  g23 = g14
  g12 = g14
  g21 = g14

  if xidd == -.5:
    g_el = np.array([[g11, g12, g13, g14], [g21, g22, g23, g24]])
    h_el = np.array([[h11, h12, h13, h14], [h21, h22, h23, h24]])
  else:
    g_el = np.array([[g13, g14, g11, g12], [g23, g24, g21, g22]])
    h_el = np.array([[h13, -h14, h11, -h12], [-h23, h24, -h21, h22]])

  return g_el, h_el


def mount_matrices(nodes, normal, E, nu, qpoint_str):
    """
    Computes H, G matrices, and q vector for the Boundary Element Method (BEM)
    applied to the linear elastic problem with radial integration for domain
    integrals.

    Args:
        node_med (ndarray): Coordinates of element midpoints.
        normal (ndarray): Normal vectors at element midpoints.
        nodes_coord (ndarray): Coordinates of all nodes.
        elem (ndarray): Element connectivity matrix.
        E (float): Material elastic modulus.
        nu (float): Material Poisson coefficient.
        qpoint_str (str): String representing body force.

    Returns:
        H (ndarray): BEM influence matrix.
        G (ndarray): BEM influence matrix.
        q (ndarray): Vector accounting for body contribution.
    """

    # Number of integration points and Gauss quadrature data
    npgauss = 6
    xi, weight = np.polynomial.legendre.leggauss(npgauss)

    # Number of elements and matrix initialization
    nn = nodes.shape[0]
    H = np.zeros((2*nn, 2*nn))
    G = np.zeros((2*nn, 2*nn))
    q = np.zeros(2*nn)

    # Constants for efficiency (avoid repeated calculations in loops)
    one_over_2pi = 1 / (2 * np.pi)

    # Loop over source elements (ii) and field elements (jj)
    I1=1.781167572309405
    I2=-0.042791644191678246
    for ii in range(nn):
        x0, y0 = nodes[ii]  # Source point coordinates

        for jj in range(nn//2):
            # Element geometry (starting/ending node coordinates and length)
            x1, y1 = nodes[2*jj,:]
            x2, y2 = nodes[2*jj+1,:]
            L = 2*np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            coef = L*one_over_2pi/2
            g1 = coef*(I1-np.log(L/2))
            g2 = coef*(I2-np.log(L/2))
            nx, ny = normal[2*jj]  # Normal vector components

            if ii == 2*jj:  # Singular integration (same element)
                g_el,h_el=compute_gh_sing(E,nu,x1,x2,y1,y2,nx,ny,-.5)
                G[2*ii:2*ii+2, 4*jj:4*jj+4] = g_el
                H[2*ii:2*ii+2, 4*jj:4*jj+4] = h_el
            elif ii == 2*jj+1:
                g_el,h_el=compute_gh_sing(E,nu,x1,x2,y1,y2,nx,ny,.5)
                G[2*ii:2*ii+2, 4*jj:4*jj+4] = g_el
                H[2*ii:2*ii+2, 4*jj:4*jj+4] = h_el

            else:  # Regular integration (different elements)
                intG = np.zeros((2,4)) # Initialize integrals
                intH = np.zeros((2,4))
                intq = np.zeros(2)
                for kk in range(npgauss):
                    # Integration point coordinates and shape functions
                    N1 = 0.5 - xi[kk]
                    N2 = 0.5 + xi[kk]
                    N = np.array([[N1,0, N2,0],[0,N1,0,N2]])
                    x = N1 * x1 + N2 * x2
                    y = N1 * y1 + N2 * y2
                    # Distance of source and field points
                    r1 = x - x0
                    r2 = y - y0
                    r = np.sqrt(r1**2+r2**2)


                    # Distance and fundamental solutions
                    uast,tast= compute_fund_solutions(r1,r2,r, nx,ny,E,nu)

                    # Radial integration
                    if(float(qpoint_str[0]) != 0.):
                        intF = np.zeros(2)
                        theta=np.arctan2(r1,r2)
                        for kkk in range(npgauss):
                            rho = r * (xi[kkk] + 1) / 2
                            r1=rho*np.cos(theta)
                            r2=rho*np.sin(theta)
                            uast2,tast2 = compute_fund_solutions(r1,r2,r,nx,ny,E,nu)
                            f=np.array([0,0])
                            intF += uast2.dot(f) * rho * r / 2 * weight[kkk]
                        intq += intF * (nx * r1 + ny * r2) / r**2 * L / 2 * weight[kk]


                    # Update integrals

                    intG += uast.dot(N) * L / 2 * weight[kk]
                    intH += tast.dot(N) * L / 2 * weight[kk]

                H[2*ii:2*ii+2, 4*jj:4*jj+4] = intH
                G[2*ii:2*ii+2, 4*jj:4*jj+4] = intG
                q[2*ii:2*ii+2] += intq



    return H, G, q


def transformation_matrix(normal):
  """
  Computes the transformation matrix for converting between global (x, y)
  and local (n, t) coordinate systems for each node in the mesh.

  Args:
    normal: A NumPy array of shape (num_nodes, 2) representing the outward
            normal vector at each node.

  Returns:
    A NumPy array of shape (2 * num_nodes, 2 * num_nodes) representing the
    transformation matrix for the entire mesh.
  """

  num_nodes = normal.shape[0]
  transformation_matrix = np.zeros((2 * num_nodes, 2 * num_nodes))

  for i in range(num_nodes):
    nx, ny = normal[i]
    local_transformation = np.array([[nx, -ny], [ny, nx]])
    transformation_matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2] = local_transformation

  return transformation_matrix

def mount_linear_system(H, G, bcs):
    """
    Assembles the system matrix A and right-hand side vector b for the BEM linear system.

    This function takes the H and G matrices (influence matrices) from the Boundary Element
    Method (BEM) and boundary conditions (bcs) to construct the linear system of equations
    that needs to be solved.

    The system of equations is of the form: A * x = b, where:
        - A is the system matrix (2nn x 2nn)
        - x is the vector of unknowns (displacements or tractions at each degree of freedom)
        - b is the right-hand side vector

    Args:
        H (np.ndarray): H influence matrix (2nn x 2nn)
        G (np.ndarray): G influence matrix (2nn x 2nn)
        bcs (np.ndarray): Boundary condition matrix. Each row has the format
                         [node_index, bc_type x, bc_value x, bc_type y, bc_value y].

    Returns:
        A (np.ndarray): Assembled system matrix (2nn x 2nn)
        b (np.ndarray): Assembled right-hand side vector (2nn,)
    """

    ne = bcs.shape[0]//2  # Number of boundary elements

    # Initialize matrices and vectors
    A = np.zeros((4*ne, 4*ne))  # System matrix
    B = np.zeros((4*ne, 4*ne))  # Temporary matrix for storing G or H, depending on BCs
    b = np.zeros(4*ne)       # Right-hand side vector
    cdc_val = np.zeros(4*ne)
    # Assemble the A matrix and B matrix based on boundary conditions
    for el in range(ne):  # Iterate over each boundary element
        if bcs[2*el, 0] == 0:  # Dirichlet BC: Displacement is known
            A[:, 4*el:4*el+4] = -G[:, 4*el:4*el+4]  # A gets -G for this column
            B[:, 4*el:4*el+4] = -H[:, 4*el:4*el+4]  # B gets -H for this column
        else:  # Neumann BC: traction is known
            A[:, 4*el:4*el+4] = H[:, 4*el:4*el+4]   # A gets H for this column
            B[:, 4*el:4*el+4] = G[:, 4*el:4*el+4]   # B gets G for this column
    for el in range(ne):
        for no in range(2):
          for gdl in range(2):
            cdc_val[4*el+2*no+gdl] = bcs[2*el+no,2*gdl+1]
    # Calculate the right-hand side vector b
    b = B @ cdc_val  # Matrix-vector multiplication: b = B * bc_values
    return A, b
import numpy as np

def mount_vectors(solution, bcs):
    """
    Constructs the boundary displacement (u) and traction (t) vectors from the
    solution vector and boundary conditions.

    This function takes the solution vector `x` obtained from solving the BEM
    linear system and the boundary conditions `bcs` to reconstruct the complete
    displacement and traction values at all boundary nodes.

    Args:
        solution (np.ndarray): The solution vector (2nn,) containing either
                               displacement or traction values at each boundary node,
                               depending on the boundary condition type.
        bcs (np.ndarray): The boundary condition matrix (2nn, 5) where each row
                          has the format:
                       [node_index, bc_type n, bc_value n,bc_type t, bc_value t]

    Returns:
        u (np.ndarray): The displacement vector (2nn,) containing the
                        displacement values at all degrees of freedom.
        t (np.ndarray): The traction vector (2nn,) containing the traction
                        values at all degrees of freedom.
    """

    ne = bcs.shape[0] // 2  # Number of boundary elements

    # Initialize matrices to store displacements and tractions values
    u = np.zeros((2 * ne, 2))
    t = np.zeros((2 * ne, 2))

    # Iterate over each boundary element (and corresponding node)
    for elem in range(ne):
        # Extract boundary conditions for the current element
        bc_n_type = bcs[2 * elem, 0]
        bc_n_value = bcs[2 * elem, 1]
        bc_t_type = bcs[2 * elem, 2]
        bc_t_value = bcs[2 * elem, 3]

        # Assign displacement and traction values based on boundary conditions
        if bc_n_type == 0:  # Dirichlet BC for n-component
            u[2 * elem, 0] = bc_n_value
            t[2 * elem, 0] = solution[4 * elem]
        else:  # Neumann BC for n-component
            u[2 * elem, 0] = solution[4 * elem]
            t[2 * elem, 0] = bc_n_value

        if bc_t_type == 0:  # Dirichlet BC for t-component
            u[2 * elem, 1] = bc_t_value
            t[2 * elem, 1] = solution[4 * elem + 1]
        else:  # Neumann BC for t-component
            u[2 * elem, 1] = solution[4 * elem + 1]
            t[2 * elem, 1] = bc_t_value


        if bcs[2 * elem + 1, 0] == 0:  # Dirichlet BC for n-component
            u[2 * elem + 1, 0] = bcs[2 * elem + 1, 1]
            t[2 * elem + 1, 0] = solution[4 * elem + 2]
        else:  # Neumann BC for n-component
            u[2 * elem + 1, 0] = solution[4 * elem + 2]
            t[2 * elem + 1, 0] = bcs[2 * elem + 1, 1]

        if bcs[2 * elem + 1, 2] == 0:  # Dirichlet BC for t-component
            u[2 * elem + 1, 1] = bcs[2 * elem + 1, 3]
            t[2 * elem + 1, 1] = solution[4 * elem + 3]
        else:  # Neumann BC for t-component
            u[2 * elem + 1, 1] = solution[4 * elem + 3]
            t[2 * elem + 1, 1] = bcs[2 * elem + 1, 3]

    return u, t


def int_point(node_int, normal, nodes, nodes_coord, E, nu, qpoint_str, u, t):
    """
    Assembles H, G matrices, and their derivatives, as well as vector q and its derivatives
    for internal points in the Boundary Element Method.

    Args:
        node_int (ndarray): Indices of internal points.
        normal (ndarray): Normal vectors at element midpoints.
        nodes (ndarray): Coordinates of all nodes.
        E (float): Material elastic modulus.
        nu (float): Poisson coefficient.
        qpoint_str (str): String representing heat source point coordinates.

    Returns:
        Hin, Gin (ndarray): BEM influence matrices for internal points (displacement and traction).
        q (ndarray): Vector of body force contributions at internal points.
        dHdx, dHdy, dGdx, dGdy (ndarray): Derivatives of H and G w.r.t. x and y.
        dqx, dqy (ndarray): Derivatives of q w.r.t. x and y.
    """

    # Number of integration points and Gauss quadrature data
    npgauss = 4
    xi, weight = np.polynomial.legendre.leggauss(npgauss)

    # Number of internal points and elements
    nintnodes = len(node_int)
    nnodes = nodes.shape[0]

    # Matrix/vector initialization (with proper shapes)
    Hin = np.zeros((2*nintnodes, 2*nnodes))
    Gin = np.zeros((2*nintnodes, 2*nnodes))
    qin = np.zeros(2*nintnodes)


    # Loop over internal points (ii) and elements (jj)
    for ii in range(nintnodes):
        x0, y0 = nodes_coord[node_int[ii]][0:2]  # Internal point coordinates

        for jj in range(nnodes//2):
            # Element geometry (starting/ending node coordinates and length)
            x1, y1 = nodes[2*jj,0:2]
            x2, y2 = nodes[2*jj+1,0:2]
            L = 2*np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            nx, ny = normal[2*jj]  # Normal vector components

            # Initialize integrals for this element
            intG = np.zeros((2,4))
            intH = np.zeros((2,4))
            intq = np.zeros(2)

            for kk in range(npgauss):
                # Integration point coordinates and shape functions
                N1 = 0.5 - xi[kk]
                N2 = 0.5 + xi[kk]
                N = np.array([[N1,0, N2,0],[0,N1,0,N2]])
                x = N1 * x1 + N2 * x2
                y = N1 * y1 + N2 * y2

                # Distances, fundamental solutions, and their derivatives
                rx, ry = x - x0, y - y0
                r = np.sqrt(rx**2 + ry**2)

                uast,tast= compute_fund_solutions(rx, ry, r, nx, ny, E, nu)

                # Update derivative integrals

                intG += uast.dot(N) * L / 2 * weight[kk]
                intH += tast.dot(N) * L / 2 * weight[kk]

                # Radial integration for q, dqx, dqy
                if(float(qpoint_str[0]) != 0.):
                    intF = np.zeros(2)
                    theta=np.arctan2(ry,rx)
                    for kkk in range(npgauss):
                        rho = r * (xi[kkk] + 1) / 2
                        x, y = x0 + rho * np.cos(theta), y0 + rho * np.sin(theta)

                        uast2,tast2 = compute_fund_solutions(rx,ry,r,nx,ny,E,nu)
                        f=np.array([0,0])
                        intF += uast2.dot(f) * rho * r / 2 * weight[kkk]

                    intq += intF * (nx * rx + ny * ry) / r**2 * L / 2 * weight[kk]

            # Update matrices/vectors for the current internal point and element

            Hin[2*ii:2*ii+2, 4*jj:4*jj+4] = intH
            Gin[2*ii:2*ii+2, 4*jj:4*jj+4] = intG
            qin[2*ii:2*ii+2] += intq


    # Compute displacement at internal points

    uint = - Hin.dot(u) + Gin.dot(t) - qin # Displacement

    return uint