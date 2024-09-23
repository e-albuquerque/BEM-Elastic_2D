import numpy as np
import meshio
import matplotlib.pyplot as plt

import sys
sys.path.append("../src")


import geometry
import boundcond
import graphics
import matrices


def input_data():
    """
    This function defines input parameters for the Boundary Element Method (BEM)
    simulation applied to the plane elasticity equation with radial integration for domain
    integrals.

    Returns:
        dict: A dictionary containing the following key-value pairs:
            * bound_cond (dict): Boundary conditions for each edge:
                *
            * E (float): Material elastic modulus
            * nu (float): Poisson's ratio
            * file_name (str): Name of the .msh file (Gmsh mesh)
            * qpoint (str): String representing the coordinates of the heat source point
    """
    # Directly define values here for scripting (modify as needed)
    # Boundary conditions are specified in local coordinate system (normal-tangent)
    # type_n = type of the boundary condition in normal direction
    # type_t = type of the boundary condition in tangent direction
    #    (type_n and type_t: 0 = displacement is know 1 = traction is known)
    bound_cond = {'fixed': {'type_n': 0, 'value_n': 0, 'type_t': 0, 'value_t': 0},
        'free': {'type_n': 1, 'value_n': 0, 'type_t': 1, 'value_t': 0},
        'loaded': {'type_n': 1, 'value_n': 1., 'type_t': 1, 'value_t': 0}
    }
    E = 1.0
    nu = 0.3
    file_name = '/workspaces/BEM-Elastic_2D/gmsh/plate'
    qpoint = '0.'  # Heat source

    return {'bound_cond': bound_cond, 'E': E, 'nu': nu, 'file_name': file_name,
            'qpoint': qpoint}
# Read information about input data
inp_data = input_data()

# Format input data
computed_data = geometry.compute_inodes(inp_data['file_name'], inp_data['bound_cond'])

# Compute the mid point of the elements and normal vector at this mid point
nodes, normal = geometry.comp_node_and_normal(computed_data['line_elements'], computed_data['coordinates'])

# Generate bcs array with boundary conditions in each element
bcs = boundcond.mount_bcs(computed_data['segments'], computed_data['bc_info'])


# Show geometry and boundary conditions
import matplotlib.pyplot as plt
nodes_coord = computed_data['coordinates']
graphics.show_problem(nodes,normal,nodes_coord,bcs,computed_data['triangle_elements'])


# Assembly H and G matrices
nodes_coord=computed_data['coordinates']

qpoint = inp_data['qpoint']
E=inp_data['E']
nu=inp_data['nu']
H, G, g = matrices.mount_matrices(nodes,normal,E,nu,qpoint)

T=matrices.transformation_matrix(normal)

Hnt=np.matmul(H,T) # H matrix in the local refence system
Gnt=np.matmul(G,T) # G matrix in the local refence system

# Assembly matrix A  and vector b
A, b = matrices.mount_linear_system(Hnt, Gnt, bcs)

# Solve the linear system
x = np.linalg.solve(A, b+g)

# Mount vectors u and t
unt, tnt = matrices.mount_vectors(x, bcs) # Displacements and tractions in the local reference system

nnodes=unt.shape[0] # Number of nodes
uvect=T.dot(unt.reshape(2*nnodes)) # Vector with diplacements in global reference system (lenght = 2nnodes)
tvect=T.dot(tnt.reshape(2*nnodes)) # Vector with tractions in global reference system (lenght = 2nnodes)

u=uvect.reshape(nnodes,2) # Matrix with diplacements in global reference system (shape = nnodes x 2)
t=tvect.reshape(nnodes,2) # Matrix with tractions in global reference system (shape = nnodes x 2)

elem = computed_data['line_elements'] # Conectivity matrix
int_nodes = computed_data['internal_nodes'] # Index of internal nodes
qpoint = inp_data['qpoint'] # Body force

# Compute displacements at internal points
uint_vet = matrices.int_point(int_nodes,normal,nodes,nodes_coord,E,nu,qpoint, uvect, tvect)

nint=uint_vet.shape[0]//2
uint=uint_vet.reshape(nint,2)

import matplotlib.tri as triang
from matplotlib import cm
# Generate the temperature color map
nodes_all=computed_data['all_nodes']

bound_nodes=computed_data["boundary_nodes"]

u_t=np.sqrt(u[:,0]**2+u[:,1]**2)
uint_t=np.sqrt(uint[:,0]**2+uint[:,1]**2)

title_fig = "Total displacement"
graphics.show_results(nodes_all,bound_nodes,int_nodes,nodes,elem,nodes_coord,u_t,uint_t,title_fig)

# Stress at internal points
stress1,stress2 = matrices.compute_stress(int_nodes,nodes, nodes_coord,normal,E,nu,uvect,tvect,qpoint)
sigmax=stress1[0::2]
sigmay=stress2[1::2]
tauxy =stress2[0::2]

# Tangent strain at boundary

epsilont = matrices.compute_epsilont(nodes, nodes_coord,normal,E,nu,unt[:,1],qpoint)

# Tangent stress at boundary

sigmat = matrices.compute_sigmat(E,nu,tnt[:,0],epsilont)

# von Mises stress at boundary

sigmaxy=matrices.compute_boundary_stresses(normal,tnt[:,0],sigmat,tnt[:,1])

vmstress_bound=matrices.compute_vonMises_stress(sigmaxy)

# von Mises stress at internal points

n_intnodes=sigmax.shape[0]
sigmaxy_int=np.zeros((n_intnodes,3))
sigmaxy_int[:,0]=sigmax
sigmaxy_int[:,1]=sigmay
sigmaxy_int[:,2]=tauxy
vmstress_int=matrices.compute_vonMises_stress(sigmaxy_int)

# Heat map of von Mises stress

title_fig = "Von Mises stress"
graphics.show_results(nodes_all,bound_nodes,int_nodes,nodes,elem,nodes_coord,vmstress_bound,vmstress_int,title_fig)