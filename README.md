# BEM-Elastic_2D
Boundary element method applied to linear elastic problems This is a python code for 2D problems. Elements used are the linear discontinuous (2 nodes per element). Singularities are analytically treated.


## Governing differential equations

Considering an infinitesimal element within a domain $\Omega$ and boundary $\Gamma$,
the balance of forces can be expressed by:

$$
\frac{\partial \sigma_x}{\partial x}+\frac{\partial \tau_{xy}}{\partial y}+\frac{\partial \tau_{xz}}{\partial z}+b_x=0
$$

$$
\frac{\partial \tau_{xy}}{\partial x}+\frac{\partial \sigma_y}{\partial y}+\frac{\partial \tau_{yz}}{\partial z}+b_y=0
$$

$$
\frac{\partial \tau_{xz}}{\partial x}+\frac{\partial \tau_{yz}}{\partial y}+\frac{\partial \sigma_{z}}{\partial z}+b_z=0
$$


that can be written as:

$$
\frac{\partial \sigma_{ij}}{\partial x_j}+b_i=0
$$

 or:

$$
\sigma_{ij,j}+b_{i}=0$$

## Integral equations

The integral equation for this problem is given by:

$$
c_{li} u_{i}+\int_\Gamma t^\star_{li}u_{i}d\Gamma=\int_{\Gamma}u^\star_{li}t_{i}d\Gamma +\iint_{\Omega}u^\star_{li}b_{i}d\Omega $$

where $u^\star_{li}$ and $t^\star_{li}$ are displacement and traction fundamental solutions, respectively, $b_i$ are domain forces like centrifugal or weight loads.

In the presence of domain loads, the domain integral is converted into a boundary integral using the radial integration method:

$$\iint_\Omega u_{li}^\star b_i dxdy=\oint_s F_l\frac{\bf n r}{R}ds$$

where

$$F_l=\int^R_0 u_{li}^\star b_i\rho d\rho $$

${\bf n}$ = normal unity vector

${\bf r}$ = radial unity vector

The following matrix equation is obtained:

$$ {\bf Hu=Gt+g}$$

and the linear system is given by:

$${\bf Ax=b+g}$$

The vector ${\bf g}$ becomes zero in the absence of body forces.
