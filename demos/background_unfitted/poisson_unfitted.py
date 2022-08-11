'''
Implementation of 2D Poisson's problem with a background-unfitted mesh
and Nitsche's method to weakly apply the BCs

To run, specify the polynomial order (--k ) and refinement level (--ref )/number of elements per edge (--n ):
python3 poisson_unfitted.py --k 1 --ref 3
or 
python3 poisson_unfitted.py --k 1 --n 24

'''
from time import time
from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
from timeit import default_timer
from matplotlib import pyplot as plt


comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)
# to make the stabilization term work in parallel
parameters["ghost_mode"] = "shared_facet"


def u_ex(x, dim):
    retval = None
    if dim == 2:
        retval = sin(pi*(x[0]*x[0] + x[1]*x[1]))*cos(pi*(x[0] - x[1]))
    elif dim == 3:
        retval = -sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])/(3*pi*pi)
    return retval
     
def F(x, dim):
    return -div(grad(u_ex(x,dim)))

def interior_A(u,v,f,mesh):
    n = FacetNormal(mesh)
    return inner(grad(u), grad(v))*dx \
            - inner(dot(grad(u), n), v)*ds 
            
def boundary_A(u,v,u_exact,mesh,
                        sym=True,
                        beta_value=8,
                        overPenalize=False):
    '''
    Formulation from Github:
    https://github.com/MiroK/fenics-nitsche/blob/master/poisson/poisson_circle_dirichlet.py
    '''
    n = FacetNormal(mesh)
    h_E = CellDiameter(mesh)
    beta = Constant(beta_value)
    sgn = 1.0
    if (sym is not True):
        sgn = -1.0
    retval = sgn*inner(-u, dot(grad(v), n))*ds
    penalty = beta*h_E**(-1)*inner(u, v)*ds
    if (overPenalize or sym):
        retval += penalty
    return retval
    
def interior_L(v,f,mesh):
    
    return inner(f, v)*dx
            
def boundary_L(v,u_exact,mesh,
                        sym=True,
                        beta_value=8,
                        overPenalize=False):
    '''
    Formulation from Github:
    https://github.com/MiroK/fenics-nitsche/blob/master/poisson/poisson_circle_dirichlet.py
    '''
    n = FacetNormal(mesh)
    h_E = CellDiameter(mesh)
    beta = Constant(beta_value)
    sgn = 1.0
    if (sym is not True):
        sgn = -1.0
    retval = -sgn*inner(u_exact, dot(grad(v), n))*ds
    penalty = -beta*h_E**(-1)*inner(-u_exact, v)*ds
    if (overPenalize or sym):
        retval += penalty
    return retval
####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n',dest='n',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--ref',dest='ref',default=-1,
                    help='Refinement level.')
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--sym',dest='symmetric',default=True,
                    help='True for symmetric Nitsche; False for nonsymmetric')
parser.add_argument('--of',dest='of',default='error_data_NC_Poisson.csv',
                    help='output file to write error data to')                    
args = parser.parse_args()

ref = float(args.ref)
if ref > -1 :
    Nel = int(4*2**ref)
else:
    Nel = int(args.n)

k = int(args.k)
dim = 2
symmetric = args.symmetric
output_filename = args.of

L_f = 2.
L_b = 4.

mesh_f, mesh_b = generateUnfittedMesh(L_f,L_b,Nel,Nel,dim=dim,rotate_f=True)


x_f = SpatialCoordinate(mesh_f)
u_exact = u_ex(x_f,dim)
f = F(x_f,dim)

####### Analysis #######
V_b = FunctionSpace(mesh_b, 'CG', k)
V_f = FunctionSpace(mesh_f, 'CG', k)
u_b = Function(V_b)
v_b = TestFunction(V_b)
u_f = Function(V_f)
v_f = TestFunction(V_f)
dx = dx(metadata={"quadrature_degree":2*k})

M = PETScDMCollection.create_transfer_matrix(V_b, V_f)

beta = 8
symmetric = False

A_f = interior_A(u_f, v_f, f, mesh_f) + boundary_A(u_f, v_f, u_exact, mesh_f, sym=symmetric, beta_value = beta)
L_f = interior_L(v_f, f, mesh_f) + boundary_L(v_f, u_exact, mesh_f, sym=symmetric, beta_value = beta)
res_f = A_f - L_f 

J_f = derivative(res_f, u_f)
M = arg2m(M)

# Transfer residual and linearization to background mesh
dR_b, R_b = assembleLinearSystemBackground(J_f, -res_f, M)

# Solve the linear system
u_p = dR_b.createVecLeft()
start = default_timer()


#print("solving linear system")
#smax,smin = estimateConditionNumber(dR_b,R_b,u_p)
#print('smin: ', smax)
#print('smax: ', smin)
solveKSP(dR_b,R_b,u_p, method='mumps', PC=None,monitor=True)



# Transfer the solution from background mesh to foreground mesh
transferToForeground(u_f, u_p, M)

h_E = CellDiameter(mesh_f)
e = u_f - u_exact
norm_L2 = assemble(inner(e, e)*dx)
norm_H10 = assemble(inner(grad(e), grad(e))*dx)
norm_edge = assemble(h_E**-1*inner(e, e)*ds)

norm_H1 = norm_L2 + norm_H10 + norm_edge
norm_L2 = sqrt(norm_L2)
norm_H1 = sqrt(norm_H1)
norm_H10 = sqrt(norm_H10)


Nitsche_type = 'Symmetric Nitsche Method'
if (symmetric is not True):
    Nitsche_type = 'Nonymmetric Nitsche Method'

if rank == 0:
    print('-'*40)
    print('-'*5, Nitsche_type, '-'*5)
    print('-'*40)
    print("Average mesh size of the foreground mesh = "
                        +str(averageCellDiagonal(mesh_f)))
    print('L2 norm:', norm_L2)
    print('H1 norm:', norm_H1)
    print('Nel:', Nel)
    print('-'*40)


write_file = False 

if write_file: 
    if rank == 0:
        f = open(output_filename, 'a')
        print('writing to file')
        f.write("\n")
        #N,L2,H1,beta,num_small_trimmed,rtol,k,grad_trim,mag_trim
        fs = str(Nel)+ ","+ str(norm_L2)+","+str(norm_H1)+',' \
            +str(beta)+','+str(k)
        f.write(fs)
        f.close()


###### Visualization ########
#import vedo.dolfin as vd
#vd.plot(mesh_f)
#vd.plot(mesh_b, title='mesh')
#vd.plot(u_f)




