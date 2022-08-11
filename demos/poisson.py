'''
Implementation of 2D/3D Poisson's problem with unfitted mesh
and Nitsche's method to weakly apply the BCs

To run, specify the spatial dimension (--dim ), polynomial order (--k ) and refinement level (--ref ) :
python3 poisson.py --k 1 --ref 3 --dim 2 

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 poisson.py --k 1 --ref 3 --dim 3 


'''

from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np

petsc4py.init()

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
        retval = sin(pi*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))*cos(pi*(x[0] + x[1] + x[2]))
    return retval
     
def F(x, dim):
    return -div(grad(u_ex(x,dim)))
    
def interiorResidual(u,v,f,mesh,int_constant,dx_,ds_):
    n = FacetNormal(mesh)
    return inner(grad(u), grad(v))*dx_ \
            - inner(dot(grad(u("+")), n("+")), v("+"))*ds_ \
            - inner(f, v)*dx_ + int_constant
            
def boundaryResidual(u,v,u_exact,mesh,int_constant,ds_,
                        sym=True,
                        beta_value=0.1,
                        overPenalize=False,
                        h=None):

    '''
    Formulation from Github:
    https://github.com/MiroK/fenics-nitsche/blob/master/poisson/poisson_circle_dirichlet.py
    '''
    n = FacetNormal(mesh)
    if h is not None:
        h_E = h
    else:
        h_E = CellDiameter(mesh)("+")
    x = SpatialCoordinate(mesh)
    beta = Constant(beta_value)
    sgn = 1.0
    if (sym is not True):
        sgn = -1.0
    retval = sgn*inner(u_exact("+")-u("+"), dot(grad(v("+")), n("+")))*ds_ + int_constant
    penalty = beta*h_E**(-1)*inner(u("+")-u_exact("+"), v("+"))*ds_ + int_constant
    if (overPenalize or sym):
        retval += penalty
    return retval
    
####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree (1 or 2).')
parser.add_argument('--dim',dest='dimension',default=2,
                    help='Problem dimension (2 or 3).')
parser.add_argument('--ref',dest='ref',default='0',
                    help='Refinement level, integers in (0,6) for 2D, (0,4) for 3D')
parser.add_argument('--sym',dest='symmetric',default=True,
                    help='True for symmetric Nitsche; False for nonsymmetric')
parser.add_argument('--solv',dest='solv',default='gmres',
                    help='Linear solver')
parser.add_argument('--pc',dest='pc',default='jacobi',
                    help='Preconditioner for linear solver')       
parser.add_argument('--wf',dest='wf',default=False,
                    help='write output data to file')     
parser.add_argument('--of',dest='of',default='../poisson_data.csv',
                    help='Destination for output data') 
parser.add_argument('--Ex',dest='Ex',default=True,
                    help='Option to solve on the FG mesh ') 
args = parser.parse_args()

k = int(args.k)
dim = int(args.dimension)
Ex = args.Ex
symmetric = args.symmetric
ref = args.ref
write_file = args.wf
output_file = args.of
LINEAR_SOLVER = args.solv
PRECONDITIONER = args.pc

# Domain:
mesh_f = Mesh()
path = "../meshes/"
if dim == 2:
    path = path + 'square/'
elif dim == 3: 
    path = path + 'cube/'

if k == 1:
    path = path + 'Linear/R'
elif k == 2:
    path = path + 'Quadratic/R'
else: 
    print('Only linear and quadratic basis functions are currently supported')
    exit()

path = path + str(ref) 

filename = path + '/mesh.xdmf'
file = XDMFFile(mesh_f.mpi_comm(),filename)
file.read(mesh_f)
nsd = mesh_f.geometry().dim()
sub_domains =MeshFunction('size_t',mesh_f,nsd)
file.read(sub_domains,'material')
#key: outside is 1, block is 2
block_ID = 2

sub_domains_surf = MeshFunction("size_t",mesh_f,nsd-1)

#print(">>> Creating surface subdomain MF...")
for facet in facets(mesh_f):
    marker = 0
    for cell in cells(facet):
        marker = marker + sub_domains[cell]
    if marker == 1 or marker == 2:
        sub_domains_surf.set_value(facet.index(), 1)
    elif marker == 4: 
        sub_domains_surf.set_value(facet.index(), 2)
    elif marker == 3: 
        sub_domains_surf.set_value(facet.index(), 3)

surf_ID = 3

dx_custom = Measure('dx', subdomain_data=sub_domains,subdomain_id=block_ID, metadata={'quadrature_degree': k})
ds_custom =Measure('dS', subdomain_data=sub_domains_surf,subdomain_id=surf_ID, metadata={'quadrature_degree': k})

V_f = FunctionSpace(mesh_f, 'CG', k)
u_f = Function(V_f)
v_f = TestFunction(V_f)
x_f = SpatialCoordinate(mesh_f)

int_constant = Constant(0.0)*v_f*dx(domain=mesh_f, subdomain_data=sub_domains)
u_exact = u_ex(x_f,nsd)

f = F(x_f,nsd)
#as the block ID > the outside ID, this should make the positive cells the one within the block
local_Size = v2p(assemble(int_constant)).getLocalSize()

fileName = path + "/ExOp_Cons.csv"
fileNames = [fileName]

if k == 2:
    nodeFileNames =  path + '/cell_nodes.csv'
else: 
    nodeFileNames = None 

size = v2p(assemble(int_constant)).getSizes()
if not Ex:
    M = getIdentity(size)
else:
    M = readExOp(fileNames,V_f,mesh_f,local_Size,nodeFileNames=nodeFileNames,k=k)

M.assemble()



num_fg_dofs,num_bg_dofs = M.getSize()


OP = False
num_fg_dofs,num_bg_dofs = M.getSize()
ave_h = num_bg_dofs**(-k/nsd)
res_interior = interiorResidual(u_f, v_f, f, mesh_f,int_constant,dx_custom,ds_custom)
res_boundary = boundaryResidual(u_f, v_f, u_exact, mesh_f,int_constant,ds_custom,sym=symmetric,beta_value=10,overPenalize=OP,h=None)
res_f = res_interior + res_boundary


J_f = derivative(res_f, u_f)

local_Size = v2p(assemble(res_f)).getLocalSize()

# Transfer residual and linearization to background mesh
dR_b, R_b = assembleLinearSystemBackground(J_f, -res_f, M)

# Solve the linear system
u_p = dR_b.createVecLeft()
if dim ==3:
    # use a direct solver to avoid conditioning problems with 3D meshes
    LINEAR_SOLVER = 'mumps'
solveKSP(dR_b,R_b,u_p, method=LINEAR_SOLVER, PC=PRECONDITIONER)

# Transfer back to the foreground mesh for visualization and analysis 
transferToForeground(u_f, u_p, M)

# Compute errors 
h_E = CellDiameter(mesh_f)
e = u_f - u_exact
norm_L2 = assemble(inner(e, e)*dx_custom)
norm_H10 = assemble(inner(grad(e), grad(e))*dx_custom)
norm_edge = assemble(h_E("+")**-1*inner(e("+"), e("+"))*ds_custom)

L2 = assemble(inner(u_exact, u_exact)*dx_custom)
H10 = assemble(inner(grad(u_exact), grad(u_exact))*dx_custom)
edge = assemble(h_E("+")**-1*inner(u_exact("+"), u_exact("+"))*ds_custom)
H1 = L2 + H10 + edge

L2 = sqrt(L2)
H1 = sqrt(H1) 
H10 = sqrt(H10)

norm_H1 = norm_L2 + norm_H10  + norm_edge
norm_L2 = sqrt(norm_L2) / L2
norm_H1 = sqrt(norm_H1) / H1
norm_H10 = sqrt(norm_H10) / H10 

Nitsche_type = 'Symmetric Nitsche Method'
if (symmetric is not True):
    Nitsche_type = 'Nonsymmetric Nitsche Method'

if rank == 0:
    if write_file: 
        #ref = os.getcwd()[-1]
        f = open(output_file,'a')
        f.write("\n")
        fs = str(ref) +"," + str(norm_H10) +"," +  str(norm_L2) +","  + str(k)
        f.write(fs)
        f.close()
    print('-'*40)
    print('-'*5, Nitsche_type, '-'*5)
    print('-'*40)
    print('L2 norm:', norm_L2)
    print('H10 norm:', norm_H10)
    print('H1 norm:', norm_H1)
    print('-'*40)

###### Visualization ########
#import vedo.dolfin as vd
#vd.plot(mesh_f, u_exact)
#XDMFFile("result.xdmf").write(u_f)
#u_exact_proj = project(u_exact,V_f) 
#XDMFFile("exact.xdmf").write(u_exact_proj)
