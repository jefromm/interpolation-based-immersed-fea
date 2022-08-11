'''
Implementation of 2D/3D biharmonic problem with Nitsches method boundary conditions

To run, specify the spatial dimension (--dim ) and refinement level (--ref ) :
python3 biharmonic.py --ref 3 --dim 2 

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 biharmonic.py --ref 3 --dim 3 

'''

from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np
import os.path

comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)
# to make the stabilization term work in parallel
# Optimization options for the form compiler (from fenics demo)
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

def lap(x):
    return div(grad(x))

def F(x,dim):
    retval = lap(lap(u_ex(x,dim)))
    return retval

def u_ex(x, dim):
    retval = None
    if dim == 2:
        retval =  (cos(0.05*pi*x[0]+0.1))*(cos(0.05*pi*x[1]+0.1))
    elif dim == 3:
        retval = (cos(pi*x[0]+0.5))*(cos(pi*x[1]+0.5))*(cos(pi*x[2]+0.5))
    return retval



    

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dim',dest='dimension',default=2,
                    help='Problem dimension (2 or 3).')
parser.add_argument('--ref',dest='ref',default='3',
                    help='Refinement level, integers in (0,6) for 2D, (0,4) for 3D')
parser.add_argument('--sym',dest='symmetric',default=False,
                    help='True for symmetric Nitsche; False for nonsymmetric')
parser.add_argument('--solv',dest='solv',default='gmres',
                    help='Linear solver')
parser.add_argument('--pc',dest='pc',default='jacobi',
                    help='Preconditioner for linear solver')                  
parser.add_argument('--wf',dest='wf',default=False,
                    help='write output data to file')
parser.add_argument('--of',dest='of',default='../biharmonic_error.csv',
                    help='output data file')
parser.add_argument('--b',dest='beta_val',default=5,
                    help='Beta penalty ')
parser.add_argument('--a',dest='alpha_val',default=5,
                    help='alpha penalty')
parser.add_argument('--ft',dest='ft',default=1e-5,
                    help='cell volume filtering tolerance')                    

args = parser.parse_args()
dim = int(args.dimension)
ref = args.ref
k = 2
symmetric = args.symmetric

write_file = args.wf
ft = float(args.ft)
output_file = args.of
beta_value = args.beta_val
alpha_value = args.beta_val

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
dim = mesh_f.geometry().dim()

sub_domains =MeshFunction('size_t',mesh_f,dim)
file.read(sub_domains,'material')
outside_ID = 1
block_ID = 2
surf_ID = 3
sub_domains_surf = MeshFunction("size_t",mesh_f,dim-1)

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



#cell volume filter
c_vol = ft
returnCount=False
values = sub_domains.array()
f_values = sub_domains_surf.array()
vol_max = mesh_f.hmax()**dim
vol_limit = vol_max*c_vol
elim_count = 0
f_elim_count = 0
for cell in cells(mesh_f):
    if cell.volume() < vol_limit:
        if values[cell.index()] == block_ID:
            elim_count+=1
            sub_domains.set_value(cell.index(),0)
            for facet in facets(cell):
                if f_values[facet.index()] == surf_ID:
                    sub_domains_surf.set_value(facet.index(),0)
                    f_elim_count += 1
if rank==0 and returnCount:
    print( "number of cells eliminated: ", elim_count)
    print( "number of facets eliminated: ", elim_count)




dx_custom = Measure('dx', subdomain_data=sub_domains,subdomain_id=block_ID, metadata={'quadrature_degree': k})
ds_custom = Measure('dS', subdomain_data=sub_domains_surf,subdomain_id=surf_ID, metadata={'quadrature_degree': k})


V = FunctionSpace(mesh_f, 'CG', k)
u = Function(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh_f)
dom_constant= Constant(0)*v*dx(domain=mesh_f, subdomain_data=sub_domains)

u_exact = u_ex(x,dim)

f = F(x,dim)

n = FacetNormal(mesh_f)
h_E = CellDiameter(mesh_f)
beta = Constant(beta_value)
alpha = Constant(alpha_value)
sgn = 1.0
if (symmetric is not True):
    sgn = -1.0

A = inner(lap(u), lap(v))*dx_custom\
            - inner(lap(u("+")),dot(grad(v("+")),n("+")))*ds_custom \
            + inner(dot(grad(lap(u("+"))),n("+")),v("+"))*ds_custom \
            + sgn*(inner(dot(grad(lap(v("+"))),n("+")),(u("+"))))*ds_custom \
            - sgn*inner(lap(v("+")),(dot(grad(u("+")),n("+"))))*ds_custom \
            + beta*(h_E("+")**(-1))*inner((dot(grad(u("+")),n("+"))), (dot(grad(v("+")),n("+"))))*ds_custom \
            + alpha*(h_E("+")**(-3))*inner((u("+")), v("+"))*ds_custom + dom_constant

b = inner(f, v)*dx_custom \
    + sgn*(inner(dot(grad(lap(v("+"))),n("+")),(u_exact("+"))))*ds_custom \
    - sgn*inner(lap(v("+")),(dot(grad(u_exact("+")),n("+"))))*ds_custom \
    + beta*(h_E("+")**(-1))*inner((dot(grad(u_exact("+")),n("+"))), (dot(grad(v("+")),n("+"))))*ds_custom \
    + alpha*(h_E("+")**(-3))*inner(u_exact("+"), v("+"))*ds_custom + dom_constant
    
res = A - b

J = derivative(res, u)

local_Size = v2p(assemble(dom_constant)).getLocalSize()


#print("creating extraction operators")
start = default_timer()
fileName = path + "/ExOp_Cons.csv"
fileNames = [fileName]

nodeFileNames = path + '/cell_nodes.csv'
M = readExOp(fileNames,V,mesh_f,local_Size,nodeFileNames=nodeFileNames,k=k)
M.assemble()
stop = default_timer()
t_extract = stop-start

num_fg_dofs,num_bg_dofs = M.getSize()

# Transfer residual and linearization to background mesh
dR_b, R_b = assembleLinearSystemBackground(J, -res, M)
u_p = dR_b.createVecLeft()

start = default_timer()

#estimateConditionNumber(dR_b,R_b,u_p,PC=None,bfr_tol=None)

# Solve the linear system
start = default_timer()
u_petsc = solveNewtonsLinear(J,-b,u,M,u_p,maxIters=20,relax_param=1,linear_method='mumps',relativeTolerance=1e-12)
u_new = Function(V)
transferToForeground(u_new, u_petsc, M)

h_E = CellDiameter(mesh_f)
e = u_new - u_exact
norm_L2 = assemble(inner(e, e)*dx_custom)
norm_H10 = assemble(inner(grad(e), grad(e))*dx_custom)
norm_edge = assemble(h_E("+")**-1*inner(e("+"), e("+"))*ds_custom)
norm_H20 = assemble(inner(lap(e),lap(e))*dx_custom)

norm_H1 = norm_L2 + norm_H10  + norm_edge
norm_H2 = norm_L2 + norm_H10 + norm_edge + norm_H20
norm_L2 = sqrt(norm_L2)
norm_H1 = sqrt(norm_H1)
norm_H2 = sqrt(norm_H2)


# exact norms : 
L2 =  assemble(inner(u_exact, u_exact)*dx_custom)
H10 =  assemble(inner(grad(u_exact), grad(u_exact))*dx_custom)
edge = assemble(h_E("+")**-1*inner(u_exact("+"), u_exact("+"))*ds_custom)
H20 = assemble(inner(lap(u_exact),lap(u_exact))*dx_custom)

H1 = L2+H10+edge
H2 = H1 + H20

L2 = sqrt(L2)
H1 = sqrt(H1)
H2 = sqrt(H2)

norm_L2_r = norm_L2/L2
norm_H1_r = norm_H1/H1
norm_H2_r = norm_H2/H2

Nitsche_type = 'Symmetric Nitsche Method'
if (symmetric is not True):
    Nitsche_type = 'Nonsymmetric Nitsche Method'

if rank == 0:
    if write_file: 
        f = open(output_file, 'a')
        print('writing to file')
        #fs = "num_dofs,L2,H1,H2,L2r,H1r,H2r"
        f.write("\n")
        #ref = os.getcwd()[-1]
        fs = str(ref) + ","+ str(norm_L2_r)+","+str(norm_H1_r)+","+str(norm_H2_r)+","+str(c_tol)
        f.write(fs)
        f.close()
    print('-'*40)
    print('L2 norm:', norm_L2)
    print('H1 norm:', norm_H1)
    print('H2 norm:', norm_H2)
    print('relative L2 norm:', norm_L2_r)
    print('relative H1 norm:', norm_H1_r)
    print('relative H2 norm:', norm_H2_r)
    print('-'*40)
