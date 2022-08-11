'''
Kirchoff-Love shell formulation, for a square shell pinned at the edges 
subjected to a uniform vertical force. 

To run, specify the mesh refinement level (--ref ) :
python3 pinned_shell.py --ref 5

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 pinned_shell.py --ref 5

'''


import sys
sys.path.append("../")
from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np


petsc4py.init()
petsc4py.PETSc.Sys.popErrorHandler()

comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)

# to make the stabilization term work in parallel
parameters["ghost_mode"] = "shared_facet"

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ref',dest='ref',default='5',
                    help='Refinement level, integers in (4,6)')
args = parser.parse_args()
ref = args.ref

MESH_PATH = "../meshes/square/Quadratic/R"  + str(ref) + "/"

k = 2
dim = 3
# quad_deg = k*2
mesh_file_f = MESH_PATH+'mesh.xdmf'

h_th = Constant(0.1)  # shell thickness
E = Constant(4.8e5)  # Young's modulus
nu = Constant(0.38)  # Poisson's ratio
arealForceDensity = 90

LINEAR_SOLVER = 'mumps' #'gmres'
PRECONDITIONER = 'jacobi'

mesh_f = Mesh()
filename = mesh_file_f
file = XDMFFile(mesh_f.mpi_comm(),filename)
file.read(mesh_f)
#print(mesh_f.type().cell_type())
nsd = mesh_f.geometry().dim()

# print(">>> Creating subdomain MF...")
sub_domains = MeshFunction('size_t',mesh_f,nsd)
#print(">>> Reading in subdomain MF...")
file.read(sub_domains,'material')
#key: outside is 1, block is 2
block_ID = 2

sub_domains_surf = MeshFunction("size_t",mesh_f,nsd-1)

# print(">>> Creating surface subdomain MF...")
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

dx_custom = Measure('dx', subdomain_data=sub_domains, 
                     subdomain_id=block_ID, 
                     metadata={'quadrature_degree': k*2})

dS_custom = Measure('dS', subdomain_data=sub_domains_surf, 
                    subdomain_id=surf_ID, 
                    metadata={'quadrature_degree': k*2})
#define tracker point
middle = [0,0]

# visualize subdomains
# XDMFFile("MatSetCells.xdmf").write(sub_domains)
# XDMFFile("MatSetFacets.xdmf").write(sub_domains_surf)

###########################################################
######## Implement Kirchhoff--Love shell SVK model ########
###########################################################
V_f = VectorFunctionSpace(mesh_f, 'CG', k, dim=dim)
u_f = Function(V_f)
v_f = TestFunction(V_f)

xi = SpatialCoordinate(mesh_f)
X = as_vector([xi[0], xi[1], Constant(0.)])
x = X + u_f

V_mat = FunctionSpace(mesh_f, 'DG',0)
material_ex = Expression("1.0 - 1.0*(x[1] > (x[0] + 0.7071067811865476)) - 1.0*(x[1]<(x[0]-0.7071067811865476)) \
     - 1.0*(x[1] > (0.7071067811865476 - x[0]))*(x[1] < (x[0] + 0.7071067811865476))*(x[1]>(x[0]-0.7071067811865476))  \
        - 1.0*(x[1] < (-0.7071067811865476 - x[0]))*(x[1] < (x[0] + 0.7071067811865476))*(x[1]>(x[0]-0.7071067811865476)) ",\
             element=V_mat.ufl_element())
material_func = interpolate(material_ex,V_mat)




def unit(v):
    return v/sqrt(inner(v,v))

# Helper function to compute geometric quantities for a midsurface
# configuration x.
def shellGeometry(x):

    # Covariant basis vectors:
    dxdxi = grad(x)
    a0 = as_vector([dxdxi[0,0],dxdxi[1,0],dxdxi[2,0]])
    a1 = as_vector([dxdxi[0,1],dxdxi[1,1],dxdxi[2,1]])
    a2 = unit(cross(a0,a1))

    # Metric tensor:
    a = as_matrix(((inner(a0,a0),inner(a0,a1)),
                   (inner(a1,a0),inner(a1,a1))))
    # Curvature:
    deriva2 = grad(a2)
    b = -as_matrix(((inner(a0,deriva2[:,0]),inner(a0,deriva2[:,1])),
                    (inner(a1,deriva2[:,0]),inner(a1,deriva2[:,1]))))
    
    return (a0,a1,a2,a,b)

A0,A1,A2,A,B = shellGeometry(X)
a0,a1,a2,a,b = shellGeometry(x)

# Strain quantities.
epsilon = 0.5*(a - A)
kappa = B - b

def cartesian(T,a,a0,a1):
    
    # Raise the indices on the curvilinear basis to obtain contravariant
    # basis vectors a0c and a1c.
    ac = inv(a)
    a0c = ac[0,0]*a0 + ac[0,1]*a1
    a1c = ac[1,0]*a0 + ac[1,1]*a1

    # Perform Gram--Schmidt orthonormalization to obtain the local Cartesian
    # basis vector e0 and e1.
    e0 = unit(a0)
    e1 = unit(a1 - e0*inner(a1,e0))

    # Perform the change of basis on T and return the result.
    ea = as_matrix(((inner(e0,a0c),inner(e0,a1c)),
                    (inner(e1,a0c),inner(e1,a1c))))
    ae = ea.T
    return ea*T*ae

# Use the helper function to compute the strain quantities in local
# Cartesian coordinates.
epsilonBar = cartesian(epsilon,A,A0,A1)
kappaBar = cartesian(kappa,A,A0,A1)

# Helper function to convert a 2x2 tensor to voigt notation, following the
# convention for strains, where there is a factor of 2 applied to the last
# component.  
def voigt(T):
    return as_vector([T[0,0],T[1,1],2.0*T[0,1]])

# The material matrix:
D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                 [nu,   1.0,  0.0         ],
                                 [0.0,  0.0,  0.5*(1.0-nu)]])
# Extension and bending resultants:
nBar = h_th*D*voigt(epsilonBar)
mBar = (h_th**3)*D*voigt(kappaBar)/12.0

N = FacetNormal(mesh_f)
# splinedx = tIGArMeasure(volumeJacobian(A), dx_custom, quad_deg)
# splineds = tIGArMeasure(surfaceJacobian(A, N),
#                         dS_custom, quad_deg)

# Compute the elastic potential energy density
Wint = 0.5*(inner(voigt(epsilonBar),nBar)
            + inner(voigt(kappaBar),mBar))*dx_custom

dWint = derivative(Wint,u_f,v_f)

f = as_vector([Constant(0.0), Constant(0.0), arealForceDensity])
u_pre = Constant([0, 0, 0])
alpha_d = Constant(1e6)
h_mesh = CellDiameter(mesh_f)

int_constant = inner(Constant([0, 0, 0]),v_f)*dx(domain=mesh_f, 
                                                 subdomain_data=sub_domains)

#res = dWint - inner(f,v_f)*dx_custom \
#    + alpha_d*h_th*E/mesh_f.hmin()*inner(u_f("+")-u_pre, v_f("+"))*dS_custom \
#    + int_constant
res = dWint - inner(f,v_f)*dx_custom \
   + (alpha_d*h_th*E/h_mesh('+'))*inner(u_f("+")-u_pre, v_f("+"))*dS_custom \
   + int_constant


Dres = derivative(res,u_f)

######## Load extraction operator ########
fileNames = [MESH_PATH+"ExOp_Cons.csv"]
local_Size = v2p(assemble(res)).getLocalSize()
#fileNames = ["ExOp_Cons_Both.csv"]
# if k == 2:
nodeFileNames = MESH_PATH+'cell_nodes.csv'
# else: 
#     nodeFileNames = None 
# M = readFromCSV(fileNames, V_f, mesh_f, local_Size, 
#                 nodeFileNames=nodeFileNames, k=k)
M = readExOp(fileNames, V_f, mesh_f, local_Size, 
                            nodeFileNames=nodeFileNames, k=2, 
                            NFields=dim)
M.assemble()
#########################################

#print(">>> Creating linear system...")
# Transfer residual and linearization to background mesh
dR_b, R_b = assembleLinearSystemBackground(Dres, -res, M)

u_soln = dR_b.createVecRight()

#print(">>> Solving linear system...")
# bfr_tol = 1e-4
#solveKSP(dR_b, R_b, u_soln, method=LINEAR_SOLVER, PC=PRECONDITIONER, 
#         bfr_tol=None, monitor=False, rtol=1e-15)
solveNonlinear(res, u_f, M, u_soln, maxIters=10,\
                     linear_method=LINEAR_SOLVER,
                     monitorNewtonConvergence=False,
                     moniterLinearConvergence=False,
                     relativeTolerance=5e-4,relax_param=1.0,
                     absoluteTolerance=1e-4, absoluteToleranceRes=1e-5, du_0_mag = None)


transferToForeground(u_f, u_soln, M)
u_f.rename("u_f", "u_f")
#XDMFFile("result_u_f.xdmf").write(u_f)

'''
#option to visualize solution
d0File = File("results/disp-x.pvd")
d1File = File("results/disp-y.pvd")
d2File = File("results/disp-z.pvd")
matFile = File("results/mat.pvd")

(d0,d1,d2) = u_f.split()
d0.rename("d0","d0")
d1.rename("d1","d1")
d2.rename("d2","d2")
material_func.rename("material_func", "material_func")

d0File << d0
d1File << d1
d2File << d2
matFile << material_func
matFile << material_func    
'''

u_x= u_f(middle[0],middle[1])[0]
u_y = u_f(middle[0],middle[1])[1]
u_z = u_f(middle[0],middle[1])[2]

if rank ==0:
    print("Center displacement: (", u_x, ",", u_y, ",",u_z,")")

write_file = False
output_filename = '../pinned_shell_disp.csv'
if write_file: 
    if rank == 0:
        f = open(output_filename, 'a')
        print('writing to file')
        f.write("\n")
        #ref = os.getcwd()[-1]
        #size,time,L2,H1,N
        fs = str(ref) + ","+ str(u_x)+","+str(u_y) + ","+ str(u_z)
        f.write(fs)
        f.close()
