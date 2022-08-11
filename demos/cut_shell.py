
'''
Demo with a Kirchoff-Love shell formulation to model a cut geometry subjected 
to a uniform follower load

To run, specify the mesh refinement level (--ref ) :
python3 cut_shell.py --ref 5

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 cut_shell.py --ref 5

'''

import sys
sys.path.append("../")
from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
'''
from tIGAr import *
from tIGAr.calculusUtils import *
from tIGAr.BSplines import *
from tIGAr.timeIntegration import *
'''

petsc4py.init()
petsc4py.PETSc.Sys.popErrorHandler()

comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)

# to make the stabilization term work in parallel
parameters["ghost_mode"] = "shared_facet"


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ref',dest='ref',default='3',
                    help='Refinement level, integers in (3,6)')
parser.add_argument('--lref',dest='lref',default='0',
                    help='Local refinement level, integers in (0,2)')
parser.add_argument('--of',dest='of',default='False',
                    help='Output result files')
args = parser.parse_args()
ref = args.ref
lref = args.lref
generate_files=args.of

k = 2
dim = 3
# quad_deg = k*2
path = "../meshes/bent_tab/FG_R" + str(lref) + "/R" + str(ref) 
mesh_file_f = path + '/mesh.xdmf'

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
outside_ID = 1
block_ID = 2
imm_boundary = 3
pinned_boundary = 4
free_boundary = 5

sub_domains_surf = MeshFunction("size_t",mesh_f,nsd-1)

# print(">>> Creating surface subdomain MF...")
for facet in facets(mesh_f):
    marker = 0
    c_count = 0 
    for cell in cells(facet):
        marker = marker + sub_domains[cell]
        c_count += 1
    
    if c_count == 1:
        #external boundary cell, check to see if it is on one of the pinned edges
        midpoint = facet.midpoint()
        point_x = midpoint[0]
        if math.isclose(point_x, -1, abs_tol=1e-6) or math.isclose(point_x, 1, abs_tol=1e-6):
            sub_domains_surf.set_value(facet.index(), pinned_boundary)
        else:
            sub_domains_surf.set_value(facet.index(), free_boundary)
    
    else:
        if marker == 1 or marker == 2:
            sub_domains_surf.set_value(facet.index(), outside_ID)
        elif marker == 4: 
            sub_domains_surf.set_value(facet.index(), block_ID)
        elif marker == 3: 
            sub_domains_surf.set_value(facet.index(), imm_boundary)

dx_custom = Measure('dx', subdomain_data=sub_domains, 
                     subdomain_id=block_ID, 
                     metadata={'quadrature_degree': k*2})

dS_custom = Measure('dS', subdomain_data=sub_domains_surf, 
                    subdomain_id=imm_boundary, 
                    metadata={'quadrature_degree': k*2})

ds_pinned = Measure('ds', subdomain_data=sub_domains_surf, 
                    subdomain_id=pinned_boundary, 
                    metadata={'quadrature_degree': k*2})


# define tracker points 
circle_tip = [0,-0.25]
corner_top_y = -1*sqrt(0.5**2 - 0.2**2)
wing_top_corner = [-0.2, corner_top_y]
wing_bottom_corner = [-0.2,-1]

# visualize subdomains
# XDMFFile("MatSetCells.xdmf").write(sub_domains)
#XDMFFile("MatSetFacets.xdmf").write(sub_domains_surf)

###########################################################
######## Implement Kirchhoff--Love shell SVK model ########
###########################################################

# define expression that is 1 on shell, and 0 outside it
def mat(X):
    tol = 1E-14
    x = X[0]
    y = X[1]
    r = np.sqrt(x**2 + y**2)

    material = 1 
    
    if r <= 0.5 or (-0.2 < x < 0.2 and y < 0 ):
        # in big circle, or cut away rectangle
        material = 0
        # account for small circle / rectangle
        if r < 0.25 or (-0.1 < x < 0.1 and y > 0):
            material = 1
        return material 


xi = SpatialCoordinate(mesh_f)

#material_ex = mat(xi)
V_mat = FunctionSpace(mesh_f, 'DG',0)
material_ex = Expression("1.0 - 1.0*((x[0]*x[0] + x[1]*x[1]) < 0.25) \
    - 1.0*(0.0>x[1])*(-0.2<x[0])*(x[0]<0.2)*((x[0]*x[0] + x[1]*x[1]) > 0.25) \
        + 1.0*((x[0]*x[0] + x[1]*x[1]) < 0.0625) \
        + 1.0*((x[0]*x[0]+x[1]*x[1]) > 0.0625)*((x[0]*x[0]+x[1]*x[1])< 0.25)*(0.0<x[1])*(-0.1<x[0])*(x[0]<0.1)",\
             element=V_mat.ufl_element())
material_func = interpolate(material_ex,V_mat)


V_f = VectorFunctionSpace(mesh_f, 'CG', k, dim=dim)
u_f = Function(V_f)
v_f = TestFunction(V_f)
QE = FiniteElement("Lagrange",mesh_f.ufl_cell(),k)
VE = MixedElement([QE,QE,QE])

V_f = FunctionSpace(mesh_f,VE)
v_f = TestFunction(V_f)
zero_vec = as_vector([Constant(0.0),Constant(0.0),Constant(0.0)])
dom_constant = inner(zero_vec,v_f)*dx(domain=mesh_f, subdomain_data=sub_domains)

F = as_vector([xi[0], xi[1], Constant(0.5)*(1.0-xi[0]**2)])
#




N = FacetNormal(mesh_f)

#replace functions previously grabbed from tIGAr (https://github.com/david-kamensky/tIGAr)
#g = getMetric(F)
DF = grad(F)
g =  DF.T*DF
#J_vol = volumeJacobian(g)
J_vol = sqrt(det(g))
#J_surf = surfaceJacobian(g,N)
J_surf = sqrt(det(g)*inner(N,inv(g)*N))

X = F
x = X + u_f

# starting config 



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

# The Young's modulus and Poisson ratio:
E = Constant(3e4)
nu = Constant(0.3)

# The shell thickness:
h_th = 0.03

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
            + inner(voigt(kappaBar),mBar))*J_vol*dx_custom

dWint = derivative(Wint,u_f,v_f)

f = as_vector([Constant(0.0), Constant(0.0), arealForceDensity])
u_pre = Constant([0, 0, 0])
alpha_d = Constant(1e5)

# External follower load magnitude:
PRESSURE = Constant(2e0)

# Divide loading into steps to improve nonlinear convergence.
N_STEPS = 100
#DELTA_T = 0.01
T_MAX = 1
DELTA_T = T_MAX/float(N_STEPS)
t = Constant(0.0)
#stepper = LoadStepper(DELTA_T)
u_f_old = Function(V_f)
u_f_dot_old = Function(V_f)
#stepper = BackwardEulerIntegrator(DELTA_T,u_f,(u_f_old,u_f_dot_old))
rho = Constant(1.0)
C_damp = Constant(1e1)
#res_unsteady = h_th*dot((rho*stepper.xddot() + C_damp*stepper.xdot()),v_f)*J_vol*dx_custom


#dWext = -(PRESSURE*stepper.t)*inner(a2,v_f)*dx_custom # (Per unit reference area)
dWext = -(PRESSURE*t)*inner(a2,v_f)*dx_custom # (Per unit reference area)
resPinn = alpha_d*E/mesh_f.hmin()*inner(u_f-u_pre, v_f)*J_surf*ds_pinned 

res = dWint + dWext + resPinn #+ res_unsteady

######## Load extraction operator ########
fileNames = [path+"/ExOp_Cons.csv"]
local_Size = v2p(assemble(dom_constant)).getLocalSize()
nodeFileNames = path+'/cell_nodes.csv'
M = readExOp(fileNames, V_f, mesh_f, local_Size, 
                            nodeFileNames=nodeFileNames, k=2, 
                            NFields=dim)


M.assemble()
'''
#get number of supported basis functions in the background mesh
a = assemble(inner(as_vector([Constant(1.0),Constant(1.0),Constant(1.0)]),v_f)*dx_custom)
a_vec = v2p(a)
a_bg_vec = AT_x(M,a_vec)
dof_count = 0
size = a_bg_vec.getSize()
for i in range(0,size):
    if a_bg_vec.getValue(i) > 1e-10:
        dof_count += 1
'''
#########################################
# Files for output:  Because an explicit B-spline is used, we can simply use
# the homogeneous (= physical) representation of the displacement in a
# ParaView Warp by Vector filter.

d0File = File("bent_shell_results/disp-x.pvd")
d1File = File("bent_shell_results/disp-y.pvd")
d2File = File("bent_shell_results/disp-z.pvd")
matFile = File("bent_shell_results/mat.pvd")

F0File = File("bent_shell_results/F-x.pvd")
F1File = File("bent_shell_results/F-y.pvd")
F2File = File("bent_shell_results/F-z.pvd")

Ffunc = project(F,V_f) # (Exact for quadratic F)
(F0,F1,F2) = Ffunc.split()
F0.rename("F0","F0")
F1.rename("F1","F1")
F2.rename("F2","F2")


circle_tip_disp = np.empty((N_STEPS,dim))
wing_top_corner_disp = np.empty((N_STEPS,dim))
wing_bottom_corner_disp = np.empty((N_STEPS,dim))


u_p = zeroDofBackground(M)

print(">>> Solving linear system...")
# bfr_tol = 1e-4
du_0_mag = None
t_num = 0.0
for i in range(0,N_STEPS):
    if(mpirank==0):
        print("------- Step: "+str(i+1)+" , t = "+str(t_num)+" -------")
    solveNonlinear(res, u_f, M, u_p, maxIters= 100,
                     linear_method='mumps',
                     zero_IDs=None, du_0_mag=du_0_mag)
    #
    t.assign(float(t)+1.0*float(DELTA_T))
    t_num += DELTA_T

    # Output solution.
    (d0,d1,d2) = u_f.split()
    print(d0)
    print(sub_domains)
    d0.rename("d0","d0")
    d1.rename("d1","d1")
    d2.rename("d2","d2")
    material_func.rename("material_func", "material_func")
    if generate_files:
        d0File << d0
        d1File << d1
        d2File << d2
        matFile << material_func
        F0File << F0
        F1File << F1
        F2File << F2    

    circle_tip_disp[i] = u_f(circle_tip[0],circle_tip[1])
    wing_top_corner_disp[i] = u_f(wing_top_corner[0],wing_top_corner[1])
    wing_bottom_corner_disp[i] = u_f(wing_bottom_corner[0],wing_bottom_corner[1])

u_f.rename("u_f", "u_f")
#XDMFFile("result_u_f.xdmf").write(u_f)

pd.DataFrame(circle_tip_disp, columns = ['d0','d1','d2']).to_csv('bent_shell_results/circle_tip.csv',index=False)
pd.DataFrame(wing_top_corner_disp, columns = ['d0','d1','d2']).to_csv('bent_shell_results/wing_top_corner.csv',index=False)
pd.DataFrame(wing_bottom_corner_disp, columns = ['d0','d1','d2']).to_csv('bent_shell_results/wing_bottom_corner.csv',index=False)

# (d0+F0-coordsX)*iHat + (d1+F1-coordsY)*jHat + (d2+F2-coordsZ)*kHat

u_x= u_f(circle_tip[0],circle_tip[1])[0]
u_y = u_f(circle_tip[0],circle_tip[1])[1]
u_z = u_f(circle_tip[0],circle_tip[1])[2]

if rank == 0 :
    print("Displacement at tip of tab: (", u_x, ",", u_y, ",",u_z,")")


output_filename = 'bent_shell_results/circle_tip_disp.csv'
if generate_files: 
    if rank == 0:
        f = open(output_filename, 'a')
        print('writing to file')
        f.write("\n")
        #ref = os.getcwd()[-1]
        #size,time,L2,H1,N
        fs = str(ref) + ","+ str(u_x)+","+str(u_y) + ","+ str(u_z)
        f.write(fs)
        f.close()
