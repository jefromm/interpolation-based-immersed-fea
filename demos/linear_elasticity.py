'''
Implementation of 2D linear elasticity of a plate with a hole 

To run, specify the polynomial order (--k ), refinement level (--ref ), and (if k=2) local refinement level (--lref) :
python3 linear_elasticity.py --k 2 --ref 3 --lref 1

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 linear_elasticity.py --k 2 --ref 3 --lref 1


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

# to make the stabilization term workhone  in parallel
parameters["ghost_mode"] = "shared_facet"


def exact(x,R,sig_inf,E,nu,x_origin,y_origin):
    #returns analytic solution in cartesian coordinates for Kirsch's problem 
    #https://www.researchgate.net/publication/339675835_On_Analytic_Solutions_of_Elastic_Net_Displacements_around_a_Circular_Tunnel
    tol= 0.0001
    x_shift = x[0] - x_origin
    y_shift = x[1] - y_origin
    r = sqrt(x_shift*x_shift + y_shift*y_shift)
    theta = atan(y_shift/x_shift)

    sig_rr = sig_inf*(1 - (R/(r+tol))**2)
    sig_tt = sig_inf*(1 + (R/(r+tol))**2)
    sig_polar = as_tensor([[sig_rr, 0],[0,sig_tt]])
    convert = as_tensor([[cos(theta), -sin(theta)],[sin(theta),cos(theta)]])
    convert_T = convert.T
    sig_cart = dot(dot(convert,sig_polar),convert_T)
    #sig_cart = dconvert[i,j]*sig_polar[j,k]*convert[l,k]

    I = Identity(2)
    eps_cart = (1/E)*((1+nu)*sig_cart - nu*tr(sig_cart)*I)

    C1 = (1 + nu)*(1 - 2*nu)*sig_inf/E
    C2 = (1 + nu)*R*R*sig_inf/E
    u_r = (C1*r) + C2/r
    u_polar = as_tensor([u_r,0])
    u_cart = dot(convert,u_polar)

    return sig_cart,eps_cart,u_cart

def problem(u,lam,mu):
    I = Identity(len(u))
    eps = sym(grad(u))
    sigma = Constant(2.0)*mu*eps+ lam*tr(eps)*I # c:eps
    psi = Constant(0.5)*inner(eps,sigma) # 0.5*eps:c:eps
    return sigma, psi 

     
####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--ref',dest='ref',default='0',
                    help='Refinement level, integers in (0,6)')
parser.add_argument('--lref',dest='lref',default='0',
                    help='Local refinement level, integers in (0,2), only available for k=2')                                    
parser.add_argument('--sym',dest='symmetric',default=True,
                    help='True for symmetric Nitsche; False for nonsymmetric')
parser.add_argument('--solv',dest='solv',default='gmres',
                    help='Linear solver')
parser.add_argument('--pc',dest='pc',default='jacobi',
                    help='Preconditioner for linear solver')    
parser.add_argument('--wf',dest='wf',default=False,
                    help='write output data to file')
parser.add_argument('--E',dest='E',default=200e9,
                    help='Youngs Modulus')
parser.add_argument('--nu',dest='nu',default=0.3,
                    help='Poissons ratio')
parser.add_argument('--of',dest='of',default='../error_data.csv',
                    help='Destination for output data') 

args = parser.parse_args()
k = int(args.k)
ref = args.ref
lref = args.lref
symmetric = args.symmetric
write_file = args.wf
E = float(args.E)
nu = float(args.nu)
LINEAR_SOLVER = args.solv
PRECONDITIONER = args.pc
output_file = args.of

hole_radius = 1
sig_inf = 1000000
center = [0,0]

lam = (E*nu)/((1+nu)*(1-nu))
K = E/(3*(1-2*nu)) 
mu = (3/2)*(K - lam)

# Domain:
mesh_f = Mesh()
path = "../meshes/hole_in_plate/"

if k == 1:
    path = path + 'Linear/R' + str(ref)
elif k == 2:
    path = path + 'Quadratic/FG_R' + str(lref) + '/R' + str(ref)
else: 
    print('Only linear and quadratic basis functions are currently supported')
    exit()

filename = path + '/mesh.xdmf'

file = XDMFFile(mesh_f.mpi_comm(),filename)
file.read(mesh_f)
#print(mesh_f.type().cell_type())
dim = mesh_f.geometry().dim()
#
#print(">>> Creating subdomain MF...")
sub_domains =MeshFunction('size_t',mesh_f,dim)
#print(">>> Reading in subdomain MF...")
file.read(sub_domains,'material')
#key: hole 1, plate is 2

# facet IDs: 
hole_ID = 1 
plate_ID = 2
rim_ID = 3
outer_edge_ID = 4
reflect_edge_left_ID = 5
reflect_edge_bottom_ID = 6
edge_top_ID = 7
edge_right_ID = 8

if k == 2:
    # correct the id of hole and plate, which are flipped for the quadratic 
    sub_domains_temp =MeshFunction('size_t',mesh_f,dim)
    for cell in cells(mesh_f):
        if sub_domains[cell] == 1:
            sub_domains_temp[cell] = 2
        elif sub_domains[cell] == 2:
            sub_domains_temp[cell] = 1

    sub_domains = sub_domains_temp


sub_domains_surf = MeshFunction("size_t",mesh_f,dim-1)


#print(">>> Creating surface subdomain MF...")
for facet in facets(mesh_f):
    marker = 0
    #edge sign: if the facet is only with one cell, it is a boundary facet, and the marker will be negative
    c_count = 1 
    for cell in cells(facet):
        marker = (np.abs(marker) + sub_domains[cell])*((-1)**c_count)
        c_count += 1
    if marker == 4: 
        #facet is in the plate
        sub_domains_surf.set_value(facet.index(), plate_ID)
    elif marker == 2 or marker == -1 :
        #facet is within the hole
        sub_domains_surf.set_value(facet.index(), hole_ID)
    elif marker == 3: 
        #facet is on the rim
        sub_domains_surf.set_value(facet.index(), rim_ID)
    elif marker == -2: 
        # facet is on the edge of the plate
        # check to see if this is a true boundary or line of symmetry, using coordinates of midpoint
        midpoint = facet.midpoint()
        point_x = midpoint[0]
        point_y = midpoint[1]
        if point_x == 0.0:
            #on lefthand side 
            sub_domains_surf.set_value(facet.index(), reflect_edge_left_ID)
        elif point_y == 0.0:
            # on bottom
            sub_domains_surf.set_value(facet.index(), reflect_edge_bottom_ID)
        elif point_y == 4.0:
            #on top
            sub_domains_surf.set_value(facet.index(), edge_top_ID)
        elif point_x == 4.0:
            #on top
            sub_domains_surf.set_value(facet.index(), edge_right_ID)



#print(">>> Defining regions")
dx_plate = Measure('dx', subdomain_data=sub_domains,subdomain_id=plate_ID, metadata={'quadrature_degree': k})
ds_top = Measure('ds', subdomain_data=sub_domains_surf,subdomain_id=edge_top_ID, metadata={'quadrature_degree': k})
ds_right = Measure('ds', subdomain_data=sub_domains_surf,subdomain_id=edge_right_ID, metadata={'quadrature_degree': k})
ds_bottom = Measure('ds', subdomain_data=sub_domains_surf,subdomain_id=reflect_edge_bottom_ID, metadata={'quadrature_degree': k})
ds_left = Measure('ds', subdomain_data=sub_domains_surf,subdomain_id=reflect_edge_left_ID, metadata={'quadrature_degree': k})
# the rim boundary needs to use 'dS' because the fenics mesh sees it as an internal boundary
ds_rim = Measure('dS',subdomain_data=sub_domains_surf,subdomain_id=rim_ID, metadata={'quadrature_degree': k})


V_mat = FunctionSpace(mesh_f, 'DG',0)
material_ex = Expression("1.0 - 1.0*((x[0]*x[0] + x[1]*x[1]) < 1)",\
             element=V_mat.ufl_element())
material_func = interpolate(material_ex,V_mat)
matFile = File("results/mat.pvd")
material_func.rename("material_func", "material_func")
matFile << material_func


# visualize subdomains
#XDMFFile("MatSetCells.xdmf").write(sub_domains)
#XDMFFile("MatSetFacets.xdmf").write(sub_domains_surf)

#print(">>> Creating function spaces and exact solutions...")

QE = FiniteElement("Lagrange",mesh_f.ufl_cell(),k)
VE = MixedElement([QE,QE])
V = FunctionSpace(mesh_f,VE)
u = Function(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh_f)
sigma_u,psi_u = problem(u,K,mu)
sigma_v,psi_v = problem(v,K,mu)


#print(">>> Defining residuals...")
# Nitsches formulation from 2017 Hansbo paper https://arxiv.org/pdf/1703.04377.pdf
beta = 10*mu  #- needs to be proportional to pressure 
n = FacetNormal(mesh_f)
h_E = CellDiameter(mesh_f)
#symmetric - sgn is positive, nonsymmetric- sgn is negative
sgn = 1 

sig_exact,eps_exact,u_exact = exact(x,hole_radius,sig_inf,E,nu,center[0],center[1])


A_h = inner(sigma_u, grad(v))*dx_plate

# weakly enforce traction boundary conditions- use exact form of stress
# traction on the rim is zero
ds_neumann = ds_top + ds_right
traction_form =  dot(dot(sig_exact,n),v)*ds_neumann

#weakly enforce symmetry conditions/ slip boundary 
ds_sym = ds_left + ds_bottom
g = Constant(0) 
nitsches_term = -sgn*dot(dot(dot(sigma_v,n),n),(dot(u,n)- g))*ds_sym - dot(dot(dot(sigma_u,n),n),dot(v,n))*ds_sym
penalty_term = beta*(h_E**(-1))*dot((dot(u,n)-g),dot(v,n))*ds_sym
L_h = traction_form 

res = A_h + nitsches_term + penalty_term - L_h 
J = derivative(res, u)


start = default_timer()

local_Size = v2p(assemble(res)).getLocalSize()
mesh_top = mesh_f.topology()

fileName = path + "/ExOp_Cons.csv"
fileNames = [fileName]

NFields = dim 
if k == 1:
    nodeFileNames =  None
else: 
    nodeFileNames =  path + '/cell_nodes.csv'

M = readExOp(fileNames,V,mesh_f,local_Size,nodeFileNames=nodeFileNames,k=k,NFields=NFields)

M.assemble()

stop = default_timer()
t_extract = stop-start


num_fg_dofs,num_bg_dofs = M.getSize()


#print(">>> Creating linear system...")
# Transfer residual and linearization to background mesh
dR_b, R_b = assembleLinearSystemBackground(J, -res, M)

# Solve the linear system
u_p = dR_b.createVecLeft()

#print(">>> Solving linear system...")
start = default_timer()
solveKSP(dR_b,R_b,u_p, method='mumps', PC=None,monitor=True)

stop = default_timer()
t_solve = stop-start
#print(">>> Transfering to foreground...")
# Transfer the solution from background mesh to foreground mesh
transferToForeground(u, u_p, M)


#print(">>> Computing errors...")


sigma_result,psi_result = problem(u,K,mu)
sigma_result_by_mat = sigma_result*material_func
V_s = TensorFunctionSpace(mesh_f, 'DG', k-1)
sig_result_proj = project(sigma_result_by_mat,V_s)
sig_exact_by_mat = sig_exact*material_func
sig_exact_proj = project(sig_exact_by_mat,V_s)
s1,s2,s3,s4 =sig_result_proj.split()


'''
#output files for visualization
s1File = File("results/s1.pvd")
s2File = File("results/s2.pvd")
s3File = File("results/s3.pvd")
s4File = File("results/s4.pvd")


s1.rename("s1","s1")
s2.rename("s2","s2")
s3.rename("s3","s3")
s4.rename("s4","s4")


s1File << s1
s2File << s2
s3File << s3
s4File << s4
'''

h_e = CellDiameter(mesh_f)
e = sigma_result - sig_exact
e_norm= assemble(inner(e, e)*dx_plate) / assemble(inner(sig_exact,sig_exact)*dx_plate)
ds_edges = ds_left + ds_bottom + ds_top + ds_top + ds_rim
norm= sqrt(e_norm)

Nitsche_type = 'Symmetric Nitsche Method'
if (symmetric is not True):
    Nitsche_type = 'Nonsymmetric Nitsche Method'


if rank == 0:
    if write_file: 
        #ref = os.getcwd()[-1]
        f = open(output_file,'a')
        f.write("\n")
        fs = str(ref) + ","+ str(norm)+","+str(t_solve)+","+str(t_extract)
        f.write(fs)
        f.close()

    print('-'*40)
    print('-'*5, Nitsche_type, '-'*5)
    print('-'*40)
    print('Time for creating M:', t_extract)
    print('Time for solve_linear:', t_solve)
    print('Extraction error norm:', norm)
    print('-'*40)


###### Visualization ########
#XDMFFile('sig_exact.xdmf').write(sig_exact_proj)
#XDMFFile('sig_result.xdmf').write(sig_result_proj)

#XDMFFile('u.xdmf').write(u)

u_file = File("u.pvd")
u.rename("u","u")
#u_file << u

V_mat = FunctionSpace(mesh_f, 'DG',0)
material_ex = Expression("1.0 - 1.0*((x[0]*x[0] + x[1]*x[1]) < 1)",\
             element=V_mat.ufl_element())
material_func = interpolate(material_ex,V_mat)
matFile = File("mat.pvd")
material_func.rename("material_func", "material_func")
#matFile << material_func
