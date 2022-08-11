'''
Implementation of 2D Taylor-Green vortex problem with unfitted mesh,
illustrating usage in unsteady problems and demonstrating 
spatio-temporal convergence under quasi-uniform space--time refinement.  

*using the VMS formulation from VarMINT : https://github.com/david-kamensky/VarMINT

To run, specify the polynomial order (--k ) and refinement level (--ref ) :
python3 tg_vortex.py --k 1 --ref 3 --dim 2 

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 tg_vortex.py --k 1 --ref 3 --dim 2 



'''
from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
import ufl

comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)

parameters["ghost_mode"] = "shared_facet"

####### Velocity solution #######

def u_IC(x):
    """
    Initial condition for the Taylor--Green vortex velocity field
    """
    x = x
    return as_vector((sin(1.0*x[0])*cos(1.0*x[1]),-cos(1.0*x[0])*sin(1.0*x[1])))
    
def u_ex(x, nu, t):
    """
    Exact Solution for the Taylor--Green vortex velocity field
    """
    solnT = exp(-2.0*nu*t)
    return solnT*u_IC(x)

def p_ex(x,nu,rho,t):
    """
    Exact Solution for the Taylor--Green vortex pressure field
    """
    return rho*0.25*exp(-4.0*nu*t)*(cos(2*1.0*x[1]) + cos(2*1.0*x[0]))

def weakDirichletBCIM(u,p,v,q,g,rho,mu,mesh,int_const,ds_,
                    G,
                    sym=True,C_pen=Constant(1e2),
                    overPenalize=False):
    n = FacetNormal(mesh)
    sgn = 1.0
    if(not sym):
        sgn = -1.0
    traction = sigma(u('+'),p('+'),mu)*n('+')
    #consistencyTerm = stableNeumannBCIM(traction,rho,u,v,n,ds_,int_const,g=g,)
    u_minus_g = u('+')-g('+')
    consistencyTerm =  -(inner(traction,v('+'))
             + rho*ufl.Min(inner(u('+'),n('+')),Constant(0.0))
             *inner(u_minus_g,v('+')))*ds_ + int_const


    # Note sign of ``q``, negative for stability, regardless of ``sym``.
    adjointConsistency = -sgn*dot(sigma(v('+'),-sgn*q('+'),mu)*n('+'),u_minus_g)*ds_ + int_const
    penalty = C_pen*mu*sqrt(dot(n('+'),G('+')*n('+')))*dot((u('+')-g('+')),v('+'))*ds_ + int_const
    #penalty = C_pen*mu*4*(h('+')**(-2))*dot(u_minus_g,v('+'))*ds_ + int_const
    retval = consistencyTerm + adjointConsistency 
    if(overPenalize or sym):
        retval += penalty
    return retval


def sigma(u,p,mu):
    return 2.0*mu*sym(grad(u)) - p*Identity(ufl.shape(u)[0])

def strongResidual(u,p,mu,rho,u_ad=None,u_t=None,f=None):
    DuDt = materialTimeDerivative(u,u_ad,u_t,f)
    i,j = ufl.indices(2)
    r_M = rho*DuDt - as_tensor(grad(sigma(u,p,mu))[i,j,j],(i,))
    r_C = rho*div(u)
    return r_M, r_C

def materialTimeDerivative(u,u_ad=None,u_t=None,f=None):
    if u_ad == None:
        u_ad = u
    DuDt = dot(u_ad,nabla_grad(u))
    if(u_t != None):
        DuDt += u_t
    if(f != None):
        DuDt -= f
    return DuDt

def interiorResidualIM(u,p,v,q,rho,mu,mesh, G,
                     u_t=None,Dt=None,
                     f=None,
                     C_I=Constant(3.0),
                     C_t=Constant(4.0),
                     stabScale=Constant(1.0),
                     dx=dx):
    if((u_t != None) and (Dt==None)):
        print("WARNING: Missing time step in unsteady problem.")
    if((Dt != None) and (u_t==None)):
        print("WARNING: Passing time step to steady problem.")

    nu = mu/rho
    # for Oseen Eq: make u in stab parameters the exact soln
    tau_M, tau_C = stabilizationParameters(u,nu,G,C_I,C_t,Dt,stabScale)
    #DuDt = materialTimeD8erivative(u,u_ad,u_t,f)
    DuDt = materialTimeDerivative(u,u_ad=u,u_t=u_t,f=f)
    r_M, r_C = strongResidual(u,p,mu,rho,u_ad=u,u_t=u_t,f=f)

    uPrime = -tau_M*r_M
    pPrime = -tau_C*r_C

    return (rho*inner(DuDt,v) + inner(sigma(u,p,mu),grad(v))
            + inner(div(u),q)
            - inner(dot(u,nabla_grad(v)) + grad(q)/rho, uPrime)
            - inner(pPrime,div(v))
            + inner(v,dot(uPrime,nabla_grad(u)))
            - inner(grad(v),outer(uPrime,uPrime))/rho)*dx

def stabilizationParameters(u,nu,G,C_I,C_t,Dt=None,scale=Constant(1.0)):
    """
    Compute SUPS and LSIC/grad-div stabilization parameters (returned as a 
    tuple, in that order).  Input parameters are the velocity ``u``, the
    kinematic viscosity ``nu``, the mesh metric ``G``, order-one constants
    ``C_I`` and ``C_t``, a time step ``Dt`` (which may be omitted for
    steady problems), and a scaling factor that defaults to unity.  
    """
    # The additional epsilon is needed for zero-velocity robustness
    # in the inviscid limit.
    denom2 = inner(u,G*u) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
    if(Dt != None):
        denom2 += C_t/Dt**2
    tau_M = scale/sqrt(denom2)
    tau_C = 1.0/(tau_M*tr(G))
    return tau_M, tau_C


####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree (1 or 2).')
parser.add_argument('--ref',dest='ref',default='0',
                    help='Refinement level, integers in (0,6) for 2D')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')
parser.add_argument('--sym',dest='symmetric',default=False,
                    help='True for symmetric Nitsche; False for nonsymmetric')
parser.add_argument('--wf',dest='wf',default=False,
                    help='write output to file')                     
parser.add_argument('--of',dest='of',default='../error_data_tg.csv',
                    help='output file to write error data to') 

                    
args = parser.parse_args()
k = int(args.k)
ref = args.ref
Re_num = float(args.Re)
Re = Constant(float(args.Re))
k = int(args.k)
T = float(args.T)
symmetric = args.symmetric
write_file = args.wf
output_filename = args.of

####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 3*k
dx = dx(metadata={"quadrature_degree":QUAD_DEG})
#print('running')
# Domain:

# Domain:
mesh_f = Mesh()
path = "../meshes/square/"

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
#print(">>> Reading in subdomain MF...")
file.read(sub_domains,'material')
#key: outside is 1, block is 2
block_ID = 2



sub_domains_surf = MeshFunction("size_t",mesh_f,nsd-1)
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


dx_custom = Measure('dx', subdomain_data=sub_domains,subdomain_id=block_ID, metadata={'quadrature_degree': QUAD_DEG})

ds_custom =Measure('dS', subdomain_data=sub_domains_surf,subdomain_id=surf_ID, metadata={'quadrature_degree': QUAD_DEG})



# Mixed velocity--pressure space:
cell = mesh_f.ufl_cell()
QE = FiniteElement("Lagrange",mesh_f.ufl_cell(),k)
VE = MixedElement([QE,QE,QE])
V_f = FunctionSpace(mesh_f,VE)


# Solution and test functions:
up_f = Function(V_f)
vq_f = TestFunction(V_f)
u_0_f,u_1_f,p_f = split(up_f)
v_0_f,v_1_f,q_f = split(vq_f)
u_f = as_vector([u_0_f,u_1_f])
v_f = as_vector([v_0_f,v_1_f])


zero_vec = as_vector([Constant(0.0),Constant(0.0),Constant(0.0)])
dom_constant = inner(zero_vec,vq_f)*dx(domain=mesh_f, subdomain_data=sub_domains)

NFields=3
local_Size = v2p(assemble(dom_constant)).getLocalSize()
if k == 1:
    nodeFileNames = None
else: 
    nodeFileNames = path + '/cell_nodes.csv'


fileName = path + "/ExOp_Cons.csv"
fileNames = [fileName]

M = readExOp(fileNames,V_f,mesh_f,local_Size,nodeFileNames=nodeFileNames,k=k,NFields=NFields)
M.assemble()

# Midpoint time integration:
N = sqrt(mesh_f.num_entities_global(2))

Dt_approx = 4/N 
N_STEPS = int(np.ceil(T/Dt_approx))

Dt = Constant(T/N_STEPS) # use this so that we end at the correct time regardless of N 


up_old_f = Function(V_f)
u_0_old_f,u_1_old_f,_ = split(up_old_f)
u_old_f = as_vector([u_0_old_f,u_1_old_f])
u_mid_f = 0.5*(u_f+u_old_f)
u_t_f = (u_f-u_old_f)/Dt

# Determine physical parameters from Reynolds number:
rho = Constant(1.0)
mu = Constant(1.0/Re)
nu = mu/rho

# Time dependence of exact solution, evaluated at time T:
t = Constant(0.0)
x_f = SpatialCoordinate(mesh_f)
u_exact = u_ex(x_f,nu,t)
p_exact = p_ex(x_f,nu,rho,t)
# Create zero PETSc vector for the background Dofs
up_p = zeroDofBackground(M)
up_exact_IC = as_vector((u_exact[0],u_exact[1],Constant(0.0)))

# Project the initial condition:
L2Project(up_p, up_old_f, up_exact_IC, M,dx_custom)
num_fg_dofs,num_bg_dofs = M.getSize()
ave_h = num_bg_dofs**(-k/nsd)

u0,u1,p = split(up_old_f)
u_old_f = as_vector([u0,u1])

# Use user-defined cell metric for the stabilization parameter
G_b = (4*ave_h**(-2))*as_tensor([[1,0],[0,1]])
C_I_num = 60
C_I = Constant(C_I_num) 

# Weak problem residual; note use of midpoint velocity:
res_interior = interiorResidualIM(u_mid_f,p_f,v_f,q_f,rho,mu,mesh_f,
                     u_ad = u_mid_f,
                     G=G_b,u_t=u_t_f,Dt=Dt,
                     C_I=C_I,
                     C_t=Constant(4),
                     stabScale=Constant(1.0),
                     dx=dx_custom)
C_pen = Constant(10)
res_boundary = weakDirichletBCIM(u_mid_f,p_f,v_f,q_f,u_exact,rho,mu,mesh_f,dom_constant,ds_custom,
                     G=G_b,sym=symmetric,C_pen=C_pen,overPenalize=False)

res_f = res_interior + res_boundary

# Time stepping loop:
for step in range(0,N_STEPS):
    if rank == 0:
        print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")
    t.assign(float(t)+0.5*float(Dt))
    #solveNonlinear(res_f, up_f, M, up_p, relativeTolerance=1e-5,
    #                monitorNewtonConvergence=True)
    solveNonlinear(res_f, up_f, M, up_p, maxIters=10, bfr_tol=None, \
                     linear_method='mumps',
                     monitorNewtonConvergence=True,
                     moniterLinearConvergence=True,
                     relativeTolerance=5e-4,relax_param=1.0,
                     absoluteTolerance=1e-4, absoluteToleranceRes=1e-5, du_0_mag = None, 
                     zero_IDs = None)
    up_old_f.assign(up_f)
    t.assign(float(t)+0.5*float(Dt))
    e_u = u_f - u_exact
    

norm_L2 = L2Norm(e_u, dx_custom)
norm_H1 = L2Norm(grad(e_u), dx_custom)

u0,u1,p = split(up_f)
u = as_vector([u0,u1])
ep = p - p_exact
grad_ep = grad(p - p_exact)
H1p = sqrt(assemble(inner(grad_ep,grad_ep)*dx_custom))
L2p = sqrt(assemble(dot(ep,ep)*dx_custom))

if rank == 0:
    if write_file: 
        f = open(output_filename, 'a')
        print('writing to file')
        f.write("\n")
        #ref = os.getcwd()[-1]
        #size,time,L2,H1,N
        fs = str(ref) + ","+ str(norm_L2)+","+str(norm_H1) \
            + ","+ str(L2p)+","+str(H1p)+','+str(k)+ ","+ fileNames[0] \
                + ","+ str(Re_num) \
                    + ","+ str(N_STEPS)
        f.write(fs)
        f.close()

    print('-'*40)
    print("L2 velocity error: ", norm_L2)
    print("H1 velocity error: ", norm_H1)
    print("L2 pressure error: ", L2p)
    print("H1 pressure error: ", H1p)
    print('-'*40)

# Visualization of the solutions
#plt.figure()
#plot(mesh_f)
#plot(mesh_b)
#plt.figure()
#plot(u_f, title="velocity")
#plt.figure()
#plot(p_f, title="pressure")
#plt.show()
