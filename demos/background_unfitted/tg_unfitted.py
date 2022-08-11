'''
Implementation of 2D Taylor-Green vortex problem with unfitted mesh,
illustrating usage in unsteady problems and demonstrating 
spatio-temporal convergence under quasi-uniform space--time refinement.  

*utillizing the VMS formulation from VarMINT: https://github.com/david-kamensky/VarMINT


To run, specify the polynomial order (--k ) and refinement level (--ref )/number of elements per edge (--ref ):
python3 tg_unfitted.py --k 1 --ref 3
or 
python3 tg.py --k 1 --n 24

also supports parallel computation, for example with mpirun:
mpirun --np 2 python3 tg_unfitted.py --k 2 --ref 3



'''
from InterpolationBasedImmersedFEA.common import *
from InterpolationBasedImmersedFEA.profile_utils import profile_separate
from InterpolationBasedImmersedFEA.la_utils import *
from timeit import default_timer
from matplotlib import pyplot as plt
import ufl

comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)

####### Velocity solution #######

def u_IC(x):
    """
    Initial condition for the Taylor--Green vortex
    """
    return as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1])))
    
def u_ex(x, nu, t):
    solnT = exp(-2.0*nu*t)
    return solnT*u_IC(x)

def p_ex(x,nu,rho,t):
    return rho*0.25*exp(-4.0*nu*t)*(cos(2*x[1]) + cos(2*x[0]))
    #return rho*0.25*exp(-4.0*nu*t)*(cos(2*x[0]) + cos(2*x[1]))

#############################################################################
# Functions from VarMINT, for explanations and documentation please see 
# https://github.com/david-kamensky/VarMINT/blob/master/VarMINT.py

def sigmaVisc(u,mu):
    return 2.0*mu*sym(grad(u))

def sigma(u,p,mu):
    return sigmaVisc(u,mu) - p*Identity(ufl.shape(u)[0])

def stabilizationParameters(u,nu,G,C_I,C_t,Dt=None,scale=Constant(1.0)):
    
    # The additional epsilon is needed for zero-velocity robustness
    # in the inviscid limit.
    denom2 = inner(u,G*u) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
    if(Dt != None):
        denom2 += C_t/Dt**2
    tau_M = scale/sqrt(denom2)
    tau_C = 1.0/(tau_M*tr(G))
    return tau_M, tau_C

def materialTimeDerivative(u,u_t=None,f=None):
    
    DuDt = dot(u,nabla_grad(u))
    if(u_t != None):
        DuDt += u_t
    if(f != None):
        DuDt -= f
    return DuDt

def strongResidual(u,p,mu,rho,u_t=None,f=None):
    
    DuDt = materialTimeDerivative(u,u_t,f)
    i,j = ufl.indices(2)
    r_M = rho*DuDt - as_tensor(grad(sigma(u,p,mu))[i,j,j],(i,))
    r_C = rho*div(u)
    return r_M, r_C

def interiorResidual(u,p,v,q,rho,mu,mesh,G,
                     u_t=None,Dt=None,
                     f=None,
                     C_I=Constant(3.0),
                     C_t=Constant(4.0),
                     stabScale=Constant(1.0),
                     dx=dx):
    # Do a quick consistency check, to avoid a difficult-to-diagnose error:
    if((u_t != None) and (Dt==None)):
        print("VarMINT WARNING: Missing time step in unsteady problem.")
    if((Dt != None) and (u_t==None)):
        print("VarMINT WARNING: Passing time step to steady problem.")
    nu = mu/rho
    tau_M, tau_C = stabilizationParameters(u,nu,G,C_I,C_t,Dt,stabScale)
    DuDt = materialTimeDerivative(u,u_t,f)
    r_M, r_C = strongResidual(u,p,mu,rho,u_t,f)

    uPrime = -tau_M*r_M
    pPrime = -tau_C*r_C

    return (rho*inner(DuDt,v) + inner(sigma(u,p,mu),grad(v))
            + inner(div(u),q)
            - inner(dot(u,nabla_grad(v)) + grad(q)/rho, uPrime)
            - inner(pPrime,div(v))
            + inner(v,dot(uPrime,nabla_grad(u)))
            - inner(grad(v),outer(uPrime,uPrime))/rho)*dx


def stableNeumannBC(traction,rho,u,v,n,g=None,ds=ds,gamma=Constant(1.0),const = None):
    if(g==None):
        u_minus_g = u
    else:
        u_minus_g = u-g
    
    if const == None:
        return -(inner(traction,v)
             + gamma*rho*ufl.Min(inner(u,n),Constant(0.0))
             *inner(u_minus_g,v))*ds
    else: 
        return -(inner(traction,v)
             + gamma*rho*ufl.Min(inner(u,n),Constant(0.0))
             *inner(u_minus_g,v))*ds + const 


def weakDirichletBC(u,p,v,q,g,rho,mu,mesh,G,ds=ds,
                    sym=True,C_pen=Constant(1e3),
                    overPenalize=False):

    n = FacetNormal(mesh)
    sgn = 1.0
    if(not sym):
        sgn = -1.0

    traction = sigma(u,p,mu)*n
    consistencyTerm = stableNeumannBC(traction,rho,u,v,n,g=g,ds=ds)
    # Note sign of ``q``, negative for stability, regardless of ``sym``.
    adjointConsistency = -sgn*dot(sigma(v,-sgn*q,mu)*n,u-g)*ds
    penalty = C_pen*mu*sqrt(dot(n,G*n))*dot((u-g),v)*ds
    retval = consistencyTerm + adjointConsistency
    if(overPenalize or sym):
        retval += penalty
    return retval
#############################################################################



####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n',dest='n',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--ref',dest='ref',default=-1,
                    help='Refinement level.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')
parser.add_argument('--sym',dest='symmetric',default=True,
                    help='True for symmetric Nitsche; False for nonsymmetric')
parser.add_argument('--mf',dest='mf',default=None,
                    help='Import mesh file as the foreground mesh')
parser.add_argument('--mb',dest='mb',default=None,
                    help='Import mesh file as the background mesh')
parser.add_argument('--of',dest='of',default='error_data_NC_tg.csv',
                    help='output file to write error data to') 
                    
args = parser.parse_args()
ref = float(args.ref)
if ref > -1 :
    Nel = int(4*2**ref)
else:
    Nel = int(args.n)

Re = Constant(float(args.Re))
k = int(args.k)
T = float(args.T)
symmetric = args.symmetric
mesh_file_f = args.mf
mesh_file_b = args.mb
output_filename = args.of

####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2*k
dx = dx(metadata={"quadrature_degree":QUAD_DEG})
#print(">>> Generating mesh...")
L_f = 2*math.pi
L_b = 4*math.pi
N_f = N_b = Nel
mesh_f, mesh_b = generateUnfittedMesh(L_f,L_b,N_f,N_b,rotate_b=True)

# Mixed velocity--pressure space:
V_f = mixedScalarSpace(mesh_f,k=k)
V_b = mixedScalarSpace(mesh_b,k=k)

# Extraction matrix from the background to the foreground mesh:
M = PETScDMCollection.create_transfer_matrix(V_b, V_f)
M = arg2m(M)


# Solution and test functions:
up_f = Function(V_f)
vq_f = TestFunction(V_f)
u_0_f,u_1_f,p_f = split(up_f)
v_0_f,v_1_f,q_f = split(vq_f)

zero_vec = as_vector([Constant(0.0),Constant(0.0),Constant(0.0)])
dom_constant = inner(zero_vec,vq_f)*dx
size = v2p(assemble(dom_constant)).getSizes()
M = getIdentity(size)

u_f = as_vector([u_0_f,u_1_f])
v_f = as_vector([v_0_f,v_1_f])

# Midpoint time integration:
N_STEPS = Nel # Space--time quasi-uniformity
Dt = Constant(T/N_STEPS)

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
# Initial condition for the Taylor--Green vortex
up_exact_IC = as_vector((u_exact[0],u_exact[1],Constant(0.0)))

# Create zero PETSc vector for the background Dofs
up_p = zeroDofBackground(M)

# Project the initial condition:
L2Project(up_p, up_old_f, up_exact_IC, M)
up_f.assign(up_old_f)

# Use user-defined cell metric for the stabilization parameter
G_b = cellMetric(mesh_f)

# Weak problem residual; note use of midpoint velocity:
res_interior = interiorResidual(u_mid_f,p_f,v_f,q_f,rho,mu,mesh_f,
                     G=G_b,u_t=u_t_f,Dt=Dt,
                     C_I=Constant(6.0*(k**4)),
                     stabScale=Constant(1.0),
                     dx=dx)
res_boundary = weakDirichletBC(u_mid_f,p_f,v_f,q_f,u_exact,rho,mu,mesh_f,
                     G_b,sym=symmetric,C_pen=Constant(1e3),overPenalize=False)

res_f = res_interior + res_boundary

L2 = []
H1 = []

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
                     moniterLinearConvergence=False,
                     relativeTolerance=5e-4,relax_param=1.0,
                     absoluteTolerance=1e-4, absoluteToleranceRes=1e-5, du_0_mag = None, 
                     zero_IDs = None,estimateCondNum=False)
    up_old_f.assign(up_f)
    t.assign(float(t)+0.5*float(Dt))
    e_u = u_f - u_exact
    L2.append(L2Norm(e_u, dx))
    H1.append(L2Norm(grad(e_u), dx))
    


write_file = True 

norm_L2 = L2Norm(e_u, dx)
norm_H1 = L2Norm(grad(e_u), dx)

u0,u1,p = split(up_f)
u = as_vector([u0,u1])
ep = p - p_exact
grad_ep = grad(p - p_exact)
H1p = sqrt(assemble(inner(grad_ep,grad_ep)*dx))
L2p = sqrt(assemble(dot(ep,ep)*dx))

size = averageCellDiagonal(mesh_f)
if rank == 0 :
    print("Average mesh size of the foreground mesh = "
                        +str(size))
    print("H1 seminorm velocity error = "+str(norm_L2))
    print("L2 norm velocity error = "+str(norm_H1))
    print("H1 seminorm pressure error = "+str(H1p))
    print("Nel: ", Nel)


write_file = False 
if write_file: 
    if rank == 0:
        f = open(output_filename, 'a')
        print('writing to file')
        f.write("\n")
        ref = os.getcwd()[-1]
        #size,time,L2,H1,N
        fs = str(Nel) + ","+ str(norm_L2)+","+str(norm_H1) + ","+ str(L2p)+","+str(H1p)+','+str(Nel)+','+str(k)
        f.write(fs)
        f.close()

# Visualization of the solutions
#plt.figure()
#plot(mesh_f)
#plot(mesh_b)
#plt.figure()
#plot(u_f, title="velocity")
#plt.figure()
#plot(p_f, title="pressure")
#plt.show()
