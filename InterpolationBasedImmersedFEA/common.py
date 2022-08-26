"""
The ``common`` module 
---------------------
contains basic functions for generating extraction data 
and importing it again for use in analysis. 
"""
from encodings import normalize_encoding
from InterpolationBasedImmersedFEA.la_utils import *
import os
import math
import numpy as np
from scipy.stats import mode
from mpi4py import MPI as pyMPI
import sys
import petsc4py
from petsc4py import PETSc 
import h5py



INFO = LogLevel.INFO
set_log_level(INFO)
parameters['std_out_all_processes'] = False

worldcomm = MPI.comm_world
mpirank = MPI.rank(worldcomm)
mpisize = MPI.size(worldcomm)

DOLFIN_FUNCTION = function.function.Function
DOLFIN_VECTOR = cpp.la.Vector
DOLFIN_MATRIX = cpp.la.Matrix
DOLFIN_PETSCVECTOR = cpp.la.PETScVector
DOLFIN_PETSCMATRIX = cpp.la.PETScMatrix
PETSC4PY_VECTOR = PETSc.Vec
PETSC4PY_MATRIX = PETSc.Mat
DEFAULT_LINEAR_SOLVER = 'gmres'


# file naming conventions
EXTRACTION_DATA_FILE = "extraction-data.h5"
EXTRACTION_INFO_FILE = "extraction-info.txt"
EXTRACTION_H5_MESH_NAME = "/mesh_f"
def EXTRACTION_H5_CONTROL_FUNC_NAME(dim):
    return "/control"+str(dim)
#def EXTRACTION_ZERO_DOFS_FILE(proc):
#    return "/zero-dofs"+str(proc)+".dat"
EXTRACTION_ZERO_DOFS_FILE = "zero-dofs.dat"
EXTRACTION_MAT_FILE = "extraction-mat.dat"
EXTRACTION_MAT_FILE_CTRL = "extraction-mat-ctrl.dat"



def generateUnfittedMesh(L_f,L_b,N_f,N_b,dim=2,
                        rotate_f=False,rotate_b=False,angle=30):
    """
    Generate simple unfitted foreground and background meshes, with the 
    forground/background mesh rotated to make the boundary cells
    cut by the background/foreground cell edges.
    
    Parameters
    ----------
    L_f, L_b: lengths of the edges
    N_f, N_b: numbers of elements on each edge
    dim: dimension of the meshes; could either be 3D or 2D.
    rotate_f, rotate_b: should be set to True for the mesh to be rotated
    
    Returns
    -------
    mesh_f, mesh_b
    """
    if dim == 2:
        mesh_f = RectangleMesh(Point(-L_f/2.,-L_f/2.), 
                                Point(L_f/2.,L_f/2.), N_f, N_b)
        mesh_b = RectangleMesh(Point(-L_b/2.,-L_b/2.), 
                                Point(L_b/2.,L_b/2.), N_b, N_b)
        if rotate_f == True:
            mesh_f.rotate(angle, 2, Point(0.,0.))
        if rotate_b == True:
            mesh_b.rotate(angle, 2, Point(0.,0.))
    elif dim == 3:
        mesh_b = BoxMesh(Point(-L_b/2.,-L_b/2.,-L_b/2.), 
                                Point(L_b/2.,L_b/2.,L_b/2.), N_b, N_b, N_b)
        mesh_f = BoxMesh(Point(-L_f/2.,-L_f/2.,-L_f/2.), 
                                Point(L_f/2.,L_f/2.,L_f/2.), N_f, N_f, N_f)
        if rotate_f == True:
            mesh_f.rotate(angle, 2, Point(0.,0.,0.))
            mesh_f.rotate(angle, 1, Point(0.,0.,0.))
        if rotate_b == True:
            mesh_b.rotate(angle, 2, Point(0.,0.,0.))
            mesh_b.rotate(angle, 1, Point(0.,0.,0.))
    else: 
        raise ValueError("Dimension of"+dim+"is not supported!")
    return mesh_f, mesh_b
    

def mixedScalarSpace(mesh,k=1):
    """
    Generate the most common choice of equal-order velocity--pressure 
    function space to use on a given ``mesh``.  The polynomial degree 
    ``k`` defaults to ``1``.  
    
    Return
    ------
    VQE: mixed function space with three scalar subspaces, 
        two for velocity in x and y direction, and one for pressure.
    """
    VE = FiniteElement("Lagrange",mesh.ufl_cell(),k)
    QE = FiniteElement("Lagrange",mesh.ufl_cell(),k)
    VQE = MixedElement([VE,VE,QE])
    return FunctionSpace(mesh,VQE)
    
def averageCellDiagonal(mesh):
    v0 = interpolate(Constant(1.0), FunctionSpace(mesh, 'DG', 0))
    total_area = assemble(v0*dx)
    num_cells = MPI.sum(MPI.comm_world, mesh.num_cells())
    average_cell_area = total_area/num_cells
    h_avg = sqrt(average_cell_area*4)
    return h_avg

def zeroDofBackground(M):
    return arg2m(M).createVecRight()
    
def transferToForeground(u_f, u_b, M):
    """
    Transfer the solution vector from the background to the forground
    mesh.
    
    Parameters
    ----------
    u_f: Dolfin function on the foreground mesh
    u_b: Dolfin function on the background mesh or PETSc vector
    M: extraction matrix from background to foreground.
    """
    M_petsc = arg2m(M)
    #print(M_petsc.getSize())
    #print(arg2v(u_b).getSize())
    #print(arg2v(u_f).getSize())
    M_petsc.assemble()
    M_petsc.mult(arg2v(u_b), arg2v(u_f))
    updateU(u_f)

def  assembleLinearSystemBackground(a_f, L_f, M):
    """
    Assemble the linear system on the background mesh, with
    variational forms defined on the foreground mesh.
    
    Parameters
    ----------
    a_f: LHS form on the foreground mesh
    L_f: RHS form on the foreground mesh
    M: extraction matrix
    
    Returns
    -------  
    A_b: PETSc matrix on the background mesh
    b_b: PETSc vector on the background mesh
    """
    A_f = assemble(a_f)
    b_f = assemble(L_f)
    A_b = AT_R_A(M, m2p(A_f))
    b_b = AT_x(M, b_f)

    return A_b, b_b
    
    
def L2Norm(u, dx):
    """
    $L^2$ norm of a Dolfin function u with user-defined quadrature
    """
    return math.sqrt(assemble(inner(u,u)*dx))
    
def L2Project(u_p, u_f, expression_f, M,dx_= None,bfr_tol=None):
    """
    Project the UFL expression of the initial condition
    onto the function spaces of the foreground mesh and 
    the background mesh, to make u_f = M*u_p.
    
    Parameters
    -----------
    u_p: PETSc Vector representing the dofs on the background mesh
    u_f: Dolfin Function on the foreground mesh
    expression_f: initial condition expression on the foreground
    M: extraction matrix from the background to the foreground
    """
    if dx_ == None:
        dx_ = dx
    V_f = u_f.function_space()
    u_f_0 = TrialFunction(V_f)
    w_f = TestFunction(V_f)
    a_f = inner(u_f_0, w_f)*dx_
    L_f = inner(expression_f, w_f)*dx_
    A_b, b_b = assembleLinearSystemBackground(a_f,L_f,M)
    #solveLinear(A_b,u_p,b_b,solver=DEFAULT_LINEAR_SOLVER)
    solveKSP(A_b,b_b,u_p,monitor=False,bfr_tol=bfr_tol)
    transferToForeground(u_f, u_p, M)

def cellMetric(mesh):
    """
    Define a custom mesh tensor 'G' used in the SUPG and LSIC/grad-div
    stabilization terms
    """
    # the maximum cell size of the foreground mesh is the same
    # as the average size of the background mesh generated by XTK
    h = mesh.hmax()
    return Constant(4.0/h**2)*Identity(mesh.geometry().dim())

def createNonzeroDiagonal(A, bfr_tol=1E-9):
    """
    Create a nonzero diagonal vector for the matrix A, by adding ones
    to the diagonal entries which used to be zeros
    
    Parameter
    ---------
    A: PETSc matrix
    
    Return
    ------
    D: PETSc vector 
    """
    #print(bfr_tol)
    D = A.createVecLeft()
    A.getDiagonal(D)
    Istart, Iend = D.getOwnershipRange()
    nz = 0
    nz_id = []
    for ind in range(Istart, Iend):
        old_val = D.getValue(ind)
        if abs(old_val) <= bfr_tol:
            D.setValue(ind, 1.0)
            nz_id.append(ind)
            nz += 1
        else: D.setValue(ind, 0.0)
    return D


def removeZeroDiagonal(A,bfr_tol=1E-9):
    """
    Changing the zeros to be ones in the diagonal of the matrix
    """

    vd = createNonzeroDiagonal(A,bfr_tol=bfr_tol)
    A0 = PETSc.Mat().create(worldcomm) 
    A0.setSizes(A.getSizes())  
    A0.setUp()
    A0.setFromOptions()
    A0.setDiagonal(vd)
    A0.assemble()
    A += A0
    A.assemble()

    return A


def getIdentity(size):
    (size_l, size_g) = size
    A = zero_petsc_mat(size_g, size_g,row_loc=size_l, col_loc=size_l)
    A = removeZeroDiagonal(A)
    return A


def trimNodes(A,b=None,bfr_tol=1E-9,target=None,zero_vec=None,monitor=False):
    """
    Identifies zeros or 'small' entries in the matrix diagonal, 
    and replaces the diagonal value with one and all other row entries with zero,
    also zeros the corresponding RHS vector value 

    Used in optional basis function removal to improve linear system conditioning during debugging

    For nonlinear Newton's method: 
    - target option: instead of zeroing the the RHS vector, 
    it will set the residual to the value of u , where:
    b = res = a(target) - L,
    A = J = da/du 
    and we are solving 
    J du = res 
    and wish du = target at the nodes identified for BFR

    """
    A.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS,True)

    A.assemble()

    nz_val = 0
    nz_id = []

    if zero_vec is not None:
        A.zeroRows(zero_vec)
        A.assemble()
        if b is not None:
            for i in zero_vec:
                if target is not None:
                    val = target.getValue(i)
                else: 
                    val = 0.0
                b.setValue(i,val)
                if val > 1e-15:
                    nz_val += 1
            b.assemble()
        if monitor:
            print("number of nodes trimmed: ", len(zero_vec))
            print("number of nonzero residuals set: ", nz_val)
        return A,b

    D = A.createVecLeft()
    A.getDiagonal(D)
   
    Istart, Iend = D.getOwnershipRange()
    #if rank == 0:
    #for ind in range(sizes[0][1]):
    for ind in range(Istart,Iend):
        if D.getValue(ind) <= bfr_tol:
            #print(ind)print
            nz_id = nz_id + [ind]
            #nz_id = nz_id + [ind-Istart]
            if b is not None:
                if target is not None:
                    val = target.getValue(ind)
                else: 
                    val = 0.0
                b.setValue(ind,val)
                if val > 1e-15:
                    nz_val += 1

    print("number of nodes trimmed: ", len(nz_id))
    print("number of nonzero residuals set: ", nz_val)

    A.zeroRows(nz_id)

    A.assemble()
    if b is not None:
        b.assemble()
    return A,b


def solveNewtonsLinear(A,L,u_f, M, u_p, 
                    maxIters=20, 
                    relativeTolerance=1e-7,
                    monitorNewtonConvergence=True,
                    moniterLinearConvergence=False,
                    linear_method=None,
                    linear_preconditioner=None,
                    relax_param=1,
                    zero_vec = None):
    '''
    Use Newtons method to solve poorly conditioned linear system
    Used for the 3D biharmonic system, where finite precision issues led to exploding matrix values
    
    '''

    A_b, L_b = assembleLinearSystemBackground(A,L,M)
    u_p = zeroDofBackground(M)
    res_b = zeroDofBackground(M)

    if zero_vec is not None:
        A_b, L_b = trimNodes(A_b,b=L_b,target=u_p,zero_vec=zero_vec,monitor=False)

    #smin,smax = estimateConditionNumber(A_b,L_b,u_p)
    #print('smin: ', smin)
    #print('smax: ', smax)

    L_b.assemble()
    converged = False   
    for i in range(0,maxIters):
        A_b.multAdd(u_p,L_b,res_b)
        currentNormRes = res_b.norm()
        #print("residual norm: ", currentNormRes)
        du_p = zeroDofBackground(M)
        #smin,smax = estimateConditionNumber(A_b,res_b,u_p)
        #print('smin: ', smin)
        #print('smax: ', smax)
        solveKSP(A_b, res_b, du_p, method=linear_method, PC = linear_preconditioner, \
            monitor=moniterLinearConvergence,bfr_tol=None,bfr_b=False)
        currentNorm = du_p.norm()
        #print('du norm: ', currentNorm)
        if(i==0):
            initialNorm = currentNorm
            initialNormRes = currentNormRes
        relativeNorm = currentNorm/initialNorm
        relativeNormRes = currentNormRes/initialNormRes

        if(monitorNewtonConvergence == True):
            if(mpirank == 0):
                log(INFO, "Newton solver iteration: "
                    +str(i)+", Relative norm of du: "+ str(relativeNorm) 
                    +", Relative norm of res: "+ str(relativeNormRes))
                sys.stdout.flush()
        if(relativeNorm < relativeTolerance) or (relativeNormRes < relativeTolerance):
            if(mpirank == 0):
                print('converged')
            converged = True
            #print(u_p.norm())
            return u_p

        u_p += -du_p*relax_param
        #normUP = u_p.norm()
        #print('u norm: ', normUP)

        transferToForeground(u_f,u_p,M)
        
    if(not converged):
        log(INFO, "ERROR: Nonlinear solver failed to converge.")
        exit()

def solveNonlinear(res_f, u_f, M, u_p, 
                    maxIters=20, 
                    relativeTolerance=1e-4,
                    monitorNewtonConvergence=True,
                    moniterLinearConvergence=False,
                    linear_method=None,
                    linear_preconditioner=None,
                    bfr_tol=None,
                    relax_param=1,
                    absoluteTolerance=1e-6,
                    absoluteToleranceRes=1e-9,
                    du_0_mag = None,
                    zero_IDs = None,
                    estimateCondNum = False):
    """
    Solve the nonlinear equation "res_f = 0" by Newton's iteration 
    on the background function space and extract the solution DoFs
    onto the foreground.
    
    Parameters
    ----------
    res_f: UFL form of the PDE residual on the foreground mesh 
    u_f: Dolfin function on the foreground mesh
    M: extraction matrix
    u_p: PETSc Vector representing the dofs on the background mesh
    """
    #A,L = trimNodes(A,L,bfr_tol=bfr_tol)
    converged = False
    for i in range(0,maxIters):

        J_f = derivative(res_f,u_f)
        dR_b, R_b = assembleLinearSystemBackground(J_f,res_f,M)
        if bfr_tol is not None:
            dR_b, R_b = trimNodes(dR_b,R_b,bfr_tol=bfr_tol,target=u_p)
        elif zero_IDs is not None:
            dR_b, R_b =  trimNodes(dR_b,b=R_b,target=u_p,zero_vec=zero_IDs,monitor=True)
         

        du_p = zeroDofBackground(M)
        if estimateCondNum:
            smin,smax = estimateConditionNumber(dR_b,R_b,du_p)
        solveKSP(dR_b, R_b, du_p, method=linear_method, PC = linear_preconditioner, \
            monitor=moniterLinearConvergence,bfr_tol=None)
        currentNorm = du_p.norm()
        currentNormRes = R_b.norm()
        if(i==0):
            initialNorm = currentNorm
            initialNormRes = currentNormRes
        if du_0_mag is not None:
            initialNorm = du_0_mag

        relativeNorm = currentNorm/initialNorm
        relativeNormRes = currentNormRes/initialNormRes

        if(monitorNewtonConvergence == True):
            #print("currentNorm: ", currentNorm)
            #print("currentNormRes: ", currentNormRes)
            if(mpirank == 0):
                log(INFO, "Newton solver iteration: "
                    +str(i)+", Relative norm of du: "+ str(relativeNorm) 
                    +", Relative norm of res: "+ str(relativeNormRes))
                sys.stdout.flush()
        if(relativeNorm < relativeTolerance and relativeNormRes<relativeTolerance):
            converged = True
            break
        if i > 1: 
            if currentNorm < absoluteTolerance or currentNormRes <absoluteToleranceRes:
            #if currentNormRes <absoluteToleranceRes:
                converged = True
                break 
        u_p += -du_p*relax_param
        transferToForeground(u_f,u_p,M)
    
    if(not converged):
        log(INFO, "ERROR: Nonlinear solver failed to converge.")
        exit()
    return 


def estimateConditionNumber(A,b,u, bfr_tol=None,rtol=1E-8, atol=1E-9, max_it=100000, PC=None):
    if bfr_tol is not None:
        A,b= trimNodes(A, b=b, bfr_tol=bfr_tol)

    ksp = PETSc.KSP().create() 
    ksp.setComputeSingularValues(True)
    ksp.setTolerances(rtol=rtol, atol = atol, max_it=max_it)
    ksp.setType(PETSc.KSP.Type.GMRES)

    A.assemble()
    ksp.setOperators(A)
    PETSc.Options().setValue("ksp_view_singularvalues","")
    ksp.setFromOptions()  
    pc = ksp.getPC()
    if PC is not None:
        pc.setType(PC)
    else: 
        pc.setType("none")
    ksp.setUp()
    ksp.setGMRESRestart(1000)
    ksp.solve(arg2v(b), arg2v(u))

    smax,smin = ksp.computeExtremeSingularValues()

    return smax,smin

def solveKSP(A,b,u,method='gmres', PC='jacobi',
            remove_zero_diagonal=False,rtol=1E-8, 
            atol=1E-9, max_it=1000000, bfr_tol=1E-9,
            monitor=True,gmr_res=3000,bfr_b=True):
    """
    solve linear system A*u=b
    A: PETSC Matrix
    b: PETSC Vector
    u: PETSC Vector
    """

    if method == None:
        method='gmres'
    if PC == None:
        PC='jacobi'

    if method == 'mumps':
        ksp = PETSc.KSP().create() 
        ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)

        if remove_zero_diagonal and bfr_tol is not None:
            if bfr_b:
                A,b = trimNodes(A,b=b,bfr_tol=bfr_tol)
            else:
                A,_ = trimNodes(A,bfr_tol=bfr_tol)

        opts = PETSc.Options("mat_mumps_")
        # icntl_24: controls the detection of “null pivot rows", 1 for on, 0 for off
        opts["icntl_24"] = 1
        # cntl_3: tolerance used to determine null pivot rows
        opts["cntl_3"] = 1e-12           

        A.assemble()
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc=ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setUp()

        ksp_d = PETScKrylovSolver(ksp)
        ksp_d.solve(PETScVector(u),PETScVector(b))
        return 


    ksp = PETSc.KSP().create() 
    ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)

    if method == 'gmres': 
        ksp.setType(PETSc.KSP.Type.FGMRES)
    elif method == 'gcr':
        ksp.setType(PETSc.KSP.Type.GCR)
    elif method == 'cg':
        ksp.setType(PETSc.KSP.Type.CG)


    if remove_zero_diagonal and bfr_tol is not None:
        A,b = trimNodes(A,b=b, bfr_tol=bfr_tol)

    if PC == 'jacobi':
        A.assemble()
        ksp.setOperators(A)
        pc = ksp.getPC()
        pc.setType("jacobi")
        ksp.setUp()
        ksp.setGMRESRestart(300)

    elif PC == 'ASM':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("asm")
        pc.setASMOverlap(1)
        ksp.setUp()
        localKSP = pc.getASMSubKSP()[0]
        localKSP.setType(PETSc.KSP.Type.FGMRES)
        localKSP.getPC().setType("lu")
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ICC':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("icc")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ILU':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("euclid")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC == 'ILUT':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("pilut")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

        '''
        setHYPREType(self, hypretype):
        setHYPREDiscreteCurl(self, Mat mat):
        setHYPREDiscreteGradient(self, Mat mat):
        setHYPRESetAlphaPoissonMatrix(self, Mat mat):
        setHYPRESetBetaPoissonMatrix(self, Mat mat=None):
        setHYPRESetEdgeConstantVectors(self, Vec ozz, Vec zoz, Vec zzo=None):                
        '''


    ksp_d = PETScKrylovSolver(ksp)
    if monitor:
        ksp_d.parameters['monitor_convergence'] = True
    ksp_d.parameters['absolute_tolerance'] = atol
    ksp_d.parameters['relative_tolerance'] = rtol
    ksp_d.parameters['maximum_iterations'] = max_it
    ksp_d.parameters['nonzero_initial_guess'] = True
    ksp_d.parameters['error_on_nonconvergence'] = False
    ksp_d.solve(PETScVector(u),PETScVector(b))

    history = ksp.getConvergenceHistory()
    if monitor:
        print('Converged in', ksp.getIterationNumber(), 'iterations.')
        print('Convergence history:', history)



def readExOp(fileNames,V,mesh,l_size,nodeFileNames=None,k=1,NFields=1):

    # extract matrix data from files
    i = 0
    for fileName in fileNames:
        if i == 0:
            matData = np.genfromtxt(fileName, dtype=[np.int, np.int, np.float])
        else:
            blockMatData = np.genfromtxt(fileName, dtype=[np.int, np.int, np.float])
            matData = np.concatenate((matData, blockMatData))
        i += 1 
    s = matData.size
    node_IDs = np.empty((s,2), dtype=np.int32)
    weights = np.empty((s,1), dtype=np.float)
    i = 0

    for extract in matData:
        node_IDs[i,0] = extract[0]
        node_IDs[i,1] = extract[1]
        weights[i,0] = extract[2]
        i = i +1

    n_net = V.dim()             # total number of basis functions in fg mesh
    m = np.max(node_IDs[:,1])   # number of basis functions per field in bg mesh
    m_net = m*NFields           # total number of basis functions in bg mesh
    # create M matrix 
    A = PETSc.Mat().create(comm=worldcomm) 

    if l_size< n_net: 
        A.setSizes(((l_size,n_net),m_net))
    else:
        A.setSizes((l_size,m_net))
    A.setUp()

    for field in range(NFields):
        #split function spaces into sub spaces for each field 
        if NFields == 1:
            V_field = V
        else:
            V_field = V.sub(field)

        if k==1:
            exoToFenicsDofMap = convertDOFsk1(V_field,mesh)
        elif k ==2:
            if mesh.geometry().dim() == 2:
                exoToFenicsDofMap = convertDOFs2Dk2(mesh,V_field,nodeFileNames)
            elif mesh.geometry().dim() == 3:
                exoToFenicsDofMap= convertDOFs3Dk2(mesh,V_field,nodeFileNames)
        else:
            print('only polynomial orders 1 and 2 are currently supported')

        #populate the extraction operator
        i = 0 
        for row in node_IDs: # for each row in the T matrix given,
            #note: exo starts indexing at 1, so we shift the input to start at zero
            exoID = int(node_IDs[i,0] -1)
            fenics_fg_ID = exoToFenicsDofMap[exoID]
            #bg_ID= node_IDs[i,1]*NFields + field - 1
            bg_ID = node_IDs[i,1] +field*m -1 
            #if i < 20:
            #    print(exoID, fenics_fg_ID,bg_ID)
            if fenics_fg_ID >= 0:
                A.setValue(fenics_fg_ID, bg_ID, weights[i])
            i += 1

    A.assemble()
    
    return A

def convertDOFsk1(V,mesh):
    '''
    Function to convert from global vertex ids to global DOF ids
    '''

    #local to local, vertex to dof
    dofMap = V.dofmap()
    n_loc_verts = mesh.num_entities(0)
    n_glo_verts = mesh.num_entities_global(0)

    loc_to_glob_vertex = mesh.topology().global_indices(0)

    vertexToDOFGlobal = (np.zeros(int(n_glo_verts),dtype=np.int32) - 1)
    for loc_vert in range(n_loc_verts):
        loc_dof = dofMap.entity_dofs(mesh, 0, [loc_vert])
        glo_vert = loc_to_glob_vertex[loc_vert]
        glo_dof = dofMap.local_to_global_index(loc_dof[0])
        vertexToDOFGlobal[int(glo_vert)] = int(glo_dof)

    return vertexToDOFGlobal
    


def convertDOFs2Dk2(mesh,V,fileName):
    #global number of dofs
    exo_dofs = np.genfromtxt(fileName,delimiter=",",dtype='int_') #each row (1,2,3,4,5,6)
    num_dofs = np.amax(exo_dofs) + 1

    exoToFenicsDofMap = (np.zeros(int(num_dofs)) - 1)
    dofmap = V.dofmap()

    # i gives global cell ID
    # i_loc = local cell ID
    i_loc = 0
    for cell in cells(mesh):
        i = cell.global_index()
        cell_dofs = dofmap.entity_closure_dofs(mesh,2,[i_loc])
        global_cell_dofs = cell_dofs #temporary assignment
        counter = 0
        for dof in cell_dofs:
            global_cell_dofs[counter] = dofmap.local_to_global_index(dof)
            counter= counter + 1

        exo_facet_dofs = [exo_dofs[i,3], exo_dofs[i,4], exo_dofs[i,5]]
        verts = cell.entities(0) # local IDS
        facets = cell.entities(1) # local IDS
        ii = 0 #vertex counter
        for v in verts:
            glo_vert = MeshEntity(mesh, 0, v).global_index()
            exoToFenicsDofMap[glo_vert] = global_cell_dofs[ii]
            ii = ii + 1

        for f in facets:
            facet_dofs = dofmap.entity_closure_dofs(mesh,1,[f])  #(a,b,c)

            global_facet_dofs = facet_dofs #temporary assignment
            counter = 0
            for dof in facet_dofs:
                global_facet_dofs[counter] = dofmap.local_to_global_index(dof)
                counter= counter + 1

            # get vertice indices 
            f_v_ID = MeshEntity(mesh, 1, f).entities(0)
            f_v_global_a = MeshEntity(mesh, 0, f_v_ID[0]).global_index()
            f_v_global_b = MeshEntity(mesh, 0, f_v_ID[1]).global_index()
            a = f_v_global_a
            b = f_v_global_b

            #if fenicsToExoDofMap[facet_dofs[0]] == exo_dofs[[i,0]] or fenicsToExoDofMap[facet_dofs[1]]== exo_dofs[[i,0]]:
            if exo_dofs[i,0] == a or exo_dofs[i,0] ==b:
                if exo_dofs[i,1] == a or exo_dofs[i,1] ==b:
                    #(a,b) = (1,2) => c=4
                    exoToFenicsDofMap[exo_dofs[i,3]] = global_facet_dofs[2]
                elif exo_dofs[i,2] == a or exo_dofs[i,2] ==b:
                    #(a,b) = (1,3) => c=6
                    exoToFenicsDofMap[exo_dofs[i,5]] = global_facet_dofs[2]
            elif exo_dofs[i,1] == a or exo_dofs[i,1] ==b:
                if exo_dofs[i,2] == a or exo_dofs[i,2] ==b:
                    #(a,b) = (2,3) => c=5
                    exoToFenicsDofMap[exo_dofs[i,4]] = global_facet_dofs[2]
        i_loc = i_loc + 1

    return exoToFenicsDofMap


def convertDOFs3Dk2(mesh,V,fileName):
    #global number of dofs
    exo_dofs = np.genfromtxt(fileName,delimiter=",",dtype='int_') #each row (1,2,3,4,5,6)
    num_dofs = np.amax(exo_dofs) + 1

    exoToFenicsDofMap = (np.zeros(int(num_dofs)) - 1)
    fenicsToExoDofMap = (np.zeros(int(num_dofs)) - 1)
    dofmap = V.dofmap()

    # i gives global cell ID
    # i_loc = local cell ID
    i_loc = 0
    for cell in cells(mesh):
        i = cell.global_index()
        cell_dofs = dofmap.entity_closure_dofs(mesh,3,[i_loc])
        global_cell_dofs = cell_dofs #temporary assignment
        counter = 0
        for dof in cell_dofs:
            global_cell_dofs[counter] = dofmap.local_to_global_index(dof)
            counter= counter + 1

        edge0 = {exo_dofs[i,0], exo_dofs[i,1]}
        edge1 = {exo_dofs[i,1], exo_dofs[i,2]}
        edge2 = {exo_dofs[i,2], exo_dofs[i,0]}
        edge3 = {exo_dofs[i,0], exo_dofs[i,3]}
        edge4 = {exo_dofs[i,1], exo_dofs[i,3]}
        edge5 = {exo_dofs[i,2], exo_dofs[i,3]}

        verts = cell.entities(0) # local IDS
        facets = cell.entities(1) # local IDS
        ii = 0 #vertex counter
        for v in verts:
            glo_vert = MeshEntity(mesh, 0, v).global_index()
            exoToFenicsDofMap[glo_vert] = global_cell_dofs[ii]

            ii = ii + 1
        for f in facets:
            facet_dofs = dofmap.entity_closure_dofs(mesh,1,[f])  #(a,b,c)uhj
            global_facet_dofs = facet_dofs #temporary assignment
            counter = 0
            for dof in facet_dofs:
                global_facet_dofs[counter] = dofmap.local_to_global_index(dof)
                counter= counter + 1

            #a = fenicsToExoDofMap[global_facet_dofs[0]]
            #b = fenicsToExoDofMap[global_facet_dofs[1]]

            # get vertice indices 
            f_v_ID = MeshEntity(mesh, 1, f).entities(0)
            f_v_global_a = MeshEntity(mesh, 0, f_v_ID[0]).global_index()
            f_v_global_b = MeshEntity(mesh, 0, f_v_ID[1]).global_index()
            end_points = {f_v_global_a, f_v_global_b}
            #local_endpoint_dofs = dofmap.entity_dofs(mesh,0,f_v_ID)
            #global_endpoint_dof_1 = dofmap.local_to_global_index(local_endpoint_dofs[0])
            #global_endpoint_dof_2 = dofmap.local_to_global_index(local_endpoint_dofs[1])


            #if fenicsToExoDofMap[facet_dofs[0]] == exo_dofs[[i,0]] or fenicsToExoDofMap[facet_dofs[1]]== exo_dofs[[i,0]]:
            if end_points == edge0:
                exoToFenicsDofMap[exo_dofs[i,4]] = global_facet_dofs[2]
                fenicsToExoDofMap[global_facet_dofs[2]] = exo_dofs[i,4]
            elif end_points == edge1:
                exoToFenicsDofMap[exo_dofs[i,5]] = global_facet_dofs[2]
                fenicsToExoDofMap[global_facet_dofs[2]] = exo_dofs[i,5]
            elif end_points == edge2:
                exoToFenicsDofMap[exo_dofs[i,6]] = global_facet_dofs[2]
                fenicsToExoDofMap[global_facet_dofs[2]] = exo_dofs[i,6]
            elif end_points == edge3:
                exoToFenicsDofMap[exo_dofs[i,7]] = global_facet_dofs[2]
                fenicsToExoDofMap[global_facet_dofs[2]] = exo_dofs[i,7]
            elif end_points == edge4:
                exoToFenicsDofMap[exo_dofs[i,8]] = global_facet_dofs[2]
                fenicsToExoDofMap[global_facet_dofs[2]] = exo_dofs[i,8]
            elif end_points == edge5:
                exoToFenicsDofMap[exo_dofs[i,9]] = global_facet_dofs[2]
                fenicsToExoDofMap[global_facet_dofs[2]] = exo_dofs[i,9]
        i_loc = i_loc + 1

    return exoToFenicsDofMap
