"""
The "la_utils" module
------------------------------
contains linear algebra operations based on PETSc4PY
"""
from dolfin import *
from petsc4py import PETSc
from dolfin.cpp.log import *

import numpy as np

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

def v2p(v):
    """
    Convert "dolfin.cpp.la.PETScVector" to 
    "petsc4py.PETSc.Vec".
    """
    return as_backend_type(v).vec()

def m2p(A):
    """
    Convert "dolfin.cpp.la.PETScMatrix" to 
    "petsc4py.PETSc.Mat".
    """
    return as_backend_type(A).mat()

def arg2v(x):
    """
    Convert dolfin Function or dolfin Vector to PETSc.Vec.
    """
    if isinstance(x, DOLFIN_FUNCTION):
        x_PETSc = as_backend_type(x.vector()).vec()
    elif isinstance(x, DOLFIN_PETSCVECTOR):
        x_PETSc = v2p(x)
    elif isinstance(x, DOLFIN_VECTOR):
        x_PETSc = v2p(x)
    elif isinstance(x, PETSC4PY_VECTOR):
        x_PETSc = x
    else:
        raise TypeError("Type " + str(type(x)) + " is not supported yet.")
    return x_PETSc

def arg2m(A):
    """
    Convert dolfin Matrix to PETSc.Mat.
    """
    if isinstance(A, DOLFIN_PETSCMATRIX):
        A_PETSc = m2p(A)
    elif isinstance(A, DOLFIN_MATRIX):
        A_PETSc = m2p(A)
    elif isinstance(A, PETSC4PY_MATRIX):
        A_PETSc = A
    else:
        raise TypeError("Type " + str(type(A)) + " is not supported yet.")
    return A_PETSc

def zero_petsc_vec(num_el, comm=MPI.comm_world):
    """
    Create zero PETSc vector of size ``num_el``.

    Parameters
    ----------
    num_el : int
    vec_type : str, optional
        For petsc4py.PETSc.Vec types, see petsc4py.PETSc.Vec.Type.
    comm : MPI communicator

    Returns
    -------
    v : petsc4py.PETSc.Vec
    """
    v = PETSc.Vec().create(comm)
    v.setSizes(num_el)
    v.setUp()
    v.assemble()
    return v

def zero_petsc_mat(row, col, comm=MPI.comm_world, row_loc=None, col_loc=None):
    """
    Create zeros PETSc matrix with shape (``row``, ``col``).

    Parameters
    ----------
    row : int
    col : int
    mat_type : str, optional
        For petsc4py.PETSc.Mat types, see petsc4py.PETSc.Mat.Type
    comm : MPI communicator

    Returns
    -------
    A : petsc4py.PETSc.Mat
    """
    A = PETSc.Mat(comm)
    A.createAIJ([(row_loc, row),(col_loc, col)],comm=comm)
    A.setUp()
    A.assemble()
    return A


def updateU(u):
    """
    Update the ghost elements in the function `u`
    
    Parameters
    ----------
    u: Dolfin Function
    """
    arg2v(u.vector()).assemble()
    arg2v(u.vector()).ghostUpdate()



def A_x_b(A, x, b):
    """
    Compute "Ax = b".

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    b : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec
    """
    return m2p(A).mult(v2p(x), v2p(b))

def AT_x(A, x):
    """
    Compute b = A^T*x.

    Parameters
    ----------
    A : dolfin Matrix, dolfin PETScMatrix or petsc4py.PETSc.Mat
    x : dolfin Function, dolfin Vector, dolfin PETScVector
        or petsc4py.PETSc.Vec

    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """

    A_PETSc = arg2m(A)
    x_PETSc = arg2v(x)
    row, col = A_PETSc.getSizes()
    b_PETSc = zero_petsc_vec(col, comm=A_PETSc.getComm())
    A_PETSc.multTranspose(x_PETSc, b_PETSc)
    return b_PETSc
 
def AT_R_A(A, R):
    """
    Compute "A^T*R*A". A,R are "petsc4py.PETSc.Mat".

    Parameters
    -----------
    A : petsc4py.PETSc.Mat
    R : petsc4py.PETSc.Mat

    Returns
    ------ 
    ATRA : petsc4py.PETSc.Mat
    """
    AT = A.transpose()
    ATR = AT.matMult(R)
    ATT = A.transpose()
    ATRA = ATR.matMult(ATT)
    return ATRA
