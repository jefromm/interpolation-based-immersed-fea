'''
Kirchoff-Love shell formulation, for a square shell pinned at the edges 
subjected to a uniform vertical force, using background-unfitted meshes

To run, specify the mesh refinement level (--ref )/number of elements per edge (--n ) :
python3 pinned_shell_unfitted.py --ref 5
or 
python3 pinned_shell_unfitted.py --n 128

'''



from tIGAr import *
from tIGAr.BSplines import *
from tIGAr.timeIntegration import *
import numpy as np
import pandas as pd

# For basic unstructured mesh generation:
from mshr import *

# Subclass of EqualOrderSpline which overrides the generateMesh()
# method to create an unstructured triangle mesh, with a resolution
# given by an additional argument to customSetup():
class CustomMeshSpline(EqualOrderSpline):
    def customSetup(self,args):
        self.numFields = args[0]
        self.controlMesh = args[1]
        self.resolution = args[2]
        self.degree = args[3]
    # Inherited through AbstractExtractionGenerator:
    def generateMesh(self):
        square = Rectangle(Point(-0.5,-0.5),Point(0.5,0.5))
        unit_square_mesh = generate_mesh(square,self.resolution)
        angle = 0.25*pi 
        # rotate 45 degrees
        c = unit_square_mesh.coordinates()
        for i, (x, y) in enumerate(c):
            xx = x*cos(angle) + y*sin(angle) 
            yy = -x*sin(angle) + y*cos(angle)

            c[i][0] = xx
            c[i][1] = yy


        return unit_square_mesh

    # Inherited through AbstractMultiFieldSpline:
    def getDegree(self,field):
        return self.degree

# define tracker points 

####### Preprocessing #######

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n',dest='n',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--ref',dest='ref',default=-1,
                    help='Refinement level.')
args = parser.parse_args()

ref = float(args.ref)
if ref > -1 :
    Nel = int(4*2**ref)
else:
    Nel = int(args.n)


if(mpirank==0):
    print("Generating extraction data...")

# Specify the number of elements in each direction.
NELu = Nel
NELv = Nel

# Specify degree in each direction.
degs = [2,2]

# Generate open knot vectors for each direction.
kvecs = [uniformKnots(degs[0],-1.0,1.0,NELu),
         uniformKnots(degs[1],-1.0,1.0,NELv)]

# Generate an explicit B-spline control mesh.  The argument extraDim allows
# for increasing the dimension of physical space beyond that of parametric
# space.  We want to model deformations in 3D, so one extra dimension is
# required of physical space.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs,extraDim=1)

# Create a spline generator with three unknown fields for the displacement
# components.  
#splineGenerator = EqualOrderSpline(3,controlMesh)
splineGenerator = CustomMeshSpline(3,controlMesh,NELu,degs[0])

# Apply clamped boundary conditions to the displacement.  (Pinned BCs are in
# comments, but need more load steps and/or a smaller load to converge.)
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
'''
for side in range(0,2):
    for direction in range(0,1):
        sideDofs = scalarSpline.getSideDofs(direction,side,
                                            ########################
                                            #nLayers=2) # clamped BC
                                            nLayers=1) # pinned BC
                                            ########################
        for i in range(0,3):
            splineGenerator.addZeroDofs(i,sideDofs)
'''
# Write extraction data to the filesystem.
DIR = "./extraction"
splineGenerator.writeExtraction(DIR)


####### Analysis #######

if(mpirank==0):
    print("Creating extracted spline...")

# Quadrature degree for the analysis:
QUAD_DEG = 6

# Generate the extracted representation of the spline.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)
#spline = ExtractedSpline(DIR,QUAD_DEG)

# Override geometric mapping from control mesh #############################
xi = spline.parametricCoordinates()
#spline.F = as_vector([spline.F[0],
#                      spline.F[1],
#                      Constant(1.0)*(1.0-xi[0]**2)])
#spline.F = as_vector([spline.F[0],
#                      spline.F[1],
#                      Constant(0.0)])
# Need to copy/paste these definitions from the constructor for
# ExtractedSpline after replacing spline.F:
spline.DF = grad(spline.F)
spline.g = getMetric(spline.F)
spline.N = FacetNormal(spline.mesh)
spline.n = mappedNormal(spline.N,spline.F)
spline.dx = tIGArMeasure(volumeJacobian(spline.g),dx,spline.quadDeg)
spline.ds = tIGArMeasure(surfaceJacobian(spline.g,spline.N),\
                         ds,spline.quadDeg,spline.boundaryMarkers)
spline.pinvDF = pinvD(spline.F)
spline.gamma = getChristoffel(spline.g)
############################################################################

print(spline.M.mat().getSize())


if(mpirank==0):
    print("Starting analysis...")
    
# Unknown midsurface displacement
y_hom = Function(spline.V) # in homogeneous representation
y = spline.rationalize(y_hom) # in physical coordinates

# Reference configuration:
X = spline.F

# Current configuration:
x = X + y

# Normalize a vector v.
def unit(v):
    return v/sqrt(inner(v,v))

# Helper function to compute geometric quantities for a midsurface
# configuration x.
def shellGeometry(x):

    # Covariant basis vectors:
    dxdxi = spline.parametricGrad(x)
    a0 = as_vector([dxdxi[0,0],dxdxi[1,0],dxdxi[2,0]])
    a1 = as_vector([dxdxi[0,1],dxdxi[1,1],dxdxi[2,1]])
    a2 = unit(cross(a0,a1))

    # Metric tensor:
    a = as_matrix(((inner(a0,a0),inner(a0,a1)),
                   (inner(a1,a0),inner(a1,a1))))
    # Curvature:
    deriva2 = spline.parametricGrad(a2)
    b = -as_matrix(((inner(a0,deriva2[:,0]),inner(a0,deriva2[:,1])),
                    (inner(a1,deriva2[:,0]),inner(a1,deriva2[:,1]))))
    
    return (a0,a1,a2,a,b)



# Use the helper function to obtain shell geometry for the reference
# and current configurations defined earlier.
A0,A1,A2,A,B = shellGeometry(X)
a0,a1,a2,a,b = shellGeometry(x)

# Strain quantities.
epsilon = 0.5*(a - A)
kappa = B - b

# Helper function to convert a 2x2 tensor T to its local Cartesian
# representation, in a shell configuration with metric a, and covariant
# basis vectors a0 and a1.
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

h_th = Constant(0.1)  # shell thickness
E = Constant(4.8e5)  # Young's modulus
nu = Constant(0.38)  # Poisson's ratio
arealForceDensity = 90


# The material matrix:
D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                 [nu,   1.0,  0.0         ],
                                 [0.0,  0.0,  0.5*(1.0-nu)]])
# The shell thickness:


# Extension and bending resultants:
nBar = h_th*D*voigt(epsilonBar)
mBar = (h_th**3)*D*voigt(kappaBar)/12.0

# Compute the elastic potential energy density
Wint = 0.5*(inner(voigt(epsilonBar),nBar)
            + inner(voigt(kappaBar),mBar))*spline.dx

# Take the Gateaux derivative of Wint in test function direction z_hom.
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)
dWint = derivative(Wint,y_hom,z_hom)

# External follower load magnitude:
f = as_vector([Constant(0.0), Constant(0.0), arealForceDensity])
u_pre = Constant([0, 0, 0])
alpha_d = Constant(1e5)
h_mesh = 1/NELu


# Full nonlinear residual:
res = dWint + - inner(f,z_hom)*spline.dx \
    + (alpha_d*E*h_th/h_mesh)*inner(y_hom-u_pre, z_hom)*spline.ds

# Consistent tangent:
dRes = derivative(res,y_hom)


# Files for output:  Because an explicit B-spline is used, we can simply use
# the homogeneous (= physical) representation of the displacement in a
# ParaView Warp by Vector filter.

d0File = File("pinned_plate_results/disp-x.pvd")
d1File = File("pinned_plate_results/disp-y.pvd")
d2File = File("pinned_plate_results/disp-z.pvd")
True
F0File = File("pinned_plate_results/F-x.pvd")
F1File = File("pinned_plate_results/F-y.pvd")
F2File = File("pinned_plate_results/F-z.pvd")


Ffunc = spline.project(spline.F,rationalize=False) # (Exact for quadratic F)

(F0,F1,F2) = Ffunc.split()
F0.rename("F0","F0")
F1.rename("F1","F1")
F2.rename("F2","F2")

print("before nonlinear solve")
spline.solveNonlinearVariationalProblem(res,dRes,y_hom)


# Output solution.
(d0,d1,d2) = y_hom.split()
d0.rename("d0","d0")
d1.rename("d1","d1")
d2.rename("d2","d2")
d0File << d0
d1File << d1
d2File << d2
F0File << F0
F1File << F1
F2File << F2


middle = [0.0,0.0]

u_x = y_hom(middle[0],middle[1])[0]
u_y = y_hom(middle[0],middle[1])[1]
u_z = y_hom(middle[0],middle[1])[2]

if mpirank ==0:
    print("Center displacement: (", u_x, ",", u_y, ",",u_z,")")


write_file = False
output_filename = 'pinned_plate_results/disp.csv'
if write_file: 
    f = open(output_filename, 'a')
    print('writing to file')
    f.write("\n")
    #size,time,L2,H1,N
    fs = str(NELu) + ","+ str(u_x)+","+str(u_y) + ","+ str(u_z)
    f.write(fs)
    f.close()

####### Postprocessing #######

# Paraview Calculator filter for displacement from parametric to
# physical space (after combining all 6 output .pvd files with an Append
# Attributes filter):
#
#  (d0+F0-coordsX)*iHat + (d1+F1-coordsY)*jHat + (d2+F2-coordsZ)*kHat

