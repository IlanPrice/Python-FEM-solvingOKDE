import random
import sympy as sym
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from sympy.utilities.codegen import ccode
from sympy import symbols
import sympy as sp
import mpi4py.MPI


class OKsystem(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

# Model parameters

epssi    = 0.08
Tfinal = 1.0
m       = 0 # mass
sigma   =  20.0


# Create mesh
leftbound = 0.0; rightbound = 1.0;
lim = 5
Error = np.zeros([1,lim])
k = 0
n = 10
C = 0.5
while (k < lim):

    h =  (rightbound-leftbound)/n
    dt = C*(h**2)

    mesh = IntervalMesh(n, leftbound, rightbound)

    # Build mixed function space for solution of O-K system
    V = FiniteElement("Lagrange", mesh.ufl_cell(), 4)
    ME = FunctionSpace(mesh, V*V)

    # Define test functions
    q, v  = TestFunctions(ME)

    # Define functions
    y   = Function(ME)  # current solution
    y0  = Function(ME)  # solution from previous converged step


    # Split mixed functions
    u,  w  = split(y)
    u0, w0 = split(y0)

    # Method of Manufactured Solutions

    u_expr = "sin(t)*cos(2*pi*x[0])"
    u_e = Expression(u_expr, t = 0.0, degree = 4) # Same rationale here for degree as in implicit scheme 1D MMS script
    w_expr = "(4*ep*ep*pi*pi -1)*sin(t)*cos(2*pi*x[0]) + sin(t)*sin(t)*sin(t)*cos(2*pi*x[0])*cos(2*pi*x[0])*cos(2*pi*x[0])"
    y_ex = Expression((u_expr, w_expr), t = 0.0, ep = epssi, degree = 4)
    f_expr = "cos(2*pi*x[0])*(cos(t) + 4*pi*pi*(4*ep*ep*pi*pi-1)*sin(t) + 12*pi*pi*sin(t)*sin(t)*sin(t)*cos(2*pi*x[0])*cos(2*pi*x[0]) - 24*pi*pi*sin(t)*sin(t)*sin(t)*sin(2*pi*x[0])*sin(2*pi*x[0]) + sig*sin(t))"
    f = Expression(f_expr, t = 0.0, sig = sigma, ep = epssi, degree = 4)
    ICs = interpolate(y_ex, ME)
    y.assign(ICs)
    y0.assign(ICs)

    #Optimise
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Weak form of the equations
    L1 = inner(u,q)*dx - inner(u0,q)*dx + dt*dot(grad(w), grad(q))*dx + dt*sigma*inner(u-m,q)*dx - dt*inner(f,q)*dx
    L2 = inner(w,v)*dx - (epssi**2)*dot(grad(u), grad(v))*dx - inner(u0**3,v)*dx + inner(u,v)*dx
    L = L1 + L2

    E = 0

    t = 0.0
    count = 0
    while (t < Tfinal):
        t += float(dt)
        ## MMS
        f.t = t; y_ex.t = t; u_e.t = t #Update t in expressions

        ## Solve O-K system
        y0.assign(y)
        solve(L == 0, y)

        Err = errornorm(u_e, y.split()[0]) #L2 Error norm at this time step

        if (Err > E):
            E = Err #Keeps max over all time steps

        count+=1

    Error[0,k] = E
    n = n * 2
    k += 1

    io.savemat("OK_1D_MMS_IMX", mdict={"Error": Error})
