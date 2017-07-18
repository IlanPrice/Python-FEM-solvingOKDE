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


#####################################################


        # Method of Manufactured Solutions

# This code can implement the MMS described in the special topic report,
# and hard-codes the manufactured solution


#####################################################



#set_log_active(False)  #Uncomment when code working well and want to stop printing to screen

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
Tfinal = 3.0
m       = 0 # mass
sigma   =  20.0


# Create mesh
leftbound = 0.0; rightbound = 1.0;
lim = 5
Error = np.zeros([1,lim]) #Initial MMS looped to 10*2^5, IMX MMS comparison looped to 10*2^4
k = 0
n = 10
C = 0.5 # C=0.5 was used for comparison with IMX, C=1 was used for initial MMS
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

    # Note, alternativelt can opt for implementation using SymPy to do symbolic differentiation, and only hard code the manufactured u
    u_expr = "sin(t)*cos(2*pi*x[0])"
    u_e = Expression(u_expr, t = 0.0, degree = 4) #The high degree of the expressions (which can be made higher still), is to ensure an accurate 'errornorm' calculation. Fenics complains when the degree is too low and errornorm is used.
    w_expr = "(4*ep*ep*pi*pi -1)*sin(t)*cos(2*pi*x[0]) + sin(t)*sin(t)*sin(t)*cos(2*pi*x[0])*cos(2*pi*x[0])*cos(2*pi*x[0])"
    y_ex = Expression((u_expr, w_expr), t = 0.0, ep = epssi, degree = 4) #Note that to interpolate in the mixed function space, we have to create the expression like this, and not create an expression u_ex and w_ex individually
    f_expr = "cos(2*pi*x[0])*(cos(t) + 4*pi*pi*(4*ep*ep*pi*pi-1)*sin(t) + 12*pi*pi*sin(t)*sin(t)*sin(t)*cos(2*pi*x[0])*cos(2*pi*x[0]) - 24*pi*pi*sin(t)*sin(t)*sin(t)*sin(2*pi*x[0])*sin(2*pi*x[0]) + sig*sin(t))"
    f = Expression(f_expr, t = 0.0, sig = sigma, ep = epssi, degree = 4)

    #Start with exaact solution at t=0 as initial condition
    ICs = interpolate(y_ex, ME)
    y.assign(ICs)
    y0.assign(ICs)


    # Calculate dphi/du
    u = variable(u)
    phi    = 0.25*(1-(u**2))**2
    dphidu = diff(phi, u)

    # Optimise
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Weak form of the equations with f term added

    L1 = inner(u,q)*dx - inner(u0,q)*dx + dt*dot(grad(w), grad(q))*dx + dt*sigma*inner(u-m,q)*dx - dt*inner(f,q)*dx
    L2 = inner(w,v)*dx - (epssi**2)*dot(grad(u), grad(v))*dx - inner(dphidu,v)*dx
    L = L1 + L2

    a = derivative(L, y)

    # Create nonlinear problem and Newton solver
    problem = OKsystem(a, L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "gmres" #Can be swapped back to LU, but this proved faster
    solver.parameters["convergence_criterion"] = "incremental"
    #solver.parameters["convergence_criterion"] = "residual"
    solver.parameters["relative_tolerance"] = 1e-6

    E = 0 #This variable will store the infinity norm of the L2 norms of the errors, over time.

    t = 0.0
    count = 0
    while (t < Tfinal):
        t += float(dt)
        ## MMS

        f.t = t; y_ex.t = t; u_e.t = t #Update t in expressions

        y0.assign(y) # update stored previous timestep
        solver.solve(problem, y.vector()) #Solve O-K system

        Err = errornorm(u_e, y.split()[0]) # L2 Error norm at this time step

        if (Err > E):
            E = Err # Keeps max over all time steps

        count+=1

    Error[0,k] = E
    n = n * 2
    k += 1


    io.savemat("OK_1D_Fully_Implicit_MMS", mdict={"Error_implicit": Error}) #Save to a matlab workspace file (inside the loop in case simulation needs to quite early, then still have results so far)
