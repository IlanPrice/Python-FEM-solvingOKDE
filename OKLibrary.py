import random
import sympy as sym
from dolfin import *
# import matplotlib
# matplotlib.use('TkAgg')
# from Tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io


# Class for interfacing with the Newton solver
class OKsystem(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)


def OKuniform(epssi, m, sigma, mesh, dt, Tfinal, ICs, outputfile=""):
# Function to simulation the OKDE with uniform time step as specified.
# Returns the relavent data to a number of results presented in the special topic report, and the solution at Tfinal

    # Build function spaxe and mixed function space for solution of O-K system
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = FunctionSpace(mesh, P1)
    ME = FunctionSpace(mesh, P1*P1)

    # Define test functions
    q, v  = TestFunctions(ME)

    # Define functions
    y   = Function(ME)  # current solution
    y0  = Function(ME)  # solution from previous converged step

    # Split mixed functions
    u,  w  = split(y)
    u0, w0 = split(y0)

    # Define Function space, function and trial and test functions needed to solve the Energy Subproblem
    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # Linear Lagrange Element
    R = FiniteElement("Real",mesh.ufl_cell(),0) # Real Element
    W = FunctionSpace(mesh, V*R)
    r, c = TrialFunction(W)
    p1, p2 = TestFunctions(W)
    ww = Function(W)

    # IC's
    ICs = interpolate(ICs,ME)
    y.assign(ICs)
    y0.assign(ICs)

    # Compute dphi/du
    u = variable(u)
    phi    = 0.25*(1-(u**2))**2
    dphidu = diff(phi, u)

    # Optimise
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Weak form of the equations
    L1 = inner(u,q)*dx - inner(u0,q)*dx + dt*dot(grad(w), grad(q))*dx + dt*sigma*inner(u-m,q)*dx
    L2 = inner(w,v)*dx - (epssi**2)*dot(grad(u), grad(v))*dx - inner(dphidu,v)*dx
    L = L1 + L2

    # Compute derivstive for nonlinear solver
    a = derivative(L, y)

    # Create nonlinear problem and Newton solver
    problem = OKsystem(a, L)
    solver = NewtonSolver()
    #Set Solver Parameters
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

    # Output file if specified to do so
    if (outputfile != ""):
        output = File(outputfile, "compressed")
        output << (y.split()[0],0.0)

    # Set up time-stepping
    t = 0.0
    N = int((1.0*Tfinal)/dt)

    # Arrays to track Energy, Mass and time
    Energy = np.zeros([1,N+3])
    Mass = np.zeros([1,N+3])
    time = np.zeros([1,N+3])

    # Calculate |Omega| (though trivial for the unit square or unit interval)
    unity = interpolate(Constant(1.0),P2)
    measure = assemble(unity*dx)

    # Energy at time 0 (not physically menaningful)

    a = (inner(grad(r),grad(p1)) + c*p1 + r*p2 )*dx
    L =  - inner(u-m,p1)*dx
    solve(a == L, ww)
    (rr, cc) = ww.split()
    # calculate three parts individually for easy modification to investigate the relationship between and dynamics of them separately
    e_part1 = ((epssi**2)/2.0)*dot(grad(u), grad(u))*dx(mesh)
    e_part2 = 0.25*(1-(u**2))**2*dx(mesh)
    e_part3 = (sigma/2.0)*rr*(u-m)*dx(mesh)
    e1 = assemble(e_part1)
    e2 = assemble(e_part2)
    e3 = assemble(e_part3)
    E = e1+e2+e3;
    time[0,0] = t
    Energy[0,0] = E
    Mass[0,0] = assemble(u*dx)/measure

    count = 1

    ## Time loop
    while (t <= Tfinal):
        t += dt
        ## Solve O-K system
        y0.vector()[:] = y.vector() #Update solution at time step n-1
        solver.solve(problem, y.vector()) #Solve system at time step n
        if (outputfile != ""):
            output << (y.split()[0], t) #Add to file

        ## Track Free Energy at each step
        a = (inner(grad(r),grad(p1)) + c*p1 + r*p2 )*dx
        L =  - inner(u-m,p1)*dx
        solve(a == L, ww)
        (rr, cc) = ww.split()
        e_part1 = ((epssi**2)/2.0)*dot(grad(u), grad(u))*dx(mesh)
        e_part2 = 0.25*(1-(u**2))**2*dx(mesh)
        e_part3 = (sigma/2.0)*rr*(u-m)*dx(mesh)
        e1 = assemble(e_part1)
        e2 = assemble(e_part2)
        e3 = assemble(e_part3)
        E = e1+e2+e3;
        time[0,count] = t
        Energy[0,count] = E

        ## Track mass at each step
        Mass[0,count] = assemble(u*dx)/measure

        count+=1

    ## Output U vector (without dof mapping for now)
    u = project(u,P2)
    uvec = u.vector().array()
    data = {"Energy": Energy, "t": time, "m": Mass, "U": uvec} #Return as dictionary for easy save to .mat file
    return y, data


def OKadaptive(epssi, m, sigma, mesh, n, Tfinal, ICs, tol, outputfile=""):
# Function to simulate the adaptive time stepping algorithm, for a user-specified tolerance parameter.
# dt_min and dt_max must be change in the body of the function

    h = 1.0/n
    # Set up time stepping
    t = 0.0
    dtmin = 0.5*(h**2)
    dtmax = 50*(h**2)
    dt = dtmin
    N = 10000 #max Number of timesteps

    # Build mixed function space for solution of O-K system
    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = FunctionSpace(mesh, V)
    ME = FunctionSpace(mesh, V*V)

    #Define  test functions
    q, v  = TestFunctions(ME)

    # Define functions
    y   = Function(ME)  # current solution
    y0  = Function(ME)  # solution from previous converged step

    # Split mixed functions
    u,  w  = split(y)
    u0, w0 = split(y0)

    # Define Function space, function and trial and test function needed to solve for Energy
    # V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("Real",mesh.ufl_cell(),0)
    W = FunctionSpace(mesh, V*R)
    r, c = TrialFunction(W)
    p1, p2 = TestFunctions(W)
    ww = Function(W)

    # IC's
    ICs = interpolate(ICs,ME)
    y.assign(ICs)
    y0.assign(ICs)

    # Compute dphi/du
    u = variable(u)
    phi    = 0.25*(1-(u**2))**2
    dphidu = diff(phi, u)

    #Optimise
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Weak form of the equations
    L1 = inner(u,q)*dx - inner(u0,q)*dx + dt*dot(grad(w), grad(q))*dx + dt*sigma*inner(u-m,q)*dx
    L2 = inner(w,v)*dx - (epssi**2)*dot(grad(u), grad(v))*dx - inner(dphidu,v)*dx
    L = L1 + L2

    # Compute derivative
    a = derivative(L, y)

    # Create nonlinear problem and Newton solver
    problem = OKsystem(a, L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

    # Output file
    if (outputfile != ""):
        output = File(outputfile, "compressed")

    # Arrays to track Energy, Mass and time
    Energy = np.zeros([1,N+2])
    Mass = np.zeros([1,N+2])
    time = np.zeros([1,N+2])
    InfNorm = np.zeros([1,N+2])
    DTs = np.zeros([1,N+2])

    # Calculate |Omega|
    unity = interpolate(Constant(1.0),P2)
    measure = assemble(unity*dx)

    count = 0
    dt = dtmin
    while (t <= Tfinal):
        time[0,count] = t
        ## Solve O-K system
        y0.vector()[:] = y.vector() #Update solution at time step n-1
        solver.solve(problem, y.vector()) #Solve system at time step n
        if (outputfile != ""):
            output << (y.split()[0], t) #Add to file

        ## Track Free Energy at each step
        # Solve Energy subproblem
        a = (inner(grad(r),grad(p1)) + c*p1 + r*p2 )*dx
        L =  - inner(u-m,p1)*dx
        solve(a == L, ww)
        # Calculate energy
        (rr, cc) = ww.split()
        e_part1 = ((epssi**2)/2.0)*dot(grad(u), grad(u))*dx(mesh)
        e_part2 = 0.25*(1-(u**2))**2*dx(mesh)
        e_part3 = (sigma/2.0)*rr*(u-m)*dx(mesh)
        e1 = assemble(e_part1)
        e2 = assemble(e_part2)
        e3 = assemble(e_part3)
        E = e1+e2+e3
        Energy[0,count] = E

        ## Track mass at each step
        Mass[0,count] = assemble(u*dx)/measure

        ## Calculate new time step

        change = project(u-u0,P2)
        inf_e_norm = max(abs(change.vector().array()))
        InfNorm[0,count] = inf_e_norm
        dt = min(max((tol*1.0)/inf_e_norm , dtmin), dtmax)
        print(dt)
        DTs[0,count] = dt
        t += dt
        count+=1

    u = project(u,P2)
    uvec = u.vector().array()
    data = {"E": Energy, "t": time, "m": Mass, "inf_norm": InfNorm, "Dt": DTs, "U": uvec}
    return y, data


def OKuniformIMX(epssi, m, sigma, mesh, dt, Tfinal, ICs, outputfile=""):


    # Build mixed function space for solution of O-K system
    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = FunctionSpace(mesh, V)
    ME = FunctionSpace(mesh, V*V)

    # Define trial and test functions
    dy    = TrialFunction(ME)
    q, v  = TestFunctions(ME)

    # Define functions
    y   = Function(ME)  # current solution
    y0  = Function(ME)  # solution from previous converged step

    # Split mixed functions
    du, dw = split(dy)
    u,  w  = split(y)
    u0, w0 = split(y0)

    # Define Function space, function and trial and test function needed to solve for Energy
#    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("Real",mesh.ufl_cell(),0)
    W = FunctionSpace(mesh, V*R)

    r, c = TrialFunction(W)
    p1, p2 = TestFunctions(W)
    ww = Function(W)

    # IC's
    ICs = interpolate(ICs,ME)
    y.assign(ICs)
    y0.assign(ICs)

    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Weak form of the equations
    L1 = inner(u,q)*dx - inner(u0,q)*dx + dt*dot(grad(w), grad(q))*dx + dt*sigma*inner(u-m,q)*dx
    L2 = inner(w,v)*dx - (epssi**2)*dot(grad(u), grad(v))*dx - inner(u0**3,v)*dx + inner(u,v)*dx
    L = L1 + L2

    # Output file
    if (outputfile != ""):
        output = File(outputfile, "compressed")
        output << (y.split()[0],0.0)

    # Set up time stepping
    t = 0.0
    N = int(Tfinal//dt)


    count = 0
    while (t <= Tfinal):
        t += dt
        ## Solve O-K system
        y0.vector()[:] = y.vector() #Update solution at time step n-1
        solve(L == 0, y) #Solve system at time step n
        if (outputfile != ""):
            output << (y.split()[0], t) #Add to file
        count+=1

    u = project(u,P2)
    uvec = u.vector().array()
    data = {"U": uvec}
    return y, data
