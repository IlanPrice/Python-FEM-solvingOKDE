## Script to run Ohta-Kawasaki simulations

from dolfin import *
import random
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from sympy.utilities.codegen import ccode
from sympy import symbols
import sympy as sp
from OKLibrary import * #Import functions for the special topic which are saved as a library



#Class representing the intial conditions, to create random initial conditions
class InitialConditions(Expression):
    def __init__(self, mass, s, **kwargs):
        random.seed(s)
        self.mass = mass
    def eval(self, values, x):
        values[0] = self.mass + 2*(0.5 - random.random()) #first term = mass value
        values[1] = 0.0
    def value_shape(self):
        return (2,)

#########################################################

    # Solving the Ohta Kawasaki System with fully implicit
    # scheme, on unit sqaure mesh.

#########################################################


### Model parameters
#epssi    = 0.08
#m       = 0 # mass
#sigma   =  10.0
#dt = epssi**2
#
#Tfinal = 8
### Mesh
#n = 40
#h = 1.0/n
#
## mesh = IntervalMesh(n,0,1)
#mesh = UnitSquareMesh(n,n)
#
### Initial conditions (1)
#ICs = InitialConditions(m,1,degree=1)
#
#y, data = OKuniform(epssi, m, sigma, mesh, dt, Tfinal, ICs)
#
##io.savemat("uniformwithmass_data", data)
#io.savemat("uniformnomass_data", data)

######################################

    # Varying Sigma

######################################

# ## Model parameters
# epssi    = 0.02
# m       = 0.4 # mass
# dt      = epssi**2
#
# Tfinal = 2

# #Mesh
# n = 40
# h = 1.0/n
# print(h)
# # mesh = IntervalMesh(n,0,1)
# mesh = UnitSquareMesh(n,n)
#
# ## Initial conditions -- 1
# ICs = InitialConditions(m, 2, degree=1)
#
# sigArray = np.array([2, 20, 200])
#
# for sigma in np.nditer(sigArray):
#     u, data = OKuniform(epssi, m, sigma, mesh, dt, Tfinal, ICs, 'PVD/varysigma/sigma_{}.pvd'.format(sigma))
#     io.savemat('VarySigma_{}'.format(sigma), data)

######################################

    # Adaptive Time stepping

######################################

# ## Model parameters
# epssi    = 0.02
# m       = 0.4 # mass
# sigma   =  40
#
# ## Mesh
# n = 50
# h = 1.0/n
#
# mesh = IntervalMesh(n,0,1)
# #mesh = UnitSquareMesh(n,n)
#
# ## Initial conditions -- 1 Random
# # ICs = InitialConditions(m, 2, degree=1)
#
# ## Initial condiditions -- 2 Initial conditions used by Li et. al, plus mass constant
# ##  Use symbolic differentiation to create w
# x = symbols('x[0]')
# u_ex = 0.1*sp.cos(20*sp.pi*x) + 0.1*sp.cos(30*sp.pi*x*x) + 0.1*sp.cos(40*sp.pi*x*x*x) + 0.4 # Initial condition for u
# u_ex_xx = u_ex.diff(x,2)
# w_ex = -(epssi**2)*u_ex_xx + u_ex*(u_ex**2 - 1) # Exact solution for w
# str1 = ccode(u_ex).replace('M_PI', 'pi') # These lines convert the sympy expressions into c++ code which can then be turned into an expression and interpolated
# str2 = ccode(w_ex).replace('M_PI', 'pi')
# ICs = Expression((str1,str2), degree = 1)
#
#
#
# ## Store norm of the difference between the smallest time step solution and the other solutions at times in T
#
# TfinalError = np.zeros([2,3])
#
# T = np.array([0.05, 0.5, 1])
# count = 0
# tol = 2e-7
#
# for i in np.nditer(T):
#     u1, data1 = OKuniform(epssi, m, sigma, mesh, 0.5*(h**2), i, ICs)
#     u2, data2 = OKuniform(epssi, m, sigma, mesh, 25*(h**2), i, ICs)
#     uadapt, datadap = OKadaptive(epssi, m, sigma, mesh, n, i, ICs, tol)
#
#     io.savemat('uniform_0_5_hh_tol_{}_sig_{}_ep{}'.format(tol,sigma, epssi), data1)
#     io.savemat('uniform_25_hh_tol_{}_sig_{}_ep{}'.format(tol,sigma, epssi), data2)
#     io.savemat('adaptive_max_50_tol_{}_sig_{}_ep{}'.format(tol,sigma, epssi), datadap)
#     TfinalError[0,count] = errornorm(u1.split()[0],u2.split()[0])
#     TfinalError[1,count] = errornorm(u1.split()[0],uadapt.split()[0])
#     count+=1
#     plt.figure()
#     plt.plot(data1["U"], label = r'0.5 h^2')
#     plt.plot(data2["U"], label = r'25 h^2')
#     plt.plot(datadap["U"], label = r'Adaptive')
#     plt.legend()
#     plt.savefig('tfinal_{}_tol_{}_sig_{}_ep{}.pdf'.format(i, tol, sigma, epssi))
#
# io.savemat('TfinalError_tol_{}_sig_{}_ep{}_fine_Fri'.format(tol, sigma, epssi),{"TfinalError": TfinalError} )
#


######################################

    # Implicit-explicit splitting

######################################

# ## Model parameters
# epssi    = 0.08
# m       = 0 # mass
# sigma   =  10.0
#
# ##Mesh
# n = 40
# h = 1.0/n
#
# #mesh = IntervalMesh(n,0,1)
# mesh = UnitSquareMesh(n,n)
#
#
# ICs = InitialConditions(m, 2, degree=1)
# dt = 2.2*epssi**2 # Step size: we increased this until the newton solver stopped converging.
# Tfinal = 500*dt

# y, data = OKuniformIMX(epssi, m, sigma, mesh, dt, Tfinal, ICs)
