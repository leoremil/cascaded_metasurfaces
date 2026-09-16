# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:47:02 2026
Derivation of the dispersion equation using the transformation matrix derived in the other script
@author: lremilla
"""

from sympy import init_printing, symbols, Matrix, Function, sympify, Eq, pi, hankel1, hankel2,sqrt
from IPython.display import display

init_printing(use_latex=True)

#Define the symbols needed

print("Setting up symbols...")
E, H = symbols("E H", cls=Function) #E and H fields
m, eps, mu = symbols("m epsilon_n mu_n")#azimuthal order and material params
k_z = symbols("k_z")#longitudinal propagation constant
t, w, f = symbols("t omega f")#time and angular frequency and frequency
k_rho = symbols("k_rhon")#radial propagation constant
H_1m, H_2m = symbols("H^{(1)}_m H^{(2)}_m", cls=Function)#Hankel functions
rho = symbols("rho")
rho_MTS, rho_PEC = symbols("rho_MTS rho_PEC")
F_0, G_0 = symbols("F_0 G_0")
F_12, F_22, G_12, G_22 = symbols("F_12 F_22 G_12 G_22")
H_z, H_phi = symbols("H_z, H_phi")

H_1m_prime, H_2m_prime = symbols("H_{m}^{(1)}' H_{m}^{(2)}'", cls=Function)
#transformation matrix for any radius within region n
with open("transformationMatrixSymbolic.txt",'r') as file:
    M = sympify(file.read())

#M at PEC and MTS radii
M_PEC = M.subs(rho,rho_PEC)
M_MTS = M.subs(rho,rho_MTS)

#%% Find modal coefficients at the origin
#Hankel functions are Jm(x)+jYm(x) or Jm(x)-jYm(x). Ym(x) goes to infinity at x=0. Solving for modal coefficients such that the Ym(x) term is zero leads to F1=F2=F0/2 and G1=G2=G0/2. Pretty easy to do on paper and hard to mess up so just state it here
coefficients_region1 = Matrix([F_0, F_0, G_0, G_0])/2

#%% Find modal coefficients at the PEC. Tangential electric fields must be zero. solve for the coefficients using the transformation matrix formalism:
print("Finding coefficients for fields in region 2...")
coefficients_region2 = Matrix([F_12, F_22, G_12, G_22])

tangential_fields_PEC = Matrix([0 ,0 ,H_z , H_phi])

PEC_fields_eq = Eq(tangential_fields_PEC, M_PEC*coefficients_region2).subs({2*pi*f:w,
                                                                            sqrt(eps*mu*w**2-k_z**2):k_rho,
                                                                            hankel1(m-1,k_rho*rho_PEC)/2-hankel1(m+1,k_rho*rho_PEC)/2: H_1m_prime(k_rho*rho_PEC),
                                                                            hankel2(m-1,k_rho*rho_PEC)/2-hankel2(m+1,k_rho*rho_PEC)/2: H_2m_prime(k_rho*rho_PEC)})

