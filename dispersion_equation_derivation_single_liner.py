# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:47:02 2026
Derivation of the dispersion equation using the transformation matrix derived in the other script
@author: lremilla
"""
from os.path import join

from sympy import init_printing, symbols, Matrix, Function, sympify, Eq, pi, hankel1, hankel2,sqrt
from IPython.display import display

from symbol_definitions import m, eps, mu, k_z, t, w, f, k_rho, H_1m, H_2m, rho, H_1m_p, H_2m_p, H_z, H_phi

init_printing(use_latex=True)

#Define the symbols needed

print("Setting up symbols...")
rho_MTS, rho_PEC = symbols("rho_MTS rho_PEC")
F_0, G_0 = symbols("F_0 G_0")
F_12, F_22, G_12, G_22 = symbols("F_12 F_22 G_12 G_22")

#transformation matrix for any radius within region n
with open(join("symbolic_expressions", "transformationMatrixSymbolic.txt"),'r') as file:
    M = sympify(file.read())

#Radial H component
with open(join("symbolic_expressions", "radialHSymbolic.txt"),'r') as file:
    H_radial = sympify(file.read())

#M at PEC and MTS radii
M_PEC = M.subs(rho,rho_PEC)
M_MTS = M.subs(rho,rho_MTS)

#%% Find modal coefficients at the origin
#Hankel functions are Jm(x)+jYm(x) or Jm(x)-jYm(x). Ym(x) goes to infinity at x=0. Solving for modal coefficients such that the Ym(x) term is zero leads to F1=F2=F0/2 and G1=G2=G0/2. Pretty easy to do on paper and hard to mess up so just state it here
coefficients_region1 = Matrix([F_0, F_0, G_0, G_0])/2

#%% Find modal coefficients at the PEC. Tangential electric fields must be zero. solve for the coefficients using the transformation matrix formalism:
print("Finding coefficients for fields in region 2...")
coefficients_region2 = Matrix([F_12, F_22, G_12, G_22])

tangential_fields_PEC = Matrix([0 ,0 ,H_z(rho_PEC) , H_phi(rho_PEC)])

PEC_fields_eq = Eq(tangential_fields_PEC, M_PEC*coefficients_region2)

