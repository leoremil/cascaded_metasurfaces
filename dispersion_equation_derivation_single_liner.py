# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:47:02 2026
Derivation of the dispersion equation using the transformation matrix derived in the other script
@author: lremilla
"""
from os.path import join

from sympy import init_printing, symbols, Matrix, Function, sympify, Eq, pi, hankel1, hankel2, sqrt, solve
from IPython.display import display

from symbol_definitions import m, eps, mu, k_z, t, w, f, k_rho, H_1m, H_2m, rho, H_1m_p, H_2m_p, H_z, H_phi, F_1, F_2, G_1, G_2

init_printing(use_latex=True)

#Define the symbols needed

print("Setting up symbols...")
rho_MTS, rho_PEC = symbols("rho_MTS rho_PEC")
F_0, G_0 = symbols("F_0 G_0")
F_12, F_22, G_12, G_22 = symbols("F_12 F_22 G_12 G_22")

#transformation matrix for any radius within region n
with open(join("symbolic_expressions", "transformationMatrixSymbolic.txt"),'r') as file:
    M = sympify(file.read())

#M at PEC and MTS radii
M_PEC = M.subs(rho,rho_PEC)
M_MTS = M.subs(rho,rho_MTS)
g_1, g_2 = symbols("g_1 g_2")
F_PEC, G_PEC = symbols("F_PEC G_PEC")

#%% Find modal coefficients at the origin
#Hankel functions are Jm(x)+jYm(x) or Jm(x)-jYm(x). Ym(x) goes to infinity at x=0. Solving for modal coefficients such that the Ym(x) term is zero leads to F1=F2=F0/2 and G1=G2=G0/2. Pretty easy to do on paper and hard to mess up so just state it here
coefficients_region1 = Matrix([F_0, F_0, G_0, G_0])/2

#%% Find modal coefficients at the PEC. Tangential electric fields and radial H must be zero. solve for the coefficients using the transformation matrix formalism:
print("Finding coefficients for fields in region 2...")
coefficients_region2 = Matrix([F_12, F_22, G_12, G_22])

PEC_tangential_fields_eq = Eq(Matrix([0,0,H_phi(rho_PEC),H_z(rho_PEC)]),M_PEC*coefficients_region2)

PEC_zero_components = PEC_tangential_fields_eq.rhs[0:2,0]
#%% Display rows of equations for computation
print("Zeroed field components at the PEC initially:")
display(PEC_zero_components[0])
display(PEC_zero_components[1])

#%% Observe that the terms for F_12 and F_22 in row 2 are present in row 1 but differ by a factor of -(k_z*m)/(k_rho**2*rho_PEC). Thus we can multiply row 2 by (k_z*m)/(k_rho**2*rho_PEC) and add it to row 1 to cancel the F_12 and F_22 terms there. Also solve for F_12 in terms of F_22
PEC_zero_components_step1 = Matrix([
    (PEC_zero_components[0] + (k_z*m)/(k_rho**2*rho_PEC)*PEC_zero_components[1]).expand().simplify().expand(),
                                    PEC_zero_components[1]
                                    ])
F_12_expression = solve(PEC_zero_components[1], F_12)[0]
#%% Display rows of equations again for computation
print("Zeroed field components at PEC step 1:")
display(PEC_zero_components_step1[0])
display(PEC_zero_components_step1[1])
#%% Observe that row 1 now shows that G_12 and G_22 are proportional to eachother as well. solve for G_12 in terms of G_22
G_12_expression = solve(PEC_zero_components_step1[0],G_12)[0]
#%% At this point, the coefficients are the same as in the thesis. Solve for the proportionality factors of F_12, and G_12 like in the thesis
display(F_12_expression)
display(G_12_expression)
#%%Solve for the proportionality factors
g_1_expression = F_12_expression.coeff(F_22)
g_2_expression = G_12_expression.coeff(G_22)