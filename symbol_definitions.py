# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:50:42 2026
Contains definitions of all symbols to be used throughout this  project
@author: lremilla
"""

from sympy import symbols, Function, sqrt

m, eps, mu = symbols("m epsilon_n mu_n")#azimuthal order and material params
k_z = symbols("k_z")#longitudinal propagation constant
t, w, f = symbols("t omega f")#time and angular frequency and frequency
k_rho = symbols("k_rhon")#radial propagation constant
F_1, F_2, G_1, G_2 = symbols("F_1n F_2n G_1n G_2n")#modal amplitude coefficients
H_1m, H_2m = symbols("H^{(1)}_m H^{(2)}_m", cls=Function)#Hankel functions
H_1m_p, H_2m_p = symbols("H^{(1)}'_m H^{(2)}'_m", cls=Function)#Hankel function derivatives
x = symbols("x")#placeholder variable
rho = symbols("rho")


#field components
E_rho, E_phi, E_z = symbols("E_rho E_phi E_z", cls=Function)
H_rho, H_phi, H_z = symbols("H_rho H_phi H_z", cls=Function)

#expression for k_rho
k_rho_expression = sqrt(eps*mu*w**2 - k_z**2)