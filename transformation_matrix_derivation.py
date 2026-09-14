# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:46:23 2026

@author: lremilla
"""

from sympy import symbols, Function, Eq, I, exp
from sympy.vector import CoordSys3D

#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols needed

E_til, H_til = symbols(r"\tilde{E} \tilde{H}", cls=Function)
E, H = symbols("E H", cls=Function)
m, eps, mu = symbols("m epsilon mu")
rho_n, rho_PEC = symbols("rho_n rho_PEC")
k_z = symbols("k_z")
t, w = symbols("t omega")

rho = coordinate_system.rho
phi = coordinate_system.phi
z = coordinate_system.z

E_rho, E_phi, E_z = symbols("E_rho E_phi E_z", cls=Function)
H_rho, H_phi, H_z = symbols("H_rho H_phi H_z", cls=Function)

E_full = Eq(E_til(rho,phi,z,t),E(rho,m,k_z)*exp(-I*m*phi)*exp(-I*k_z*z)*exp(I*w*t))
H_full = Eq(H_til(rho,phi,z,t),H(rho,m,k_z)*exp(-I*m*phi)*exp(-I*k_z*z)*exp(I*w*t))

