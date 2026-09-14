# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:46:23 2026

@author: lremilla
"""

from sympy import symbols, Function, Eq, I
from sympy.vector import CoordSys3D

#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols needed

E_til, H_til = symbols(r"\tilde{E} \tilde{H}", cls=Function)
m, eps, mu = symbols("m epsilon mu")
rho_n, rho_PEC = symbols("rho_n rho_PEC")
k_z = symbols("k_z")
t, w = symbols("t omega")

E_rho, E_phi, E_z = symbols("E_rho E_phi E_z", cls=Function)
H_rho, H_phi, H_z = symbols("H_rho H_phi H_z", cls=Function)

E_full = Eq(E_til(rho,phi,z,t),)