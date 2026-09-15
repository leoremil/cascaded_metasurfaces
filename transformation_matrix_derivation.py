# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:46:23 2026

@author: lremilla
"""

from sympy import symbols, Function, Eq, I, exp, Derivative, solve, zeros
from sympy.vector import CoordSys3D, divergence, curl

#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols needed


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


# the equations for total fields assuming time harmonics
E_tot = (E_rho(rho,phi,z)*coordinate_system.i + E_phi(rho,phi,z)*coordinate_system.j + E_z(rho,phi,z)*coordinate_system.k)*exp(I*w*t)
H_tot = (H_rho(rho,phi,z)*coordinate_system.i + H_phi(rho,phi,z)*coordinate_system.j + H_z(rho,phi,z)*coordinate_system.k)*exp(I*w*t)

# plug into Maxwell's to get vector equations. Turn into Matrix objects
faraday_law = (curl(E_tot).doit() + mu*Derivative(H_tot,t).doit()).to_matrix(coordinate_system)
ampere_law = (curl(H_tot).doit() - eps*Derivative(E_tot,t).doit()).to_matrix(coordinate_system)

# Isolate each field component. Each row is rho, phi, and z component respectively
H_vec = zeros(3,1)
E_vec = zeros(3,1)

for i, c in enumerate([H_rho(rho,phi,z), H_phi(rho,phi,z), H_z(rho,phi,z)]):
    H_vec[i] = solve(faraday_law[i],c)
for i, c in enumerate([E_rho(rho,phi,z), E_phi(rho,phi,z), E_z(rho,phi,z)]):
    E_vec[i] = solve(ampere_law[i],c)
    
#Find each component only in terms of H_z and E_z by subbing in derivatives of other components
