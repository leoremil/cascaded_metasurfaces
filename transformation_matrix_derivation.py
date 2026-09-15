# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:46:23 2026

@author: lremilla
"""

from sympy import symbols, Function, Eq, I, exp, Derivative, solve, zeros, init_printing
from sympy.vector import CoordSys3D, curl
from IPython.display import display

init_printing(use_latex=True)
#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols needed

print("Setting up symbols...")
E, H = symbols("E H", cls=Function)
m, eps, mu = symbols("m epsilon_n mu_n")
rho_n, rho_PEC = symbols("rho_n rho_PEC")
k_z = symbols("k_z")
t, w = symbols("t omega")
k_rho = symbols("k_rho")

rho = coordinate_system.rho
phi = coordinate_system.phi
z = coordinate_system.z

E_rho, E_phi, E_z = symbols("E_rho E_phi E_z", cls=Function)
H_rho, H_phi, H_z = symbols("H_rho H_phi H_z", cls=Function)


# the equations for total fields assuming time harmonics
E_tot = (E_rho(rho,phi)*coordinate_system.i + E_phi(rho,phi)*coordinate_system.j + E_z(rho,phi)*coordinate_system.k)*exp(I*w*t)*exp(-I*k_z*z)
H_tot = (H_rho(rho,phi)*coordinate_system.i + H_phi(rho,phi)*coordinate_system.j + H_z(rho,phi)*coordinate_system.k)*exp(I*w*t)*exp(-I*k_z*z)

#todo: define the ansatz for the z-components

print("Solving Maxwell's equations...")
# plug into Maxwell's to get vector equations. Turn into Matrix objects
faraday_law = (curl(E_tot).doit() + mu*Derivative(H_tot,t).doit()).to_matrix(coordinate_system)
ampere_law = (curl(H_tot).doit() - eps*Derivative(E_tot,t).doit()).to_matrix(coordinate_system)

# Isolate each field component. Each row is rho, phi, and z component respectively
H_vec = zeros(3,1)
E_vec = zeros(3,1)

for i, c in enumerate([H_rho(rho,phi), H_phi(rho,phi), H_z(rho,phi)]):
    H_vec[i] = solve(faraday_law[i],c)
for i, c in enumerate([E_rho(rho,phi), E_phi(rho,phi), E_z(rho,phi)]):
    E_vec[i] = solve(ampere_law[i],c)

#Find each component only in terms of H_z and E_z by subbing equations for rho and phi components from Maxwell's
print("Solving for field components in terms of z-components...")
non_z_component_rhss = [H_vec[0], H_vec[1], E_vec[0], E_vec[1]]
non_z_components = [H_rho(rho,phi), H_phi(rho,phi),E_rho(rho,phi), E_phi(rho,phi)]
H_vec_in_z_terms = zeros(3,1)
E_vec_in_z_terms = zeros(3,1)

#go through each component RHS (equation)
for component_rhs_index, component_rhs in enumerate(non_z_component_rhss):
    desired_component = non_z_components[component_rhs_index]
    #For current RHS find the non-z component present
    for search_component_index, search_component in enumerate(non_z_components):
        if solve(component_rhs,search_component):
            #First and second indices are H_rho and H_phi. Third and fourth are E_rho and E_phi.
            #Messy, but solve for desired component (same index as RHS) after subbing in the found non-z component. Subtract desired component as these are expressions and they are implicitly equal to zero.
            search_component_rhs = non_z_component_rhss[search_component_index]
            if component_rhs_index in [0,1]:
                H_vec_in_z_terms[component_rhs_index] = solve(component_rhs.subs(search_component,search_component_rhs) - desired_component,desired_component)
            else:
                E_vec_in_z_terms[component_rhs_index - 2] = solve(component_rhs.subs(search_component,search_component_rhs) - desired_component, desired_component)

#Sub in k_rho for easy comparison to references
H_vec_in_z_terms = H_vec_in_z_terms.subs(eps*mu*w**2-k_z**2,k_rho**2)
E_vec_in_z_terms = E_vec_in_z_terms.subs(eps*mu*w**2-k_z**2,k_rho**2)
#%% Display the field components and verify
display(E_vec_in_z_terms[0])
display(E_vec_in_z_terms[1])
display(H_vec_in_z_terms[0])
display(H_vec_in_z_terms[1])
#%%