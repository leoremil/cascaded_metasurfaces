# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:46:23 2026
Derive the field transformation matrix starting from the Ampere, Faraday law, and
basic assumptions about the field variation in the waveguide.

This version has the sinusoidal variation assumed at the beginning like in Romina's work
@author: lremilla
"""

from sympy import symbols, Function, Eq, I, exp, Derivative, solve, init_printing, Matrix, simplify, hankel1, hankel2, Dummy
from sympy.vector import CoordSys3D, curl
from IPython.display import display

init_printing(use_latex=True)
#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols needed

print("Setting up symbols...")
E, H = symbols("E H", cls=Function) #E and H fields
m, eps, mu = symbols("m epsilon_n mu_n")#azimuthal order and material params
rho_n, rho_PEC = symbols("rho_n rho_PEC")#radii of the MTS
k_z = symbols("k_z")#longitudinal propagation constant
t, w = symbols("t omega")#time and angular frequency
k_rho = symbols("k_rhon")#radial propagation constant
F_1, F_2, G_1, G_2 = symbols("F_1n F_2n G_1n G_2n")#modal amplitude coefficients
H_1m, H_2m = symbols("H^{(1)}_m H^{(2)}_m", cls=Function)#Hankel functions
M = Matrix(4, 4, lambda i, j: symbols(f'M_{i+1}_{j+1}'))#matrix of symbols to be solved for the transformation matrix

#Coordinates
rho = coordinate_system.rho
phi = coordinate_system.phi
z = coordinate_system.z

#field components
E_rho, E_phi, E_z = symbols("E_rho E_phi E_z", cls=Function)
H_rho, H_phi, H_z = symbols("H_rho H_phi H_z", cls=Function)

#Modal coefficient vector
coefficients = Matrix([F_1, F_2, G_1, G_2])

# the equations for total fields assuming time harmonics, sinusoidal longitudinal and azimuthal variation
E_tot = (E_rho(rho)*coordinate_system.i + E_phi(rho)*coordinate_system.j + E_z(rho)*coordinate_system.k)*exp(I*w*t)*exp(-I*k_z*z)*exp(-I*m*phi)
H_tot = (H_rho(rho)*coordinate_system.i + H_phi(rho)*coordinate_system.j + H_z(rho)*coordinate_system.k)*exp(I*w*t)*exp(-I*k_z*z)*exp(-I*m*phi)

#define the ansatz for the z-components
E_z_expression = (F_1*H_1m(k_rho*rho) + F_2*H_2m(k_rho*rho))
H_z_expression = (G_1*H_1m(k_rho*rho) + G_2*H_2m(k_rho*rho))

print("Solving Maxwell's equations...")
# Plug total field equations into Maxwell's to get vector equations. Turn into Matrix objects. Must be written RHS - LHS = 0 because the Eq() function doesn't work for vector calc stuff apparently
faraday_law = (curl(E_tot).doit() + mu*Derivative(H_tot,t).doit()).to_matrix(coordinate_system)
ampere_law = (curl(H_tot).doit() - eps*Derivative(E_tot,t).doit()).to_matrix(coordinate_system)

# Isolate each field component. Each row is rho, phi, and z component respectively
H_eqs = Eq(Matrix([H_rho(rho),
                   H_phi(rho),
                   H_z(rho)]),
           Matrix([solve(faraday_law[0],H_rho(rho))[0],
                   solve(faraday_law[1],H_phi(rho))[0],
                   solve(faraday_law[2],H_z(rho))[0]]))

E_eqs = Eq(Matrix([E_rho(rho),
                   E_phi(rho),
                   E_z(rho)]),
           Matrix([solve(ampere_law[0],E_rho(rho))[0],
                   solve(ampere_law[1],E_phi(rho))[0],
                   solve(ampere_law[2],E_z(rho))[0]]))



#Find each component only in terms of H_z and E_z by subbing equations for rho and phi components from Maxwell's
print("Solving for field components in terms of z-components...")
non_z_component_rhss = [H_eqs.rhs[0], H_eqs.rhs[1], E_eqs.rhs[0], E_eqs.rhs[1]]
non_z_components = [H_eqs.lhs[0], H_eqs.lhs[1], E_eqs.lhs[0], E_eqs.lhs[1]]
H_vec_in_z_terms = [0,0,H_z(rho)]
E_vec_in_z_terms = [0,0,E_z(rho)]

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
                H_vec_in_z_terms[component_rhs_index] = solve(component_rhs.subs(search_component,search_component_rhs) - desired_component,desired_component)[0]
            else:
                E_vec_in_z_terms[component_rhs_index - 2] = solve(component_rhs.subs(search_component,search_component_rhs) - desired_component, desired_component)[0]

#Sub in k_rho for easy comparison to references
H_vec_in_z_terms = Matrix(H_vec_in_z_terms).subs(eps*mu*w**2-k_z**2,k_rho**2)
E_vec_in_z_terms = Matrix(E_vec_in_z_terms).subs(eps*mu*w**2-k_z**2,k_rho**2)
#%% Display the field components and verify with Romina's thesis
display(E_vec_in_z_terms[0].expand())
display(E_vec_in_z_terms[1].expand())
display(H_vec_in_z_terms[0].expand())
display(H_vec_in_z_terms[1].expand())
#%% Create a matrix equation: on one side have the field components after subbing in the expressions for the z components, on the other have the modal amplitude coefficiencts multiplied by an arbitrary matrix which is the field transformation matrix in Romina's thesis. Solve for these components.
# Order of components will differ from Romina's. She does it one way then reorders it for some reason in her code anyways.
#My order: [E_phi, E_z, H_phi, H_z], and then [F_1, F_2, G_1, G_2] for coefficients
print("Solve for the transformation matrix...")
tangential_fields = Matrix(H_vec_in_z_terms[1:3,0].col_join(E_vec_in_z_terms[1:3,0])).subs({E_z(rho):E_z_expression,H_z(rho):H_z_expression}).doit()

transformation_definition_eq = Eq(tangential_fields,M*coefficients)

#Way to basically solve for the tranformation matrix coefficients by inspection. Matches the matrix element value to the modal coefficient factor
matrix_coefficients = {}
for row in range(4):
    row_eq_lhs = transformation_definition_eq.lhs[row].expand().collect(coefficients)
    row_eq_rhs = transformation_definition_eq.rhs[row].collect(coefficients)
    print("current row equations:")
    display(row_eq_lhs), display(row_eq_rhs)
    print("\n")
    for mode_coef in coefficients:
        matrix_coefficients[row_eq_rhs.coeff(mode_coef)] = row_eq_lhs.coeff(mode_coef)
        print("Current mode coeff:"),display(mode_coef)
        print("Detected matrix element:"),display(row_eq_rhs.coeff(mode_coef))
        print("Detected matrix element value:"),display(row_eq_lhs.coeff(mode_coef))
        print("")#placeholder

M_final = M.subs(matrix_coefficients)

#%% Sub in hankel functions for their symbols to perform final verification
transformation_verification_eq = Eq(tangential_fields,M_final*coefficients).subs({
    H_1m(k_rho*rho) : hankel1(m,k_rho*rho),
    H_2m(k_rho*rho) : hankel2(m,k_rho*rho)
    })
#Evaluated derivatives of a function symbol must be substituted seperately. Obtain the dummy variable so we can sub in the derivatives.
dummy_var = list(transformation_verification_eq.atoms(Dummy))[0]
transformation_verification_eq = transformation_verification_eq.xreplace({
    Derivative(H_1m(dummy_var),dummy_var).subs(dummy_var,k_rho*rho) : hankel1(m,dummy_var).diff(dummy_var).subs(dummy_var,k_rho*rho),
    Derivative(H_2m(dummy_var),dummy_var).subs(dummy_var,k_rho*rho) : hankel2(m,dummy_var).diff(dummy_var).subs(dummy_var,k_rho*rho)
    })

if simplify(transformation_verification_eq.expand()):
    print("Success: M matches definition.")