# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:46:23 2026
Derive the field transformation matrix starting from the Ampere, Faraday law, and
basic assumptions about the field variation in the waveguide.

z components are assumed to take the form used in Romina's thesis so compare the results here to that document.

Inputs:
    modules_lambdification (dict): each key is a python module and their entries are a string consisting of a comma seperated list of functions to import from those modules.
    
Exports:
    The transformation matrix expression as a numeric python function (transformationMatrix.py)
    The transformation matrix expression as a fully symbolic sympy matrix (transformationMatrixSymbolic.py)
@author: lremilla
"""
from inspect import getsource
from os.path import join

from sympy import symbols, Eq, I, exp, Derivative, solve, init_printing, Matrix, simplify, hankel1, hankel2, Dummy, lambdify, pi, srepr
from sympy.vector import CoordSys3D, curl
from IPython.display import display

from symbol_definitions import m, eps, mu, k_z, t, w, f, k_rho, F_1, F_2, G_1, G_2, H_1m, H_2m, H_1m_p, H_2m_p, E_rho, E_phi, E_z, H_rho, H_phi, H_z, k_rho_expression, rho

#================================================
#Put the imports for the final python function export here. Requires trial and error to get right
file_name_numeric = "transformationMatrix.py"
modules_lambdification = {"mpmath":"pi, sqrt, hankel1, hankel2, mpf, matrix"}
#================================================

file_name_symbolic = "transformationMatrixSymbolic.txt"
file_name_radial_H = "radialHSymbolic.txt"

init_printing(use_latex=True)

#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols and expressions needed
print("Setting up symbols and expressions...")

M = Matrix(4, 4, lambda i, j: symbols(f'M_{i+1}_{j+1}'))#matrix of symbols to be solved for the transformation matrix

#Coordinate system position variables. THESE ARE DISTINCT FROM USER DEFINED SYMBOLS AND ARE NOT FRIENDLY FOR EXPORT OR IMPORT!!!!
_rho = coordinate_system.rho
_phi = coordinate_system.phi
_z = coordinate_system.z

#Modal coefficient vector
coefficients = Matrix([F_1, F_2, G_1, G_2])

# the equations for total fields assuming time harmonics, sinusoidal longitudinal and azimuthal variation
E_tot = (E_rho(_rho)*coordinate_system.i + E_phi(_rho)*coordinate_system.j + E_z(_rho)*coordinate_system.k)*exp(I*w*t)*exp(-I*k_z*_z)*exp(-I*m*_phi)
H_tot = (H_rho(_rho)*coordinate_system.i + H_phi(_rho)*coordinate_system.j + H_z(_rho)*coordinate_system.k)*exp(I*w*t)*exp(-I*k_z*_z)*exp(-I*m*_phi)

#define the ansatz for the z-components
E_z_expression = (F_1*H_1m(k_rho*rho) + F_2*H_2m(k_rho*rho))
H_z_expression = (G_1*H_1m(k_rho*rho) + G_2*H_2m(k_rho*rho))

print("Solving Maxwell's equations...")
# Plug total field equations into Maxwell's to get vector equations. Turn into Matrix objects. Must be written RHS - LHS = 0 because the Eq() function doesn't work for vector calc stuff apparently
#Remove the special coordinate _rho for the user defined rho
faraday_law = (curl(E_tot).doit() + mu*Derivative(H_tot,t).doit()).to_matrix(coordinate_system).subs(_rho,rho)
ampere_law = (curl(H_tot).doit() - eps*Derivative(E_tot,t).doit()).to_matrix(coordinate_system).subs(_rho,rho)

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
non_z_component_rhss = [H_eqs.rhs[0], H_eqs.rhs[1], E_eqs.rhs[0], E_eqs.rhs[1]]#right hand sides
non_z_components = [H_eqs.lhs[0], H_eqs.lhs[1], E_eqs.lhs[0], E_eqs.lhs[1]]
H_vec_in_z_terms = [0,0,H_z(rho)]
E_vec_in_z_terms = [0,0,E_z(rho)]

#go through each component RHS (equation)
for desired_component_rhs_index, desired_component_rhs in enumerate(non_z_component_rhss):
    desired_component = non_z_components[desired_component_rhs_index]
    
    #For current RHS find the non-z component present
    for search_component_index, search_component in enumerate(non_z_components):   
        if solve(desired_component_rhs,search_component):
            search_component_rhs = non_z_component_rhss[search_component_index]
            
            #Messy, but solve for desired component (same index as RHS) after subbing in the found non-z component. Subtract desired component as these are expressions and they are implicitly equal to zero.
            #First and second indices are H_rho and H_phi. Third and fourth are E_rho and E_phi.
            if desired_component_rhs_index in [0,1]: 
                H_vec_in_z_terms[desired_component_rhs_index] = solve(desired_component_rhs.subs(search_component,search_component_rhs) - desired_component,desired_component)[0]
                
            else:
                E_vec_in_z_terms[desired_component_rhs_index - 2] = solve(desired_component_rhs.subs(search_component,search_component_rhs) - desired_component, desired_component)[0]

#Sub in k_rho for easy comparison to references
H_vec_in_z_terms = Matrix(H_vec_in_z_terms).subs(k_rho_expression**2,k_rho**2)
E_vec_in_z_terms = Matrix(E_vec_in_z_terms).subs(k_rho_expression**2,k_rho**2)

#%% Display the field components and verify with Romina's thesis
display(E_vec_in_z_terms[0].expand())
display(E_vec_in_z_terms[1].expand())
display(E_vec_in_z_terms[2])
display(H_vec_in_z_terms[0].expand())
display(H_vec_in_z_terms[1].expand())
display(H_vec_in_z_terms[2])

#%% Create a matrix equation: on one side have the field components after subbing in the expressions for the z components, on the other have the modal amplitude coefficients multiplied by an arbitrary matrix which is the field transformation matrix in Romina's thesis. Solve for the elements of this matrix.

# Order of components will differ from Romina's. She does it one way then reorders it for some reason in her code anyways.
#My order: [E_phi, E_z, H_phi, H_z], and then [F_1, F_2, G_1, G_2] for coefficients
print("Solve for the transformation matrix...")
tangential_fields = Matrix(E_vec_in_z_terms[1:3,0].col_join(H_vec_in_z_terms[1:3,0])).subs({
    E_z(rho):E_z_expression,
    H_z(rho):H_z_expression
    }).doit()

transformation_definition_eq = Eq(tangential_fields,M*coefficients)

#Way to automate solving for the tranformation matrix elements by inspection. Matches the matrix element value to the modal coefficient factor
matrix_elements = {}
for row in range(4):
    row_eq_lhs = transformation_definition_eq.lhs[row].expand().collect(coefficients)
    row_eq_rhs = transformation_definition_eq.rhs[row].collect(coefficients)
    print("current row equations:")
    display(row_eq_lhs), display(row_eq_rhs)
    print("\n")
    
    for mode_coef in coefficients:
        matrix_elements[row_eq_rhs.coeff(mode_coef)] = row_eq_lhs.coeff(mode_coef)
        print("Current mode coeff:"),display(mode_coef)
        print("Detected matrix element:"),display(row_eq_rhs.coeff(mode_coef))
        print("Detected matrix element value:"),display(row_eq_lhs.coeff(mode_coef))
        print("")#placeholder for breakpoint

M_final = M.subs(matrix_elements)

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
#%% Create a python function that generates a transformation matrix that is useable in numeric calculations
# First sub in some expressions so the matrix depends only on frequency, material params, radius under consideration, and mode order

#Create a temporary symbolic expression for M before it gets exported. All Hankel functions and their derivatives are subbed in
M_final_numeric_syms = M_final.subs({
    H_1m(k_rho*rho) : hankel1(m,k_rho*rho),
    H_2m(k_rho*rho) : hankel2(m,k_rho*rho)
    })    

M_final_numeric_syms = M_final_numeric_syms.xreplace({
    Derivative(H_1m(dummy_var),dummy_var).subs(dummy_var,k_rho*rho) : hankel1(m,dummy_var).diff(dummy_var).subs(dummy_var,k_rho*rho),
    Derivative(H_2m(dummy_var),dummy_var).subs(dummy_var,k_rho*rho) : hankel2(m,dummy_var).diff(dummy_var).subs(dummy_var,k_rho*rho)
    })

M_final_numeric_syms = M_final_numeric_syms.subs(k_rho, k_rho_expression)
M_final_numeric_syms = M_final_numeric_syms.subs(w, 2*pi*f)

M_final_numeric = lambdify([f, k_z, m, eps, mu, rho], M_final_numeric_syms, modules = list(modules_lambdification))

#force mpmath matrix function
#todo: find a way to make this better so you don't get problems if you need to use numpy instead
function_code_numeric = getsource(M_final_numeric).replace("_lambdifygenerated","transformationMatrix").replace("ImmutableDenseMatrix","matrix")

with open(join("numeric_expressions",file_name_numeric),"w") as file:
    for module in list(modules_lambdification):
        file.write(f"from {module} import {modules_lambdification[module]}\n\n")
    file.write(function_code_numeric)
#%% Same but keep it as a symbolic expression for the dispersion relation derivation

#Replace all evaluated derivatives of Hankel functions with a H'_1m or H'_2m symbol. Other symbolic calcs are easier leaving the derivatives un-expanded.
M_final_symbolic = M_final.xreplace({
    Derivative(H_1m(dummy_var),dummy_var).subs(dummy_var,k_rho*rho) : H_1m_p(k_rho*rho),
    Derivative(H_2m(dummy_var),dummy_var).subs(dummy_var,k_rho*rho) : H_2m_p(k_rho*rho)
    })
with open(join("symbolic_expressions", file_name_symbolic),'w') as file:
    file.write(srepr(M_final_symbolic))