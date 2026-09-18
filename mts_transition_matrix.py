# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:27:33 2026
Derive the transition matrix for the tangential fields for the MTS

@author: lremilla
"""
from os.path import join

from sympy import Eq, Matrix, init_printing, Function, symbols,solve
from IPython.display import display

from symbol_definitions import Y_pp, Y_zz, Y_zp, Y_pz, H_z, H_phi, E_z, E_phi, rho

init_printing(use_latex=True)

#symbol definitions unique to this file
H_1phi, H_1z, E_1phi, E_1z = symbols("H_1phi H_1z E_1phi E_1z", cls=Function)
H_2phi, H_2z, E_2phi, E_2z = symbols("H_2phi H_2z E_2phi E_2z", cls=Function)

T_MTS = Matrix(4, 4, lambda i, j: symbols(f'T_{i+1}_{j+1}')) #unknown elements of the transition matrix

#difference in tangential H across the MTS
delta_H = Matrix([
    -(H_2z(rho)-H_1z(rho)),
      H_2phi(rho)-H_1phi(rho)
                  ])

#average E across the MTS
average_E = Matrix([
    (E_2phi(rho)+E_1phi(rho))/2,
    (E_2z(rho)+E_1z(rho))/2])

#Surface admittance tensor of the MTS
Y = Matrix([
    [Y_pp, Y_pz],
    [Y_zp, Y_zz]
    ])

GSTC = Eq(delta_H,Y*average_E).expand()

#%% Reorder the GSTC into a field transition matrix which gives the fields on the inner surface from the fields on the outer surface

T_MTS_definition = Eq(Matrix([E_1phi(rho), E_1z(rho), H_1phi(rho), H_1z(rho)]),T_MTS*Matrix([E_2phi(rho), E_2z(rho), H_2phi(rho), H_2z(rho)]))

#Isolate the region 1 H fields in the GSTC. Sub them into the definition for T_MTS and match matrix elements. enforce continuity of tangential electric fields.
T_MTS_definition = T_MTS_definition.subs({
    H_1z(rho):GSTC.rhs[0]+H_2z(rho),
    H_1phi(rho):-1*(GSTC.rhs[1]-H_2phi(rho)),
    E_1z(rho):E_2z(rho),
    E_1phi(rho):E_2phi(rho)})

#Match coefficients by using the following rules:
#   -Tangential E is continous, i.e. E_2phi = E_1phi, E_z1 = E_z2 so only the z-component can match the z component!
#   -Only electric currents can cause a discontinuity in H, i.e. H_z1 = admittance*E_tangential + H_z2, H_phi cannot affect H_z or vice versa with this type of structure!
