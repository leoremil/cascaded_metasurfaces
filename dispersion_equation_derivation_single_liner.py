# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:47:02 2026
Derivation of the dispersion equation using the transformation matrix derived in the other script
@author: lremilla
"""

from sympy import init_printing, symbols, Matrix, Function, sympify
from sympy.vector import CoordSys3D
from IPython.display import display

init_printing(use_latex=True)
#Create cylindrical coordinate system
coordinate_system = CoordSys3D('CS', transformation='cylindrical',variable_names=('rho','phi','z'))

#Define the symbols needed

print("Setting up symbols...")
E, H = symbols("E H", cls=Function) #E and H fields
m, eps, mu = symbols("m epsilon_n mu_n")#azimuthal order and material params
k_z = symbols("k_z")#longitudinal propagation constant
t, w, f = symbols("t omega f")#time and angular frequency and frequency
k_rho = symbols("k_rhon")#radial propagation constant
F_1, F_2, G_1, G_2 = symbols("F_1n F_2n G_1n G_2n")#modal amplitude coefficients
H_1m, H_2m = symbols("H^{(1)}_m H^{(2)}_m", cls=Function)#Hankel functions
M = Matrix(4, 4, lambda i, j: symbols(f'M_{i+1}_{j+1}'))#matrix of symbols to be solved for the transformation matrix
p = symbols("rho")#actual rho for subbing in at the end because exporting CS variables sucks
rho_MTS, rho_PEC = symbols("rho_MTS rho_PEC")

#Coordinates
rho = coordinate_system.rho
phi = coordinate_system.phi
z = coordinate_system.z

with open("transformationMatrixSymbolic.txt",'r') as file:
    M = sympify(file.read())