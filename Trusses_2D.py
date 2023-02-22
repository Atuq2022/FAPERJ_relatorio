#Written by J.Roca
#2D Trusses solved by Finite Element Analysis

import numpy as np
import matplotlib.pyplot as plt
import os

os.system('clear')
   
def dof(nele, nlnode, nldof, connectivity):
    out = np.zeros((nldof, nele), dtype = int)
    for iele in range(nele):
        for ilnode in range(nlnode):
            ignode = connectivity[ilnode, iele]
            out[2 * ilnode, iele] = 2 * ignode
            out[2 * ilnode + 1, iele] = 2 * ignode + 1
    ndof = np.max(np.max(out, 1)) + 1
    return out, ndof

def cosineTable(nele, xydata, connectivity):
    out = np.zeros((nele, 3))
    for iele in range(nele):
        ignode = connectivity[:, iele]
        ab = xydata[ignode[1], :] - xydata[ignode[0], :]
        le = (ab[0] ** 2 + ab[1] ** 2) ** (1 / 2)
        out[iele, 0] = le
        out[iele, 1] = ab[0] / le#l
        out[iele, 2] = ab[1] / le#m
    return out

#Data
nele = 11#number_elements
nnodes = 7#number_nodes
nldof = 4#local_dof_number
nlnode = 2#local_nodes_number
E = 200e9 * np.ones(nele)#[Pa]
A = 3000e-6 * np.ones(nele)#[m2]
Stress = np.zeros(nele)#[Pa]
xydata = np.array([[0., 0.],
        [3.6, 0.],
       [2 * 3.6, 0.],
       [3 * 3.6, 0.],
       [2 * 3.6 + 1.8, 3.118],
       [3.6 + 1.8, 3.118],
       [1.8, 3.118]])
connectivity = np.array([[0., 1., 2., 3., 4, 5, 0, 1, 1, 2, 2],
                        [1., 2., 3., 4., 5, 6, 6, 6, 5, 5, 4]], dtype = int)

#Calling_functions
dof, ndof = dof(nele, nlnode, nldof, connectivity)
K = np.zeros((ndof, ndof))
P = np.zeros(ndof)#[N]
P[1] = -280e3
P[3] = -210e3
P[5] = -280e3
P[7] = -360e3

cosTable=cosineTable(nele, xydata, connectivity)
#Assemblying_global_matrix
for iele in range(nele):
    le = cosTable[iele, 0]
    l = cosTable[iele, 1]
    m = cosTable[iele, 2]
    Kelem = E[iele] * A[iele] / le * np.array([[l * l, l * m, -l * l, -l * m], [l * m, m * m, -l * m, -m * m], [-l * l, -l * m, l * l, l * m], [-l * m, -m * m, l * m, m * m]])
    for ildof in range(nldof):
        igdof = dof[ildof, iele]
        for jldof in range(nldof):
            jgdof = dof[jldof, iele]
            K[igdof, jgdof] = K[igdof, jgdof] + Kelem[ildof, jldof]
F = P
KR = np.copy(K)
FR = np.copy(F)
#Boundary_conditions
K[0, :] = 0
K[0, 0] = 1
F[0] = 0
K[1, :] = 0
K[1, 1] = 1
F[1] = 0
K[7, :] = 0
K[7, 7] = 1
F[7] = 0
#Solving
Q = np.linalg.solve(K, F)
print('Displacement[m]:', Q)

dQ = np.zeros((nnodes, 2))
for i in range(nnodes):
    dQ[i, :] = [Q[2 * i], Q[2 * i + 1]]

#Getting_stress
for iele in range(nele):
    le = cosTable[iele, 0]
    l = cosTable[iele, 1]
    m = cosTable[iele, 2]
    B = 1. / le * np.array([-l, -m, l, m])
    for ildof in range(nldof):
        igdof = dof[ildof, iele]
        Stress[iele] = Stress[iele] + E[iele] * B[ildof] * Q[igdof]
print('Stress[Pa]:', Stress)
print('max.stress[Pa]', np.max(np.abs(Stress)))
#Getting_reactions
R = np.dot(KR, Q) - FR
print('Reaction[N]:', R)

xydata2 = xydata + 50 * dQ

for iele in range(nele):
    i = connectivity[0][iele]
    f = connectivity[1][iele]
    plt.plot([xydata[i][0], xydata[f][0]], [xydata[i][1], xydata[f][1]], 'g')
    plt.plot([xydata2[i][0], xydata2[f][0]], [xydata2[i][1], xydata2[f][1]], 'b')
plt.grid()
plt.show()