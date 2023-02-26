#written by J.Roca
#2D Trusses solved by Finite Element Analysis

from ast import Eq
import numpy as np
import matplotlib.pyplot as plt

#Required_functions##############################################   
def dofFunction(nele, nlnode, nldof, connectivity):
    dof = np.zeros((nldof, nele), dtype = int)
    for iele in range(nele):
        for ilnode in range(nlnode):
            ignode = connectivity[ilnode, iele]
            dof[2 * ilnode, iele] = 2 * ignode
            dof[2 * ilnode + 1, iele] = 2 * ignode + 1
    ndof = np.max(np.max(dof, 1)) + 1
    return dof, ndof

def cosineTable(nele, xydata, connectivity):
    cosTable = np.zeros((nele, 3))
    for iele in range(nele):
        ignode = connectivity[:, iele]
        ab = xydata[ignode[1], :] - xydata[ignode[0], :]
        le = (ab[0] ** 2 + ab[1] ** 2) ** (1 / 2)
        cosTable[iele, 0] = le
        cosTable[iele, 1] = ab[0] / le#l
        cosTable[iele, 2] = ab[1] / le#m
    return cosTable

def FEM(nele, xydata, connectivity, E, A, bcNodes, Pload):
    nldof = 4#local_dof_number
    nlnode = 2#local_nodes_number
    nnodes = xydata.shape[0]
    Stress = np.zeros(nele)#[Pa]
    #Calling_functions###########################################
    dof, ndof = dofFunction(nele, nlnode, nldof, connectivity)
    K = np.zeros((ndof, ndof))
    P = np.zeros(ndof)#[N]
    for i in bcNodes[1]:
        P[i] = Pload
    cosTable = cosineTable(nele, xydata, connectivity)
    #Assemblying_global_matrix###################################
    for iele in range(nele):
        le = cosTable[iele, 0]
        l = cosTable[iele, 1]
        m = cosTable[iele, 2]
        Kelem = E[iele] * A[iele] / le * np.array([[l * l, l * m, -l * l, -l * m], \
                                                   [l * m, m * m, -l * m, -m * m], \
                                                    [-l * l, -l * m, l * l, l * m], \
                                                    [-l * m, -m * m, l * m, m * m]])
        for ildof in range(nldof):
            igdof = dof[ildof, iele]
            for jldof in range(nldof):
                jgdof = dof[jldof, iele]
                K[igdof, jgdof] = K[igdof, jgdof] + Kelem[ildof, jldof]
    F = P
    KR = np.copy(K)
    FR = np.copy(F)
    #Boundary_conditions#########################################
    for i in bcNodes[0]:
        K[i, :] = 0
        K[i, i] = 1
        F[i] = 0
    #Solving#####################################################
    Q = np.linalg.solve(K, F)
    dQ = np.zeros((nnodes, 2))
    for i in range(nnodes):
        dQ[i, :] = [Q[2 * i], Q[2 * i + 1]]
    #Getting_stress##############################################
    for iele in range(nele):
        le = cosTable[iele, 0]
        l = cosTable[iele, 1]
        m = cosTable[iele, 2]
        B = 1. / le * np.array([-l, -m, l, m])
        for ildof in range(nldof):
            igdof = dof[ildof, iele]
            Stress[iele] = Stress[iele] + E[iele] * B[ildof] * Q[igdof]
    R = np.dot(KR, Q) - FR
    return Q, dQ, Stress, R 


#Data############################################################
nele = 11
xydata = np.array([[0., 0.],
                       [3.6, 0.],
                       [2 * 3.6, 0.],
                       [3 * 3.6, 0.],
                       [2 * 3.6 + 1.8, 3.118],
                       [3.6 + 1.8, 3.118],
                       [1.8, 3.118]])
connectivity = np.array([[0., 1., 2., 3., 4, 5, 0, 1, 1, 2, 2],
                         [1., 2., 3., 4., 5, 6, 6, 6, 5, 5, 4]], dtype = int)

E = 200e9 * np.ones(11) #[Pa]
A = 3000e-6 * np.ones(11) #[m2]

boundaryNodes = np.array([0, 1, 7])
loadingNodes = np.array([1, 3, 5, 7])
Pload = -300e3
bcNodes = (boundaryNodes, loadingNodes)

scaleFactor = 50

#Calling_solver##################################################
[Q, dQ, Stress, R] = FEM(nele, xydata, connectivity, E, A, bcNodes, Pload)
            
print('Stress[MPa]:', Stress / 1e6)
print('max.stress[Pa]', np.max(np.abs(Stress)))
print('Reaction[N]:', R)

#Plotting########################################################
xydata2 = xydata + scaleFactor * dQ

fig = plt.figure(1)
ax = fig.add_subplot(111)
for iele in range(nele):
    i = connectivity[0][iele]
    f = connectivity[1][iele]
    if Stress[iele] <= 0:
        color = 'r'
    else:
        color = 'b'
    plt.plot([xydata[i][0], xydata[f][0]], [xydata[i][1], xydata[f][1]], 'k--')
    plt.plot([xydata2[i][0], xydata2[f][0]], [xydata2[i][1], xydata2[f][1]], color)

ax.set_aspect('equal')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.show() 
