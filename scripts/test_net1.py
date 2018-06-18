import PyVal
import numpy as np
import pickle
import matplotlib.pyplot as plt



k_min = 0.1
n_points = 8
nList = np.array([25,50]) #,100,200

M = []
rs = []
delta = []
solvent = []
connectivity = []
rs_var = []
delta_var = []
solvent_var = []


for N in nList:
    pList = np.union1d(np.linspace(0.1/N,10.0/N,n_points/2), np.linspace(0.8/N,1.2/N,n_points/2))
    nw = PyVal.BS_Network()
    MList = []
    rsList= []
    deltaList = []
    solventList = []
    rsVarList= []
    deltaVarList = []
    solventVarList = []
    #vOutList = []
    for p in pList:
        nw.run(N, p, 0.8, 0, 2.0, 0.0, 2000)
        MList.append(nw.get_M().copy())
        rsList.append(nw.get_rs().copy())
        solventList.append(nw.get_solvent().copy())
        deltaList.append(nw.get_delta_jacobians().copy())
        rsVarList.append(nw.get_rs().copy())
        solventVarList.append(nw.get_solvent().copy())
        deltaVarList.append(nw.get_delta_jacobians().copy())
    M.append(np.array(MList))
    rs.append(np.array(rsList))
    delta.append(np.array(deltaList))
    solvent.append(np.array(solventList))
    connectivity.append(np.array(pList*N))

#vOutList = [(1.0-np.array(MList[i])).dot(np.array(rsList[i])) for i in range(len(pList))]

bak = {'M': M, 'rs': rs, 'delta': delta, 'solvent': solvent, 'rs var': rs, 'delta var': delta, 'solvent var': solvent}
np.save("bak.npy", bak)
