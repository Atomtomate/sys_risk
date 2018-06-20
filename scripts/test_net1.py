import PyVal
import numpy as np
import pickle
import matplotlib.pyplot as plt

val = 0.2


k_min = 0.1
n_points = 12
nList = np.array([50, 75, 100, 200]) #,100,200

M = []
rs = []
delta = []
solvent = []
connectivity = []
rs_var = []
delta_var = []
solvent_var = []


for N in nList:
    pList = np.union1d(np.linspace(0.01/N,1.8/N,n_points), np.linspace(0.7/N,1.3/N,n_points))
    #pList = np.union1d(pList, np.linspace(np.log(N+10)/(N+10) , np.log(N-10)/(N-10), n_points))
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
        nw.run(N, p, val, 2, 1.0, 0.0, 1000, 40)
        MList.append(nw.get_M().copy())
        rsList.append(nw.get_rs().copy())
        solventList.append(nw.get_solvent().copy())
        deltaList.append(nw.get_delta_jacobians().copy())
        rsVarList.append(nw.get_rs_var().copy())
        solventVarList.append(nw.get_solvent_var().copy())
        deltaVarList.append(nw.get_delta_jacobians_var().copy())
    M.append(np.array(MList))
    rs.append(np.array(rsList))
    delta.append(np.array(deltaList))
    solvent.append(np.array(solventList))
    connectivity.append(np.array(pList*N))

deltaSum = np.array([np.sum(el,axis=(1,2)) for el in delta])
#vOutList = [(1.0-np.array(MList[i])).dot(np.array(rsList[i])) for i in range(len(pList))]
#plt.plot(np.array(connectivity).T, np.array(deltaSum).T/nList);

bak = {'M': M, 'rs': rs, 'delta': delta, 'solvent': solvent, 'rs var': rs_var, 'delta var': delta_var, 'solvent var': solvent_var}
np.save("val0_2_2.npy", bak)
