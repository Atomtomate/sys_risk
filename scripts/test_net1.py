import PyVal
import numpy as np
import pickle
import matplotlib.pyplot as plt

N = 100
nw = PyVal.BS_Network()

pList = np.linspace(0.001,1.0,40)
MList = []
rsList= []
deltaList = []
solventList = []
vOutList = []

for p in pList:
    nw.run(N, p, 0.8, 0, 2.0, 0.0, 10000)
    MList.append(nw.get_M().copy())
    rsList.append(nw.get_rs().copy())
    solventList.append(nw.get_solvent().copy())
    deltaList.append(nw.get_delta_jacobians().copy())

vOutList = [(1.0-np.array(MList[i])).dot(np.array(rsList[i])) for i in range(len(pList))]


with open("run2_p.p") as f:
  pickle.dump(pList, f)
with open("run2_M.p") as f:
  pickle.dump(pList, f)
with open("run2_rs.p") as f:
  pickle.dump(pList, f)
with open("run2_delta.p") as f:
  pickle.dump(pList, f)
with open("run2_solvent.p") as f:
  pickle.dump(pList, f)
with open("run2_vOutList.p") as f:
  pickle.dump(pList, f)
