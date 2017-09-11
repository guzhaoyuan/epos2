import matplotlib.pyplot as plt
from epos2.srv import *
import rospy
import pickle
import numpy as np
pickle_file = 'reward/single-09-09-04:14.pkl'
with open(pickle_file,'rb') as f:
	GLOBAL_RUNNING_R = pickle.load(f)
	GLOBAL_MEAN_R = pickle.load(f)


pickle_file = 'reward/double-09-09-04:42.pkl'
with open(pickle_file,'rb') as f:
	GLOBAL_RUNNING_R2 = pickle.load(f)
	GLOBAL_MEAN_R2 = pickle.load(f)

GLOBAL_MEAN_R2 = GLOBAL_MEAN_R2[0::2]


plt.plot(np.arange(len(GLOBAL_MEAN_R)), GLOBAL_MEAN_R)
plt.plot(np.arange(len(GLOBAL_MEAN_R2)), GLOBAL_MEAN_R2)
plt.xlabel('step')
plt.ylabel('Total moving reward')
plt.show()