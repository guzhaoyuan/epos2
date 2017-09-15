import matplotlib.pyplot as plt
import pickle
import numpy as np
# pickle_file = 'disturbance/single-09-09-04:14.pkl'
pickle_file = 'swingup/single-09-15-05-44:49.pkl'
with open(pickle_file,'rb') as f:
	GLOBAL_RUNNING_R = pickle.load(f)


pickle_file = 'swingup/double-09-15-05-38:10.pkl'
with open(pickle_file,'rb') as f:
	GLOBAL_RUNNING_R2 = pickle.load(f)

# GLOBAL_MEAN_R2 = GLOBAL_MEAN_R2[0::2]


plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
plt.plot(np.arange(len(GLOBAL_RUNNING_R2)), GLOBAL_RUNNING_R2)
plt.xlabel('step')
plt.ylabel('Total moving reward')
plt.legend(['single', 'double'])
plt.show()