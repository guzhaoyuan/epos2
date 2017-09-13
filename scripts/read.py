#!/usr/bin/env python

'''
this fiile read the pickle file and plot the reward from the stored data
'''

# pickle_file = 'data/dataset-09-03-06:59.pkl' # this is trained 114 episodes, but found the time lag is severe for pickle write
# pickle_file = 'data/dataset-09-03-07:53.pkl' # trained for 2 hours, in local minimum, dont use
# pickle_file = 'data/dataset-09-03-10:13.pkl' # this is trained faster, but still not working, until I found the true reason, mod in cpp was wrong
# pickle_file = 'data/dataset-09-03-11:53.pkl' # this 300 episode as start training, not good enough
# pickle_file = 'data/dataset-09-05-07:15.pkl' # 300 episode, trained after solid structure, final episode stand on the pendulum, very exciting
# pickle_file = 'data/dataset-09-05-08:32.pkl' # former 150 episode is usable, but stuck in local minimum
import pickle

# f = open('data/dataset-09-03-05:44.pkl','rb')
# num = 0
# for i in range(100):
# 	num += 1
# 	dic = pickle.load(f)
# 	print dic['a'], 'num',num

pickle_file = 'data/dataset-09-05-08:32.pkl'
with open(pickle_file,'rb') as f:
    unpickled = []
    while True:
        try:
            unpickled.append(pickle.load(f))
        except EOFError:
            break

print len(unpickled)
# print type(unpickled[0][62]['a'])
print unpickled[0]['s']

# pickle_file = 'data/dataset-09-03-06:03.pkl'
# with open(pickle_file,'wb') as f:
# 	for i in range(1000):
# 		pickle.dump(unpickled[0],f)