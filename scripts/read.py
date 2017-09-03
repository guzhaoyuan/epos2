#!/usr/bin/env python
import pickle

# f = open('data/dataset-09-03-05:44.pkl','rb')
# num = 0
# for i in range(100):
# 	num += 1
# 	dic = pickle.load(f)
# 	print dic['a'], 'num',num

pickle_file = 'data/dataset-09-03-06:03.pkl'
with open(pickle_file,'rb') as f:
    unpickled = []
    while True:
        try:
            unpickled.append(pickle.load(f))
        except EOFError:
            break

print len(unpickled)
print type(unpickled[0][62]['a'])
# print unpickled[0]['a']

# pickle_file = 'data/dataset-09-03-06:03.pkl'
# with open(pickle_file,'wb') as f:
# 	for i in range(1000):
# 		pickle.dump(unpickled[0],f)