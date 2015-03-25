import sys,re
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from operator import itemgetter
if len(sys.argv) == 1:
	sys.stderr.write("%s <score list> <fea dep file> <weight for propagation> <all | best> <0: use base score, 1: recompute base score (default = 0)>\n" %sys.argv[0])
	exit()

weight = float(sys.argv[3])
if sys.argv[4] == 'all':
	all_flag = 1
else:
	all_flag = 0
flag = 0
if len(sys.argv) == 6:
	flag = int(sys.argv[5])

# read slot + score
slotList = []
for slot in file(sys.argv[1]).read().split('\n'):
	if len(slot) > 0:
		seg = slot.split()
		slotList.append((seg[0], float(seg[1])))

P = np.zeros(shape=(len(slotList), len(slotList)))
base_vec = np.zeros(shape=(len(slotList), 1))

freqDict = {}
relDict = {}
for dep in file(sys.argv[2]).read().split('\n'):
	if len(dep) > 0:
		m = re.match(r'(\w+)\(([\w_]+)-\d+, ([\w_]+)-\d+\)', dep)
		edge = m.group(1)
		w1 = m.group(2)
		w2 = m.group(3)
		if w1 != w2:
			# edge
			if (w1, w2) not in relDict:
				relDict[(w1, w2)] = {'total': 0, 'best': ('null', 0)}
			if edge not in relDict[(w1, w2)]:
				relDict[(w1, w2)][edge] = 0
			relDict[(w1, w2)][edge] += 1
			relDict[(w1, w2)]['total'] += 1
			if relDict[(w1, w2)][edge] > relDict[(w1, w2)]['best'][1]:
				relDict[(w1, w2)]['best'] = (edge, relDict[(w1, w2)][edge])
			# node
			if w1 not in freqDict:
				freqDict[w1] = 0
			if w2 not in freqDict:
				freqDict[w2] = 0
			freqDict[w1] += 1
			freqDict[w2] += 1

# original base score
if flag == 0:
	for i in range(0, len(slotList)):
		base_vec[i] = np.exp(slotList[i][1])
# recompute base score
if flag == 1:
	for i in range(0, len(slotList)):
		if slotList[i][0] in freqDict:
			base_vec[i] = freqDict[slotList[i][0]]
		else:
			base_vec[i] = 0

for i in range(0, len(slotList)):
	w1 = slotList[i][0]
	for j in range(0, len(slotList)):
		w2 = slotList[j][0]
		if i != j:
			if all_flag == 1:
				if (w1, w2) in relDict:
					P[i][j] += relDict[(w1, w2)]['total']
				if (w2, w1) in relDict:
					P[i][j] += relDict[(w2, w1)]['total']
			else:
				if (w1, w2) in relDict:
					P[i][j] += relDict[(w1, w2)]['best'][1]
				if (w2, w1) in relDict:
					P[i][j] += relDict[(w2, w1)]['best'][1]
#for rel in relDict:
#	print rel, relDict[rel]
	
# normalization
base_vec = normalize(base_vec, axis=0, norm='l1')
P = normalize(P, axis=0, norm='l1')

r =  np.dot(base_vec, np.ones(shape=(1, len(slotList))))
P2 = (1 - weight) * r + weight * P
if weight > 0:
	eigval, eigvec = LA.eig(P2)
	best_eigval = abs(eigval[0].real)
	best_eigvec = abs(eigvec[:, 0].real)
	for i in range(0, len(best_eigvec)):
		if best_eigvec[i] < 0:
			sys.stderr.write('Warning: eigen vec is negative!\n')
			print best_eigval
			print best_eigvec
			exit()
else:
	best_eigvec = base_vec[:, 0]

newList = []
for i in range(0, len(slotList)):
	newList.append((slotList[i][0], best_eigvec[i]))
newList.sort(key=itemgetter(1), reverse=True)
for slot, score in newList:
	print slot, np.log(score)
#print slotList
#print newList
