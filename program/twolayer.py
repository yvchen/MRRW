import sys,re,argparse
import numpy as np
from operator import itemgetter
from numpy import linalg as LA

def read_sim_matrix( infile ):
	weightMtx = []
	for line in file(infile).readlines():
		weightMtx.append(np.array(map(float, line.split())))
	return np.array(weightMtx)

def remove_diag( inMtx ):
	num_row, num_col = np.shape(inMtx)
	for i in range(0, num_row):
		inMtx[i][i] = 0
	return inMtx

def check_valid( inMtx ):
	# check whether there are negative weights
	if np.sum((inMtx<0).astype(np.int)) > 0:
		return 1
	return 0

def keep_top( inMtx, N ):
	row = np.shape(inMtx)[0]
	outMtx = np.zeros((row, row))
	for i in range(0, row):
		sortList = []
		for j in range(0, row):
			sortList.append((j, inMtx[i][j]))
		sorted(sortList, key=itemgetter(1), reverse=True)
		for k in range(0, N):
			j, s = sortList[k]
			outMtx[i][j] = s
	return outMtx

def check_size( Mtx1, Mtx2, Mtx12 ):
	row1, col1 = np.shape(Mtx1)
	row2, col2 = np.shape(Mtx2)
	row12, col12 = np.shape(Mtx12)
	if row1 != col1 or row2 != col2 or row1 != row12 or row2 != col12:
		return 1
	sys.stderr.write("#node in 1st layer: %d\n#node in 2nd layer: %d\n" %(row1, row2))
	return 0

def row_normalize( inMtx ):
	sum = np.sum(inMtx, 1)
	return (inMtx/sum[:, np.newaxis]).T

parser = argparse.ArgumentParser(description = "Running a two-layer MRRW algorithm.")
parser.add_argument('interweight1', metavar='layer1-edge-weight', help=': edge weight matrix for the 1st layer')
parser.add_argument('interweight2', metavar='layer2-edge-weight', help=': edge weight matrix for the 2nd layer')
parser.add_argument('betweenweight', metavar='layer1to2-edge-weight', help=': edge weight matrix between two layers')
parser.add_argument('--initscore1', metavar='layer1-init-score', help=': initial scores of nodes in the 1st layer (default = 1/N)')
parser.add_argument('--initscore2', metavar='layer2-init-score', help=': initial scores of nodes in the 2nd layer (default = 1/N)')
parser.add_argument('-w', metavar='alpha', type=float, help=': weight for propagation (default = 0.9)', default=0.9)
parser.add_argument('-n', metavar='N', type=int, help=': leave top N highest edges for each nodes (default = all)')

args = parser.parse_args()

E_11 = remove_diag(read_sim_matrix(args.interweight1))
E_22 = remove_diag(read_sim_matrix(args.interweight2))
E_12 = read_sim_matrix(args.betweenweight)

if check_size(E_11, E_22, E_12):
	sys.stderr.write("Error: The dimensions do not match.\n")
	exit()

if args.n != None:
	E_11 = keep_top(E_11, args.n)
	E_22 = keep_top(E_22, args.n)

if check_valid(E_11) or check_valid(E_22) or check_valid(E_12):
	sys.stderr.write("Error: The input weights have negative values.\n")
	exit()

num1 = np.shape(E_11)[0]
num2 = np.shape(E_22)[0]

L_11 = row_normalize(E_11)
L_22 = row_normalize(E_22)
L_12 = row_normalize(E_12.T)
L_21 = row_normalize(E_12)

# initial score setting
S1 = np.ones(num1)/num1
if args.initscore1 != None:
	S1 = read_sim_matrix(args.interweight1)
	if check_valid(S1):
		sys.stderr.write("Error: Some predefined scores are negative.\n")
		exit()
S1 = S1[:, np.newaxis]
		
S2 = np.ones(num2)/num2
if args.initscore1 != None:
	S2 = read_sim_matrix(args.interweight1)
	if check_valid(S2):
		sys.stderr.write("Error: Some predefined scores are negative.\n")
		exit()
S2 = S2[:, np.newaxis]

former = (1 - args.w) * S1 + args.w * (1 - args.w) * np.dot(L_11, np.dot(L_12, S2))
latter = args.w * args.w * np.dot(L_11, np.dot(L_12, np.dot(L_22, L_21)))
combine = former * np.ones(num1) + latter

w, v = LA.eig(combine)
score1 = abs(v[:, 0])/sum(abs(v[:, 0]))
score2 = (1 - args.w) * S2 + args.w * np.dot(L_22, np.dot(L_21, S1))

sys.stdout.write("Scores for layer1: ")
for i in range(0, num1):
	sys.stdout.write("%f " %score1[i])
sys.stdout.write("\n")
sys.stdout.write("Scores for layer2: ")
for i in range(0, num2):
	sys.stdout.write("%f " %score2[i])
sys.stdout.write("\n")
