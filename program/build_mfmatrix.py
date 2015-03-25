import re,sys,json,argparse
sys.path.append('program/lib')
import depvec
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def add_diag( inMat ):
	return inMat + np.identity(len(inMat))

def build_sim_matrix( itemList, embedding ):
	tmpList, vecDict, dim = depvec.load_vec(embedding) # read embeddings
	simMat = depvec.build_matrix(itemList, vecDict, dim)
	for i in range(0, len(simMat)):
		if itemList[i][0] not in vecDict:
			simMat[i] = np.zeros(len(simMat))
			simMat.T[i] = np.zeros(len(simMat))
	return depvec.filter_edge(simMat, 10)

def build_cxtsim_matrix( itemList, depfile, embedding ):
	tmpList, vecDict, dim = depvec.load_vec(embedding)
	freqDict, relDict = depvec.read_dep(depfile)
	simMat = depvec.build_fdvec_matrix(itemList, relDict, vecDict)
	return depvec.filter_edge(simMat, 10)

def build_deprel_matrix( itemList, depfile ):
	freqDict, relDict = depvec.read_dep(depfile)
	base_vec, relMat = depvec.build_dep_matrix(itemList, freqDict, relDict, 0, 0)
	return relMat
#	return depvec.filter_edge(relMat, 10)

parser = argparse.ArgumentParser(description = "Build a mtrix (default: row -> utt, col -> word/frm).")
parser.add_argument('trainfile', metavar='train-utt', help=': file including training utterances')
parser.add_argument('testfile', metavar='test-utt', help=': file including testing utterances')
parser.add_argument('trainfeafile', metavar='fea-file', help=': file including training SEMAFOR outputs')
parser.add_argument('outfile', metavar='out-matrix', help=': output built matrix')
parser.add_argument('oututtfile', metavar='test-utt-id', help=': the list of testing utterances\' ids')
parser.add_argument('outfeafile', metavar='fea-id', help=': the list of frame ids')
parser.add_argument('outfrmfile', metavar='frm', help=': the list of frames')
parser.add_argument('--feajson', help='feature file including slots and corresponding fillers')
parser.add_argument('--depfrm', help='file including dep parsing of frmaes')
parser.add_argument('--depword', help='file including dep parsing of words')
parser.add_argument('--dvecfrm', help='file including dep embeddings of frmaes')
parser.add_argument('--dvecword', help='file including dep embeddings of words')
parser.add_argument('--vecword', help='file including embeddings of words')
parser.add_argument('-t', help=': flag about transposing the matrix (default = 0)', default=0)
args = parser.parse_args()
transpose_flag = args.t
print args
#if len(sys.argv) == 1 or len(sys.argv) < 8:
#	sys.stderr.write("%s <train utt> <test utt> <feature file> <output matrix (.mtx)> <test utt list> <pred fea list> <out slot list> <transpose flag (default = 0)>\n" %sys.argv[0])
#	exit()


# save filename
#trainfile = sys.argv[1]
#testfile = sys.argv[2]
#trainfeafile = sys.argv[3]
#outfile = sys.argv[4]
#oututtfile = sys.argv[5]
#outfeafile = sys.argv[6]
#outfrmfile = sys.argv[7]

trainuttList = file(args.trainfile).read().strip().split('\n')
testuttList = file(args.testfile).read().strip().split('\n')
uttList = trainuttList + testuttList

slotList = []
for line in file(args.trainfeafile).readlines():
	frameList = json.loads(line)
	slotfea = {}
	for frm in frameList:
		slotfea[frm['F']] = 1
	slotList.append(slotfea)


# create n-gram part
vec = CountVectorizer(min_df=1, binary=True)
X1 = vec.fit_transform(uttList).todense()

wordList = []
for word in vec.get_feature_names():
	wordList.append((word, 0))


# create slot part
vec = DictVectorizer()
X2 = vec.fit_transform(slotList).todense()

frmList = []
f = open(args.outfrmfile, 'w')
for slot in vec.get_feature_names():
	frm = 'F_' + slot
	frmList.append((frm, 0))
	f.write("%s\n" %frm)
f.close()

# compute the number of samples
num_row1, num_col1 = np.shape(X1)
num_row2, num_col2 = np.shape(X2)
train_row = num_row2
test_row = num_row1 - num_row2
Z2 = np.zeros((test_row, num_col2))
print np.shape(Z2)
print np.shape(X2)

# output test utterance list
f = open(args.oututtfile, 'w')
for uid in range(train_row + 1, train_row + test_row + 1):
	f.write("%d\n" %uid)
f.close()

# output predct feature list
f = open(args.outfeafile, 'w')
for fid in range(num_col1 + 1, num_col1 + num_col2 + 1):
	f.write("%d\n" %fid)
f.close()

# compute the matrix for pair-wised frame similarities
if args.dvecfrm != None:
	S_dff = add_diag(build_sim_matrix(frmList, args.dvecfrm))
#	X2 = X2 * S_dff

#	wordList, vecDict, dim = depvec.load_vec(args.dvecfrm)
#	simMat = np.zeros((len(frmList), len(frmList)))
#	for i in range(0, len(frmList)):
#		for j in range(0, len(frmList)):
#			if frmList[i][0] in fdvecDict and frmList[j][0] in fdvecDict:
#				simMat[i][j] = depvec.nor_cos(fdvecDict[frmList[i][0]][1], fdvecDict[frmList[j][0]][1])
#	X2 = X2 * simMat

if args.dvecword != None:
	S_dww = build_sim_matrix(wordList, args.dvecword)
#	X1 = np.concatenate((X1[0:train_row] * S_dww, X1[train_row:]), axis=0)

# compute the matrix for pair-wised word similarities
if args.vecword != None:
	S_ww = build_sim_matrix(wordList, args.vecword)
#	X1 = np.concatenate((X1[0:train_row] * S_ww, X1[train_row:]), axis=0)


if args.depfrm != None:
	S_cff = build_cxtsim_matrix(frmList, args.depfrm, args.dvecfrm)
	R_ff = normalize(build_deprel_matrix(frmList, args.depfrm), axis=0, norm='l1')
#		for j in range(0, len(R_ff)):
#			print i, j, R_ff[i][j]
#	X2 = X2 * S_cff

if args.depword != None:
	S_cww = build_cxtsim_matrix(wordList, args.depword, args.dvecword)
	R_ww = normalize(build_deprel_matrix(wordList, args.depword), axis=0, norm='l1')

if '.Ft.' in args.outfile:
	M_ff = add_diag(S_dff)
	M_ww = np.identity(len(wordList))
elif '.Fts.' in args.outfile:
	M_ff = add_diag(S_dff + S_cff * R_ff)
	M_ww = np.identity(len(wordList))
elif '.Wt.' in args.outfile:
	M_ff = np.identity(len(frmList))
	M_ww = add_diag(S_dww)
elif '.Wts.' in args.outfile:
	M_ff = np.identity(len(frmList))
	M_ww = add_diag(S_dww + S_cww * R_ww)
elif '.FtWt.' in args.outfile:
	M_ff = add_diag(S_dff)
	M_ww = add_diag(S_dww)
elif '.FtsWt.' in args.outfile:
	M_ff = add_diag(S_dff + S_cff * R_ff)
	M_ww = add_diag(S_dww)
elif '.FtWts.' in args.outfile:
	M_ff = add_diag(S_dff)
	M_ww = add_diag(S_dww + S_cww * R_ww)
elif '.FsWs.' in args.outfile:
	M_ff = add_diag(S_cff * R_ff)
	M_ww = add_diag(S_cww * R_ww)
elif '.FtsWts.' in args.outfile:
	M_ff = add_diag(S_dff + S_cff * R_ff)
	M_ww = add_diag(S_dww + S_cww * R_ww)
else:
	M_ff = np.identity(len(frmList))
	M_ww = np.identity(len(wordList))

sys.stderr.write("#ngram = %d, #feature = %d, #train = %d, #test = %d\n" %(num_col1, num_col2, train_row, test_row))
X1 = np.concatenate((X1[0:train_row] * M_ww, X1[train_row:]), axis=0)
X2 = np.concatenate((X2 * M_ff, Z2), axis=0)
X3 = np.concatenate((X1, X2), axis=1)

if transpose_flag == 1:
	X3 = X3.T
mmwrite(args.outfile, csr_matrix(X3))

