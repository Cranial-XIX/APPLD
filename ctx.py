import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import csv
import os

def make_data():
	dirpath = '/home/xuesu/jackal_ws/context_classification/'
	X, Y = [], []
	label = 0
	for data in os.listdir(dirpath):
		if 'data' in data:
			continue
		tmp = []
		for f in os.listdir(dirpath + data):
			if '.bag' not in f:
				for ff in os.listdir(dirpath+data+'/'+f):
					with open(dirpath+data+'/'+f+'/'+ff) as csvfile:
						reader = csv.reader(csvfile)
						D = []
						for i, line in enumerate(reader):
							if i == 0:
								continue
							string = line[14]
							string = string[2:-2]
							string = string.split(', ')
							string = string[1:-1]
							#try:
							string = [float(x) for x in string]
							#except:
							#	import pdb; pdb.set_trace()
							D.append(string)
						D = np.array(D[1:])
						tmp.append(D)
		X.append(np.concatenate(tmp, 0))
		Y.append(np.array([label] * X[-1].shape[0]))
		#datas[data] = {'X':np.concatenate(datas[data], 0), 'Y':label}
		label += 1

	X = np.concatenate(X, 0)
	Y = np.concatenate(Y, 0)
	N = X.shape[0]
	idx = np.random.choice(N, N, False)
	X = X[idx]
	Y = Y[idx]
	N_train = int(N * 0.75)
	N_test = N - N_train

	TRAIN = {
		'Xtr': torch.from_numpy(X[:N_train]),
		'Ytr': torch.from_numpy(Y[:N_train]),
	}
	TEST = {
		'Xte': torch.from_numpy(X[N_train:]),
		'Yte': torch.from_numpy(Y[N_train:]),
	}

	torch.save(TRAIN, 'train.pt')
	torch.save(TEST, 'test.pt')

def test(X, Y, model):
	N = X.shape[0]
	acc = 0
	for i in range(0, N, 64):
		x = X[i:min(i+64, N)]
		y = Y[i:min(i+64, N)]
		x = x.float().clamp(0.1, 30)
		y_ = model(x)
		y_ = y_.argmax(1)
		acc += (y_ == y).sum()
	print("[TEST] accuracy: %10.4f" % (acc/float(N)))

def train():
	train_data = torch.load('train.pt')
	test_data = torch.load('test.pt')

	Xtr, Ytr = train_data['Xtr'], train_data['Ytr']
	Xte, Yte = test_data['Xte'], test_data['Yte']

	#model = nn.Linear(718, 3)
	model = nn.Sequential(
		nn.Linear(718, 10),
		nn.ReLU(),
		nn.Linear(10, 3))

	opt = torch.optim.Adam(model.parameters(), lr=1e-3)

	test(Xte, Yte, model)
	N = Xtr.shape[0]
	for episode in range(100):
		idx = np.random.choice(N, N, False)
		X, Y = Xtr[idx], Ytr[idx]
		tt_loss = 0
		for i in range(0, N, 64):
			x = X[i:min(i+64, N)]
			y = Y[i:min(i+64, N)]
			x = x.float().clamp(0.1, 30)
			y_ = model(x)
			opt.zero_grad()
			loss = F.cross_entropy(y_, y)
			loss.backward()
			opt.step()
			tt_loss += loss.item()
		print("[TRAIN] episode %d | loss %10.4f" % (episode, tt_loss / (N//64)))
		test(Xte, Yte, model)

if __name__ == "__main__":
	train()






