import cma
import numpy as np
import os.path
import time
import sys

n_samples = 12
dim = 5
n_iters = dim * 100

LOWER = np.array([0.42, 0.28, 0, 0,  0])
UPPER = np.array([ 1.0,  1.0, 6, 6, 20])

ENV = sys.argv[1]

DOCKER_NAME = {
	"bwi"     : "bwi-evaluation-gdc-joydeep",
}

SLEEP_TIME = {
	"bwi"     : 25,
}

def transform(th):
	return (th / 10) * (UPPER-LOWER) + LOWER

def execute(i, th):
	template = "nohup docker run --cpuset-cpus={} --rm {} /home/xuesu/bwi_ws/evaluation/evaluate.sh -v {} -w {} -i {} -e {} -c {} > ./results/sample{}.txt 2>&1 &"
	th = transform(th)
	os.system(template.format(i, DOCKER_NAME[ENV], th[0], th[1], th[2], th[3], th[4], i))

def calc_loss(sample_id):
	with open("./results/sample%d.txt" % sample_id, 'r') as f:
		try:
			lines = f.readlines()[-5:]
			v, w, x, y, phi = [float(a) for a in lines]
			return v + w
		except:
			return 1000

def evaluate(thetas):
	err = [0] * 12
	P = 12
	for j in range(P):
		execute(j, thetas[j])

	time.sleep(SLEEP_TIME[ENV])

	for j in range(P):
		err[j] = calc_loss(j)
	return err

def CMA():
	opt = cma.CMAOptions()
	opt['tolfun'] = 1e-11
	opt['popsize'] = n_samples
	opt['maxiter'] = n_iters
	opt['bounds'] = [0, 10]
	mu = np.ones((dim)) * 5
	es = cma.CMAEvolutionStrategy(mu, 2., opt)

	stats = {
		'loss': [],
		'theta': [],
		'mintheta': [],
	}

	best_solution = None
	best_loss = np.inf
	t0 = time.time()

	for i in range(n_iters):
		solutions = es.ask()
		loss = evaluate(solutions)
		loss = np.array(loss)
		idx = np.argmin(loss)
		es.tell(solutions, loss)
		curr_best = np.array(solutions).mean(0)
		curr_min = np.array(solutions)[idx]
		stats['theta'].append(curr_best)
		stats['loss'].append(loss.mean())
		stats['mintheta'].append(curr_min)
		print("[INFO] iter %2d | time %10.4f | avg loss %10.4f | min loss %10.4f" % (
			i,
			time.time() - t0,
			loss.mean(), loss.min()))
		V = transform(curr_best)
		print(V)
		M = transform(curr_min)
		print(M)
		if (i+1) % 5 == 0:
			with open("./stats/{}.npy".format(ENV), 'w') as f:
				np.save(f, stats)


if __name__ == "__main__":
	print("[INFO] training docker environment ", ENV)
	CMA()
