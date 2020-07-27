import os
import numpy as np
import time

DOCKER_NAME = {
	"default" : "jackal-evaluation-ahg-all",
	"bwi"     : "bwi-evaluation-gdc-joydeep",
	"all"     : "jackal-evaluation-ahg-all",
	"open"    : "jackal-evaluation-ahg-open",
	"curve"   : "jackal-evaluation-ahg-curve",
	"obstacle": "jackal-evaluation-ahg-obstacle",
	"gap"     : "jackal-evaluation-ahg-gap",
}

SLEEP_TIME = {
	"all"     : 80,
	"bwi"     : 50,
	"default" : 80,
	"open"    : 20, 
	"curve"   : 20,
	"obstacle": 18,
	"gap"     : 40,
}

ENV = 'obstacle'

PARAMS = {
	"bwi"     : np.array([0.65, 0.35031436,0.52281052,0.04229913,15.36932933]),
	#"bwi"     : np.array([0.5, 0.5, 3, 2, 15]),
	#"bwi"     : np.array([1, 0.5, 1, 2, 10]),
	"default" : np.array([0.5, 1.57, 6, 20, 0.1, 0.75, 1, 0.3]),
    "gap"     : np.array([0.25, 1.34, 8, 59, 0.43, 0.65, 0.98, 0.4]),   # gap best
    "open"    : np.array([1.59, 0.89, 18, 18, 0.4, 0.46, 0.27, 0.42]), # open best
    "curve"   : np.array([0.80, 0.73, 6, 42, 0.04, 0.98, 0.94, 0.19]),    # curve best
    "obstacle": np.array([0.71, 0.91, 16, 53, 0.55, 0.54, 0.91, 0.39]),   # tuned obstacle best
	"all"     : np.array([1.55, 0.98, 10, 3, 0.006, 0.87, 0.99, 0.46]),   # tuned all
}

#  1.5881357   0.88066972 18.46359684 19.0577594   0.40288669  0.46062402
#  0.27211434  0.42384207]
  

#DEFAULT = np.array([0.5, 1.57, 6, 20, 0.1, 0.75,   1,  0.3]) 
#BEST = np.array([0.25, 1.34, 8, 59, 0.43, 0.65,  0.98,  0.4]) # gap best
#BEST = np.array([1.50, 0.89, 16, 36, 0.65, 0.25,  0.27,  0.02]) # open best
#BEST = np.array([0.80, 0.73, 6, 42, 0.04, 0.98, 0.94, 0.19]) # curve best
#BEST = np.array([0.71, 0.91, 16, 53, 0.55, 0.54, 0.91, 0.39]) # in experiment obstacle best
#BEST = np.array([0.71, 1.79, 11, 48, 0.12, 0.57, 0.74, 0.26]) # tuned obstacle best

BESTALL = np.array([1.55, 0.98, 10, 3, 0.006, 0.87, 0.99, 0.46]) # tuned all

template = "nohup docker run --cpuset-cpus={} --rm {} /home/xuesu/jackal_ws/evaluation/evaluate.sh -v {} -w {} -s 2.0 -x {} -t {} -o {} -p {} -g {} -i {} -l 100 > ./results/default{}.txt 2>&1 &"
bwi = "nohup docker run --cpuset-cpus={} --rm {} /home/xuesu/bwi_ws/evaluation/evaluate.sh -v {} -w {} -i {} -e {} -c {} > ./results/default{}.txt 2>&1 &"

for ENV in ['bwi']:
	for i in range(10):
		if ENV == 'bwi':
			a,b,c,d,e = PARAMS[ENV]
			os.system(bwi.format(i,DOCKER_NAME[ENV], a,b,c,d,e, i))
		else:
			a,b,c,d,e,f,g,h = PARAMS[ENV]
			os.system(template.format(i,DOCKER_NAME[ENV], a,b,int(c),int(d),e,f,g,h, i))

	time.sleep(SLEEP_TIME[ENV])

	def calc_loss(sample_id):
		with open("./results/default%d.txt" % sample_id, 'r') as f:
			try:
				if ENV == 'all' or ENV == 'default':
					lines = f.readlines()[-8:]
					v1, w1 = [float(a) for a in lines[0].split()[:2]]
					v2, w2 = [float(a) for a in lines[2].split()[:2]]
					v3, w3 = [float(a) for a in lines[4].split()[:2]]
					v4, w4 = [float(a) for a in lines[6].split()[:2]]
					return v1+w1, v2+w2, v3+w3, v4+w4
				else:
					lines = f.readlines()[-5:]
					v, w, x, y, phi = [float(a) for a in lines]
					return v + w
			except:
				return 100

	x = []
	for i in range(10):
		x.append(calc_loss(i))
	x = np.array(x)
	print 'ENV: ', ENV, x.mean(0), x.std(0)
