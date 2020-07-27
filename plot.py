import numpy as np
import matplotlib.pyplot as plt

stats = np.load("./stats/stats-new.npy", allow_pickle=True).item()
loss, theta = stats['loss'], stats['theta']

fig, axs = plt.subplots(nrows=9, ncols=1, figsize=(5, 10), sharex=True)
x = np.arange(len(loss))

offset = 25
axs[0].plot(x, loss, label='loss')
axs[0].set_title('open')
axs[0].legend()
axs[0].annotate("%.2f" % loss[-1], (x[-1]+offset, loss[-1]))

theta = np.transpose(np.array(theta))

labels = ['v', 'w', 'x', 't', 'o', 'p', 'g', 'i']
LOWER = np.array([0.2, 0.7,  3,  3, 0, 0, 0,   0])
UPPER = np.array([2.0, 2.2, 20, 60, 1, 1, 1, 0.5])
for _ in range(8):
	axs[_+1].plot(x, theta[_], label=labels[_])
	th = theta[_][-1]
	#th = LOWER[_] + (UPPER[_] - LOWER[_]) * (1 - np.cos(np.pi * th / 10)) / 2
	th = LOWER[_] + (UPPER[_] - LOWER[_]) * th / 10
	if _ == 2 or _ == 3:
		th = int(th)
		axs[_+1].annotate("%2d" % th, (x[-1]+offset, theta[_][-1]))
	else:
		axs[_+1].annotate("%.2f" % th, (x[-1]+offset, theta[_][-1]))
	axs[_+1].legend(loc="upper right")
plt.savefig("img/open.png")
plt.close()
