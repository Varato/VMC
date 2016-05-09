import numpy as np
from numpy import linalg as la
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


global n_walkers  # Number of walker
global alpha      # Parameter for variational
global step_size  # The max step in a walk
global r0         # old coordinates of 2 electrons
global E_sum, Esqrd_sum, nAccept
global E_ave, E_var
global MCSteps
global sample
E_sum = Esqrd_sum = nAccept = 0
MCSteps = 5000
n_walkers = 150
step_size = 1
sample=[]

def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'%s' % num)
	sys.stdout.flush()

def initialize():
	global n_walkers
	global r0
	r0 = np.zeros([n_walkers, 3])
	for w in range(n_walkers):
		r0[w]=np.random.uniform(-0.5, 0.5, 3)
		if (r0[w] == np.zeros([1,3])).all():
			r0[w][0] = 0.1

def zeroAccumulate():
	global E_sum, Esqrd_sum
	E_sum=0
	Esqrd_sum=0

def trial_wave_func(r):
	global alpha
	# r is a array
	r = la.norm(r)
	psi = np.exp(-alpha*r)
	return psi

def local_energy(r):
	global alpha
	r = la.norm(r)
	E_loc = -(alpha**2-2*alpha/r)/2-1/r
	return E_loc

def Metropolis_step(walker):
	global r0, nAccept, E_sum, Esqrd_sum, alpha, sample
	while 1:
		trial_r = r0[walker]+step_size*np.random.uniform(-1, 1, 3)
		# avoid origin point being chosen
		if not (trial_r == np.zeros([1,3])).all(): break

	w0 = trial_wave_func(r0[walker])
	w1 = trial_wave_func(trial_r)
	p = (w1/w0)**2
	# Is this trial acceptable?
	if random.random() < p: 
		# update
		r0[walker]=trial_r
		nAccept += 1

		# record sampled result at alpha = 1
		if alpha == 1.0:
			sample.append(trial_r)
	E_local = local_energy(r0[walker])
	E_sum = E_sum+E_local
	Esqrd_sum = Esqrd_sum+E_local**2

def oneMento_Carlo_step():
	global n_walkers
	for w in range(n_walkers):
		Metropolis_step(w)

def runMonte_Carlo():
	global MCSteps, step_size, n_walkers
	global E_ave, E_var, nAccept
	# Firstly use 20% MCSteps to adjust step_size so that 
	# the acceptance ratio is about 50%
	thermSteps = int(0.2*MCSteps)
	adjust_interval = int(0.1*thermSteps)+1
	nAccept=0

	print "-----------------------------"
	print "Thermalizing...."
	for i in range(thermSteps):
		flushPrint("current step: ", i+1)
		oneMento_Carlo_step()
		if (i+1) % adjust_interval == 0:
			step_size *= nAccept/(0.5*n_walkers*adjust_interval)
			nAccept = 0
	print
	print "Adjusted step_size = %f" % step_size
	print "-----------------------------"

	#After adjust step_size, let's get E_ave and E_var
	zeroAccumulate()
	nAccept=0
	print "Start real computation:"
	for i in range(MCSteps):
		flushPrint("current step: ", i+1)
		oneMento_Carlo_step()
	print

	E_ave = E_sum/n_walkers/MCSteps
	E_var = Esqrd_sum/n_walkers/MCSteps - E_ave**2

if __name__=="__main__":
	initialize()
	a = np.arange(0.5, 1.5, 0.05)
	# a = [1.0]
	total_points = len(a)
	Ev = []
	var = []

	fig=plt.figure(figsize=(16,9))
	
	k = 0
	for aa in a:
		alpha = aa
		k+=1
		print "======================================="
		print "CURRENT Alpha: %f, %d of %d points" % (alpha, k, total_points)
		runMonte_Carlo()
		Ev.append(E_ave)
		var.append(E_var)
		print "Acceptance ratio = %.3f%% " % (1.0*nAccept/n_walkers/MCSteps)
	np.save('sample.npy', sample)

	plt.subplot(121)
	plt.plot(a, Ev, 'r*-')
	plt.xlabel("Parameter")
	plt.ylabel("$<E>$")
	plt.subplot(122)
	plt.plot(a, var, 'b*-')
	plt.xlabel("a")
	plt.ylabel("$\sigma$")
	plt.show()




