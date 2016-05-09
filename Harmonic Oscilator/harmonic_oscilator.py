import numpy as np
from numpy import linalg as la
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


global n_walkers  # Number of walker
global alpha      # Parameter for variational
global step_size  # The max step in a walk
global x0         # old coordinates of 2 electrons
global xMin, xMax
global E_sum, Esqrd_sum, nAccept
global E_ave, E_var
global MCSteps
global sample
E_sum = Esqrd_sum = nAccept = 0
MCSteps = 5000
n_walkers = 150
step_size = 1.0
xMin = -10.0
xMax = 10.0

def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'%s' % num)
	sys.stdout.flush()

def initialize():
	global n_walkers
	global x0
	x0 = np.random.uniform(-0.5, 0.5, n_walkers)

def zeroAccumulate():
	global E_sum, Esqrd_sum
	E_sum=0
	Esqrd_sum=0

def trial_wave_func(x):
	global alpha
	psi = np.exp(-alpha*x**2)
	return psi

def local_energy(x):
	global alpha
	E_loc = alpha+x**2*(0.5-2*alpha**2)
	return E_loc

def Metropolis_step(walker):
	global x0, nAccept, E_sum, Esqrd_sum, alpha, sample
	trial_x = x0[walker]+step_size*np.random.uniform(-1, 1)
	if trial_x < xMin: trial_x = xMin
	if trial_x > xMax: trial_x = xMax
	w0 = trial_wave_func(x0[walker])
	w1 = trial_wave_func(trial_x)
	p = (w1/w0)**2
	# Is this trial acceptable?
	if random.random() < p: 
		# update
		x0[walker]=trial_x
		nAccept += 1
	E_local = local_energy(x0[walker])
	E_sum = E_sum+E_local
	Esqrd_sum = Esqrd_sum+E_local**2

def oneMento_Carlo_step():
	global n_walkers
	for i in range(n_walkers):
		Metropolis_step(i)

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

	E_ave = 1.0*E_sum/n_walkers/MCSteps
	E_var = 1.0*Esqrd_sum/n_walkers/MCSteps - E_ave**2

if __name__=="__main__":
	global alpha
	initialize()
	a = np.arange(0.1,1.4,0.05)
	total_points = len(a)
	Ev = []
	var = []

	fig=plt.figure(figsize=(16,9))
	
	k = 0
	for alpha in a:
		k+=1
		print "======================================="
		print "CURRENT Alpha: %f, %d of %d points" % (alpha, k, total_points)
		runMonte_Carlo()
		Ev.append(E_ave)
		var.append(E_var)
		print "Acceptance ratio: ", 1.0*nAccept/n_walkers/MCSteps

	plt.subplot(121)
	plt.plot(a, Ev, 'r*-')
	plt.xlabel("Parameter")
	plt.ylabel("Energy")
	plt.yticks([0.,.2,.4,.5,.6,.8,1.,1.2,1.4])
	plt.plot([0,1.4],[0.5,0.5],'k-.')
	# plt.axis([0, 1.4, 0, 1.4])
	plt.subplot(122)
	plt.plot(a, var, 'b*-')
	plt.xlabel("Parameter")
	plt.ylabel("Varience")
	plt.show()




