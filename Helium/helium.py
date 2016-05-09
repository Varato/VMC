# Variational Mento Carlo for Helium atom
# by Xinchen 20160224
# Here we use natural units
# Experimentally result of ground state E0 = -2.904 
import numpy as np
from numpy import linalg as la
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import thread

# ----convention of parameters----
global spacial_dim               # Number of spacial dimention
global n_electrons               # Number of electrons
global n_walkers                 # Number of walker
global alpha                     # Parameter for variational
global step_size                 # The max step in a walk
global r0                        # Prime coordinates of 2 electrons
global E_sum, Esqrd_sum, nAccept # Quantities to accumulate
global E_ave, E_var              # Expectation and Variance of Local_energy
global MCSteps                   # Number of steps for Mento Carlo
global sample                    # A list to store sampled points
# --------------------------------
sample=[]
E_sum = Esqrd_sum = nAccept = 0
n_electrons = 2
spacial_dim=3
# --computation configuration---
MCSteps = 5000               #|
n_walkers = 150               #|
step_size = 1                 #|
# ------------------------------

def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'%s' % num)
	sys.stdout.flush()

def initialize():
	global n_walkers, n_electrons, spacial_dim
	global r0
	r0 = np.zeros([n_walkers, n_electrons, spacial_dim])
	for w in range(n_walkers):
		for ele in range(n_electrons):
			r0[w][ele] = np.random.uniform(-0.5, 0.5, spacial_dim)
			if (r0[w][ele] == np.zeros([1,3])).all(): r0[w][ele][0]=0.1
def zeroAccumulate():
	global E_sum, Esqrd_sum
	E_sum=0
	Esqrd_sum=0

def trial_wave_func(r):
	global alpha
	# r is a 2d array
	r1 = la.norm(r[0])
	r2 = la.norm(r[1])
	r12 = la.norm(r[0]-r[1])
	tmp = -2*r1-2*r2+0.5*r12/(1+alpha*r12)
	psi = np.exp(tmp)
	return psi

def local_energy(r):
	global alpha
	r1 = la.norm(r[0])
	r2 = la.norm(r[1])
	r12 = la.norm(r[0]-r[1])
	dotprod = np.dot((r[0]-r[1])/r12, r[0]/r1-r[1]/r2)
	denom = 1.0/(1+alpha*r12)
	denom2 = denom*denom
	denom3 = denom2*denom
	denom4 = denom2*denom2
	E_loc = -4 + alpha*denom + alpha*denom2 + alpha*denom3 - 0.25*denom4 + dotprod*denom2
	return E_loc

def Metropolis_step(walker):
	global r0, nAccept, E_sum, Esqrd_sum, alpha, sample
	trial_r = np.zeros([n_electrons, spacial_dim])
	for ele in range(n_electrons):
		while 1:
			trial_r[ele] = r0[walker][ele] + step_size*np.random.uniform(-1, 1, spacial_dim)
			if not (trial_r[ele] == np.zeros([1,3])).all(): break
	w0 = trial_wave_func(r0[walker])
	w1 = trial_wave_func(trial_r)
	p = (w1/w0)**2
	# Is this trial acceptable?
	if random.random() < p: 
		# update
		r0[walker]=trial_r
		nAccept += 1
		# store sampled points
		if alpha == result_alpha:
			sample.append(trial_r)
	E_local = local_energy(r0[walker])
	E_sum += E_local
	Esqrd_sum += E_local**2

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
	result_alpha = 0.33
	# a = np.linspace(0.05, 0.4, 6)
	a=[result_alpha]
	total_points = len(a)
	Ev = []
	var = []

	k = 0
	for aa in a:
		alpha = aa
		k+=1
		print "======================================="
		print "CURRENT Alpha: %f, %d of %d points" % (alpha, k, total_points)
		runMonte_Carlo()
		Ev.append(E_ave)
		var.append(E_var)
		if alpha == result_alpha:
			sample = np.array(sample)
			E0 = E_ave
		np.save('sample.npy', sample)
		print "Acceptance ratio = %.3f%%" % (100.0*nAccept/n_walkers/MCSteps)
	print "======================================="
	print "**********************************"
	print "Result E0 = %.3f" % E0
	print "Expermentally result E0 = -2.904"
	print "Error = %.2f%%" % (100.0*np.abs(-2.904-E0)/2.904)
	print "**********************************"

	plt.figure(figsize=(16,9))
	plt.subplot(121)
	plt.plot(a, Ev, 'r*-')
	plt.xlabel("Parameter")
	plt.ylabel("Energy")
	plt.subplot(122)
	plt.plot(a, var, 'b*-')
	plt.xlabel("Parameter")
	plt.ylabel("Variance")
	plt.savefig("helium.png")
	plt.show()




