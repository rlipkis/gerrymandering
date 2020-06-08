### modules

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
import copy
import sys

### problem structure classes

class Person:
	def __init__(self, x, a):
		self.x = x
		self.a = a

class Population:
	def __init__(self):
		self.register = list()
		self.n = 0

	def add(self, p):
		self.register.append(p)
		self.n += 1

	def populate(self, n, ws, mode):	
		if mode == 'grf':
			# GRF parameters
			alpha = -3
			n_grid = 100
			# sampling
			ws /= np.sum(ws) # normalize weights
			for a, w in enumerate(ws):
				n_samples = int(w*n)
				data = grf_distribution(alpha, n_samples, n_grid)
				for row in data:
					self.add(Person(row, a))
		else:
			# uniform distribution
			data = np.random.rand(n, 3)
			for row in data:
				self.add(Person(row[0:2], int(np.round(row[2]))))

class District():
	def __init__(self, x):
		self.x = x
		self.n = 0
		self.c = Counter()

	def register(self, p):
		self.n += 1
		self.c[p.a] += 1

	def majority(self): 
		return self.c.most_common(1)[0][0]

class Configuration:
	def __init__(self, x, pop):
		# parameters
		self.n = pop.n
		self.k = int(len(x)/2)
		
		# initialize districts
		self.districts = list()
		for i in range(self.k):
			loc = x[2*i:2*i+2]
			self.districts.append(District(loc))

		# registration and voting
		self.general = Counter()
		for p in pop.register:
			idx = np.argmin([np.linalg.norm(p.x - d.x) for d in self.districts])
			self.districts[idx].register(p)
			self.general[p.a] += 1
		self.congress = Counter([d.majority() for d in self.districts])

		# calculate distribution discrepancy (0 to 1)
		disc = 0
		for a in self.general.keys():
			disc += (self.general[a]/self.n - self.congress[a]/self.k)**2
		self.disc = np.sqrt(disc/2)
			
		# calculate district population discrepancy (0 to (n-1)/n)
		ineq = 0
		for d in self.districts:
			ineq += np.abs(d.n - self.n/self.k)
		self.ineq = ineq/(2*self.n)

		# total cost
		self.cost = self.disc + self.ineq

	def plot(self, pop):
		centers = np.array([list(d.x) for d in self.districts])
		v = Voronoi(centers)
		voronoi_plot_2d(v, show_vertices=False, point_size=10)
		locs = np.array([list(p.x) for p in pop.register])
		affs = [p.a for p in pop.register]
		plt.scatter(locs[:,0], locs[:,1], c=affs, cmap='jet', marker='.')	
		plt.xlim((0,1))
		plt.ylim((0,1))
		plt.show()

	def disp(self):
		print('=================================')
		print('general: {}'.format(self.general))
		print('congress: {}'.format(self.congress))
		print('disc: {0:.6f}'.format(self.disc))
		print('ineq: {0:.6f}'.format(self.ineq))
		print('cost: {0:.6f}'.format(self.cost))
		print('=================================')

### population distributions

def power_spectrum(kx, ky, alpha):
		if kx != 0 or ky != 0:
			return (kx**2 + ky**2)**(alpha/4)
		return 0
		
def gaussian_random_field(alpha, n):
	noise = np.fft.fft2(np.random.normal(size=(n, n)))
	amplitude = np.zeros((n, n))
	k = lambda i, n: ((i + n/2) % n) - n/2 # wavenumber
	for i in range(n):
		for j in range(n):
			amplitude[i, j] = power_spectrum(k(i, n), k(j, n), alpha)
	grf = np.fft.ifft2(noise*amplitude)
	return np.real(grf)

def grf_distribution(alpha, n_samples, n_grid):
	# discretized GRF
	g = gaussian_random_field(alpha, n=n_grid)
	
	# normalization
	g -= np.min(g)
	g /= np.sum(g)

	# sampling
	idx = np.random.choice(g.size, size=n_samples, p=g.flatten())
	data = 1.0*np.array([list(np.unravel_index(i, g.shape)) for i in idx])
	data += np.random.rand(data.shape[0], data.shape[1])
	data /= n_grid
	return data

### solution analysis

def vary(pop):
	sig = 0.01
	pop_new = copy.deepcopy(pop)
	for p in pop_new.register:
		p.x += sig*np.random.randn(2)
	return pop_new

def robustness(x, pop):
	m = 100
	costs = [Configuration(x, vary(pop)).cost for _ in range(m)]
	plt.hist(costs)
	plt.show()
