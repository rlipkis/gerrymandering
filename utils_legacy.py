import numpy as np
from scipy.optimize import differential_evolution
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class Person:
	def __init__(self, location, affiliation):
		self.x = location
		self.a = affiliation

class Population:
	def __init__(self):
		self.register = list()
		self.num_people = 0

	def add(self, person):
		self.register.append(person)
		self.num_people += 1

	def vote(self):
		# preference distribution
		ballots = Counter([person.a for person in self.register])
		return ballots

	def populate(self, num_people):
		np.random.seed(0)
		data = np.random.rand(num_people, 3)
		for i in range(num_people):
			p = Person(data[i,0:2], np.round(data[i,2]))
			self.add(p)

class District(Population):
	def __init__(self, location):
		Population.__init__(self)
		self.x = location

	def vote(self):
		# winner takes all
		ballots = Counter([person.a for person in self.register])
		a_majority = ballots.most_common(1)[0][0]
		return a_majority

class Configuration:
	def __init__(self, x, population):
		# x is design variable (concatenated coordinates of district centers)
		self.num_people = population.num_people
		self.num_districts = int(len(x)/2)
		self.districts = list()

		# initialize districts
		for i in range(self.num_districts):
			location = x[2*i:2*i+2]
			self.districts.append(District(location))

		# populate districts
		for p in population.register:
			idx_closest = np.argmin([np.linalg.norm(p.x - d.x) for d in self.districts])
			self.districts[idx_closest].add(p)

		# elections
		self.general = population.vote()
		self.congress = Counter([d.vote() for d in self.districts])

		# calculate distribution discrepancy
		# min: 0, max: sqrt(2)
		disc = 0
		for a in self.general.keys():
			prob_gen = self.general[a]/self.num_people
			prob_con = self.congress[a]/self.num_districts
			disc += (prob_gen - prob_con)**2
		self.disc = np.sqrt(disc)
		
		# calculate district population discrepancy
		# min: 0, max: 2(n - 1)/n
		pop_avg = self.num_people/self.num_districts
		self.ineq = np.sum([np.abs(d.num_people - pop_avg) for d in self.districts])/self.num_people

		# total cost
		self.cost = self.disc + self.ineq

	def plot(self):
		centers = np.array([list(d.x) for d in self.districts])
		v = Voronoi(centers)
		voronoi_plot_2d(v, show_vertices=False, point_size=10)
		for d in self.districts:
			locs = np.array([list(p.x) for p in d.register])
			affs = [p.a for p in d.register]
			plt.scatter(locs[:,0], locs[:,1], c=affs, cmap='jet', marker='.')
		plt.xlim((0,1))
		plt.ylim((0,1))
		plt.show()




		









