from utils import *

def dist_test():
	np.random.seed(0)
	n = 1000
	k = 10
	x = np.random.rand(2*k)
	pop = Population()
	pop.populate(n, ws=[0.75, 0.25], mode='grf')
	c = Configuration(x, pop)
	c.disp()
	c.plot(pop)

def grf_test():
	alpha = -2.5
	n_samples = 1000
	n_grid = 100
	data = grf_distribution(alpha, n_samples, n_grid)
	plt.plot(data[:,0], data[:,1], 'k.', markersize=0.5)
	plt.show()

def f(x, *args):
	return Configuration(x, args[0]).cost

def g(x):
	x2 = x**2
	return x2[:-2:2] + x2[1:-2:2] - (x2[2::2] + x2[3::2])

def diff_ev():
	# parameters
	n = 1000
	k = 5
	maxiter = 1000

	# initialize auxiliary structures
	pop = Population()
	pop.populate(n)
	bounds = 2*k*[(0,1)]
	lb = -np.inf*np.ones(k - 1)
	ub = np.zeros(k - 1)
	cstr = NonlinearConstraint(g, lb, ub)

	# differential evolution
	res = differential_evolution(f, bounds, args=(pop,), disp=True, polish=False,
		maxiter=maxiter, updating='deferred', workers=-1, constraints=cstr)

	# printing
	print(res)
	c = Configuration(res.x, pop)
	c.disp()
	c.plot(pop)

if __name__ == '__main__':
	dist_test()
