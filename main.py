from utils import *
np.random.seed(0)

def dist_test():
	n = 1000
	k = 10
	x = np.random.rand(2*k)
	pop = Population()
	pop.populate(n, ws=[0.7, 0.3], mode='grf')
	c = Configuration(x, pop)
	c.disp()
	c.plot(pop)

def grf_test():
	alpha = -3
	n_samples = 10000
	n_grid = 100
	data = grf_distribution(alpha, n_samples, n_grid)
	plt.plot(data[:,0], data[:,1], 'k.', markersize=0.5)
	plt.show()

def f_avg(x, *args):
	# averaged objective function
	n_avg = 10
	pop = args[0]
	costs = [f(x, vary(pop)) for _ in range(n_avg)]
	return np.mean(costs)

def f(x, *args):
	# objective function
	return Configuration(x, args[0]).cost

def g(x):
	# inequality constraint function
	p = 2 # norm
	x2 = np.abs(x)**p
	return x2[:-2:2] + x2[1:-2:2] - (x2[2::2] + x2[3::2])

def diff_ev():
	# parameters
	n = 1000
	k = 10
	maxiter = 500 # was 10000

	# initialize auxiliary structures
	pop = Population()
	pop.populate(n, ws=[0.7, 0.3], mode='grf')
	bounds = 2*k*[(0,1)]
	lb = -np.inf*np.ones(k - 1)
	ub = np.zeros(k - 1)
	cstr = NonlinearConstraint(g, lb, ub)

	# differential evolution
	res = differential_evolution(f_avg, bounds, args=(pop,), disp=True, polish=False,
		maxiter=maxiter, constraints=cstr) #, updating='deferred', workers=-1)

	# printing and logging
	with open('xopt.txt', 'w') as log:
		log.write('{}'.format(res.x))
	print(res)
	c = Configuration(res.x, pop)
	c.disp()
	c.plot(pop)

	# analysis
	robustness(res.x, pop)

if __name__ == '__main__':
	diff_ev()
