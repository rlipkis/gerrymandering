from utils import *
np.random.seed(1)

def diff_ev():
	# parameters
	n = 1000
	k = 10
	maxiter = 1000 # or higher

	# initialize auxiliary structures
	pop = Population()
	pop.populate(n, ws=[0.7, 0.3], mode='grf')
	bounds = 2*k*[(0,1)]
	lb = -np.inf*np.ones(k - 1)
	ub = np.zeros(k - 1)
	cstr = NonlinearConstraint(g, lb, ub)

	# differential evolution (specify function f or f_avg)
	res = differential_evolution(f, bounds, args=(pop,), 
		seed=0,
		disp=True, 
		polish=False,
		maxiter=maxiter, 
		constraints=cstr,
		updating='deferred', 
		workers=-1)

	# printing and logging
	with open('xopt.txt', 'w') as log:
		log.write('{}'.format(res.x))
	print(res)
	c = Configuration(res.x, pop)
	c.disp()
	c.plot(pop)

	# analysis
	robustness(res.x, pop) # somewhat slow

if __name__ == '__main__':
	diff_ev()
