'''
Simple script for running Covasim
'''

import sciris as sc
import covasim as cv

# Run options
do_plot = 1
verbose = 0
interv  = 1

# Configure the sim -- can also just use a normal dictionary
pars = sc.objdict(
    pop_size     = 10000,    # Population size
    pop_infected = 0,       # Number of initial infections
    n_imports     = 1,
    n_days       = 120,      # Number of days to simulate
    rand_seed    = 2,        # Random seed
    pop_type     = 'hybrid', # Population to use -- "hybrid" is random with household, school,and work structure
)

# Optionally add an intervention
pars.interventions = cv.test_prob(symp_prob = 1.0, 
                                  asymp_prob=1.0, 
                                  symp_quar_prob=None, 
                                  asymp_quar_prob=None, 
                                  subtarget=None, 
                                  ili_prev=None,
                                  test_sensitivity=1.0, 
                                  loss_prob=0.0, 
                                  test_delay=1, 
                                  start_day=0, 
                                  end_day=None)

# Make, run, and plot the sim
sim = cv.Sim(pars=pars)
sim.initialize()
sim.run(verbose=verbose)
if do_plot:
    sim.plot()
