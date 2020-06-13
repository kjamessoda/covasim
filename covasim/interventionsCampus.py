'''
These are additional interventions and helper functions that are useful for SimCampus objects
'''

import covasim.interventions as cvi
import covasim.utils as cvu
import numpy as np
import sciris as sc

def weekly_testing(sim):
    '''This helper function will implement a test_num or test_prob intervention on a weekly basis'''
    if sim.t%7 == 0:
        indices = []
    else:
        indices = [True] * sim['pop_size']

    return indices

class symptomQuarantine(cvi.Intervention):
    '''
    Identify individuals who have COVID-like symptoms, place them in quarantine, and test them for COVID. This 
    Intervention subclass is modeled on the test_prob subclass.

    Args:
        symp_prob (float): Probability of identifying a symptomatic person on any particular day
        subtarget (dict): subtarget intervention to people with particular indices (see test_num() for details)
        test_sensitivity (float): Probability of a true positive
        ili_prev (float or array): Prevalence of influenza-like-illness symptoms in the population
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention
        end_day (int): When to end the intervention
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = (symp_prob=0.5) # The probability that a symptomatic person will be identified, quarantined and 
                                      tested is 0.5.

    '''
    def __init__(self, symp_prob, subtarget=None, ili_prev=None,test_sensitivity=1.0,
                    test_delay=0, start_day=0, end_day=None, **kwargs):
        super().__init__(**kwargs)
        self._store_args()
        self.symp_prob = symp_prob
        self.subtarget = subtarget
        self.ili_prev  = ili_prev
        self.test_sensitivity = test_sensitivity
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
        return


    def initialize(self,sim):
        '''This is borrowed from interventions.test_prob'''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.ili_prev  = cvi.process_daily_data(self.ili_prev, sim, self.start_day)
        self.initialized = True
        return

    def apply(self, sim):
        '''Some of this code is also borrowed from interventions.test_prob'''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        #Find all agents that display COVID-like symptoms
        if self.ili_prev is None:
            covidLikeInds = cvu.true(sim.people.symptomatic)
        else:
            rel_t = t - self.start_day
            ili_indices   = cvu.n_binomial(self.ili_prev[rel_t],sim['pop_size'])
            covidLikeInds = cvu.true(np.logical_or(ili_indices,sim.people.symptomatic))

        reportedInds  = cvu.binomial_filter(self.symp_prob,covidLikeInds)

        #Quarantine and test the selected indices
        sim.people.quarantine(reportedInds)
        sim.people.test(reportedInds, self.test_sensitivity, 0.0, self.test_delay, True)

        return 

