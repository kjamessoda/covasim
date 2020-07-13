'''
These are additional interventions and helper functions that are useful for SimCampus objects
'''

import covasim.interventions as cvi
import covasim.utils as cvu
import numpy as np
import sciris as sc

import random


def check_schedule(currentDay,schedule,front = True):
    '''
    Convenience function. It returns False if schedule has length zero or if its first (front = True) or last (front = False) 
    element is greater than currentDay. It returns True otherwise.
    '''
    if front:
        i = 0
    else:
        i = -1

    if len(schedule):
        result = currentDay >= schedule[i]
    else:
        result = False
    return result

def weekly_testing(sim):
    '''This helper function will implement a test_num or test_prob intervention on a weekly basis'''
    if sim.t%7 == 0:
        indices = []
    else:
        indices = [True] * sim['pop_size']

    return indices


#These classes provide different schemes for generating pools in pooled sampling
class RandomTestingPools:
    '''
    This is the simplest scheme. It provides a requested number of pools, each of the same size.

    Args:
        nPools   (int): Number of pools to generate
        nSamples (int): Number of agents per pool
        lock (Boolean): Whether only one set of pools should be generated (True) or if a new set of pools should
                        be called every time the member function create is called (False; default).

    '''
    def __init__(self,nPools,nSamples,lock = False):
        #Store arugments
        self.nPools   = nPools
        self.nSamples = nSamples
        self.lock     = lock

        #Create slots for the pools
        self.pools    = None

    def create(self,sim):
        if not (self.pools and self.lock):
            self.pools = {}
            allPools   = np.random.choice(sim['pop_size'],(self.nPools,self.nSamples),replace = False)
            for i in range(self.nPools):
                self.pools["pool_" + str(i)] = allPools[i,:]
        return self.pools


class FloorTargetedPools:
    '''
    Generate a sampling scheme in which a (roughly) set proportion of agents associated with floor in a Dorm object are 
    included in a pool, where every agent in the same pool is from the same pool and floor, and one agent from every room
    is sampled before multiple agents from the same room.

    Args:
        sampleProportion (float): The (approximate) proportion of agents that should be sampled. It is only approximate
                                  because there is some rounding error involved in the exact number of tests.
        assureCoverage (Boolean): Whether every floor in a Dorm object should be represented with a pool of at least one individual 
                                  (False;default).
        skipQuarantine (Boolean): Whether individuals in quarantine/isolation should be skipped when sampling agents for a pool (False;default).
        skipDiagnosed  (Boolean): Whether individuals who have previously been diagnosed as COVID positive should be skipped when 
                                  sampling agents for a pool (False;default).
        lock (Boolean)          : Whether only one set of pools should be generated (True) or if a new set of pools should
                                  be called every time the member function create is called (False; default).
    '''
    def __init__(self,sampleProportion,assureCoverage = False,skipQuarantine = False,skipDiagnosed = False,lock = False):
        self.sampleProportion = sampleProportion
        self.lock = lock

        #Instructions on 
        self.assureCoverage = assureCoverage
        self.skipQuarantine = skipQuarantine
        self.skipDiagnosed  = skipDiagnosed

        #Create slots for the pools
        self.pools    = {}


    def create(self,sim):
        if not (len(self.pools) != 0 and self.lock):
            self.pools = {}
            dormCounter = 0
            for dormName,dorm in sim.dorms.items():
                currentTotalFloor = max(dorm['f']) + 1
                for i in range(currentTotalFloor):
                    samplesToTake = int(round(self.sampleProportion * (dorm['f'] == i).sum()))
                    if samplesToTake < 1:
                        if self.assureCoverage:
                            samplesToTake = 1
                        else:
                            continue
                    newPool       = np.array([-1] * samplesToTake)
                    uniqueRooms   = set(dorm['r'][dorm['f'] == i])
                    if samplesToTake < len(uniqueRooms):
                        sampledRooms = random.sample(uniqueRooms,samplesToTake)
                    else:
                        sampledRooms  = list(uniqueRooms)
                        sampledRooms += random.sample(uniqueRooms,samplesToTake - len(uniqueRooms))

                    poolCounter = 0
                    for room in sampledRooms:
                        resample = True
                        targetedAgents = cvu.true(dorm['r'] == room)
                        while resample:
                            #TODO: Find a way to make this work
                            if len(targetedAgents) == 0:
                                #uniqueRooms doubles as a collection of rooms that could be sampled. If we know that 
                                #no agent associated with a room does not fit the desired criteria, the room is removed.
                                uniqueRooms -= set([room])
                                #If there are no more rooms that fit the desired criterion, quit searching. 
                                if len(uniqueRooms) == 0:
                                    candidate = -1
                                    break
                                if len(uniqueRooms) > len(sampledRooms):
                                    possibleRooms = uniqueRooms - set(sampledRooms) 
                                    room = random.sample(possibleRooms,1)[0]
                                else:
                                    room = random.sample(uniqueRooms,1)[0]
                                targetedAgents = cvu.true(dorm['r'] == room)

                            candidate = np.random.choice(targetedAgents)
                            resample = False
                            if self.skipQuarantine:
                                resample = resample or sim.people.quarantined[candidate + sim.dorm_offsets[dormCounter]] or (sim.people.diagnosed[candidate + sim.dorm_offsets[dormCounter]] * ~sim.people.recovered[candidate + sim.dorm_offsets[dormCounter]])
                            if self.skipDiagnosed:
                                resample = resample or sim.people.diagnosed[candidate + sim.dorm_offsets[dormCounter]]
                            #Remove the candidate from targetedAgents so that if a resample is necessary, the 
                            #rejected agent will not be selected again.
                            targetedAgents = targetedAgents[targetedAgents != candidate]
                        newPool[poolCounter] = candidate
                        poolCounter += 1

                    self.pools[dormName + "_Floor" + str(i)] = newPool[newPool != -1] + sim.dorm_offsets[dormCounter]
                dormCounter += 1

        return self.pools




#These are Intervention classes meant to be used in conjunction with 
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

        interv = symptomQuarantine(symp_prob=0.5) # The probability that a symptomatic person will be identified, quarantined and 
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


class PooledTesting(cvi.Intervention):
    '''
    Test a pool of individuals for COVID. Upon a positive result, quarantine the individuals and implement individual
    tests on each member of the pool.

    Args:
        poolGenerator(dict or function): Either a dict where each value is an array listing the individuals 
            in one pool or a function that returns such a dict. If a function, there should be one argument, 
            the Sim object.
        schedule (int or []String or []int): If a list, the days in the simulation where a pooled test should be implemented.
            Dates can be specified as in-simulation dates (i.e., int) or as Strings (run sciris.readdate() for acceptable formats). 
            If an int, a pooled tests will be implemented serially this number of days apart.
        test_sensitivity (float): Probability of a true positive
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention
        end_day (int): When to end the intervention
        kwargs (dict): passed to Intervention()

    **Examples**::
    '''

    def __init__(self, poolGenerator,schedule=None,test_sensitivity=1.0,
                    test_delay=0, start_day=0, end_day=None, debug = False,**kwargs):
        super().__init__(**kwargs)
        self._store_args()
        self.poolGenerator     = poolGenerator
        self.pooledSchedule    = schedule
        self.test_sensitivity  = test_sensitivity
        self.test_delay        = test_delay
        self.start_day         = start_day
        self.end_day           = end_day
        self.debug             = debug

        #Additonal members that facilitate the Intervention's operations
        self.individualSchedule = []
        self.individualsToTest  = []

        return


    def initialize(self,sim):
        '''This is borrowed from interventions.test_prob'''
        self.start_day      = sim.day(self.start_day)
        if self.end_day is None:
            if sim['end_day'] is None:
                self.end_day = sim['n_days']
            else:
                self.end_day = sim['end_day']
        self.end_day        = sim.day(self.end_day)
        self.days           = [self.start_day, self.end_day]
        if isinstance(self.pooledSchedule,list):
            #The schedule has a stack-like structure for efficiency. The -1 indicates that there are no more pooled tests scheduled.
            if len(self.pooledSchedule) == 1:
                self.pooledSchedule = [sim.day(self.pooledSchedule)]
            else:
                self.pooledSchedule = sim.day(self.pooledSchedule).sort(reverse = True)
        elif isinstance(self.pooledSchedule,int):
            tempStorage = [self.start_day]
            print(self.end_day)
            while tempStorage[-1] + self.pooledSchedule <= self.end_day:
                tempStorage += [tempStorage[-1] + self.pooledSchedule]
            tempStorage.sort(reverse = True)
            self.pooledSchedule = tempStorage
        self.initialized    = True
        return

    def apply(self, sim):
        '''Some of this code is also borrowed from interventions.test_prob'''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        #Implement a pooled test, if one is scheduled
        if check_schedule(t,self.pooledSchedule,False):
            print("Implementing pooled testing.")
            if callable(self.poolGenerator):
                currentPools = self.poolGenerator(sim)
            else:
                currentPools = self.poolGenerator

            newSchedule, newPools    = sim.people.test_pooled(currentPools,self.test_sensitivity, self.test_delay)
            self.individualSchedule += newSchedule
            self.individualsToTest  += newPools
            self.pooledSchedule.pop()

            if self.debug:
                print(currentPools)

        #Implement individual tests, if one is scheduled        
        while check_schedule(t,self.individualSchedule):
            currentPool = self.individualsToTest.pop(0)
            #The Intervention places individuals into quarantine directly. If sim.step() places them in quarantine, the 
            #   schedule for removing individuals that test negative becomes inaccurate.
            sim.people.quarantined[currentPool] = True 

            sim.people.test(currentPool, self.test_sensitivity, 0.0, self.test_delay, True)
            self.individualSchedule.pop(0)

        return 

