'''
These are additional interventions and helper functions that are useful for SimCampus objects
'''

import covasim.interventions as cvi
import covasim.populationCampus as cvpc
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


class generalWeeklyTesting:
    '''This basic class generalizes the function weekly_testing so that it can run a once weekly intervention starting on a day other than the first'''
    def __init__(self,firstDay = 0):
        #the first day (in simulation time) that the function should be run
        self.firstDay   = firstDay

    def implement(self,sim):
        if (sim.t - self.firstDay)%7 == 0:
            indices = []
        else:
            indices = [True] * sim['pop_size']

        return indices


#These classes provide different schemes for generating pools in pooled sampling and other targeted sampling
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


class RandomNonResidentSample:
    def __init__(self,samplingRate,subgroup = 'none',lock = False):
        self.samplingRate = samplingRate
        self.lock         = lock
        #This data member can be used to specify a subgroup of non-resident students. Your options are:
        #   none = (default) Sample on non-resident students
        #   ug   = Sample non-residential undergraduate students
        #   grad = Sample non-residential grad students
        subgroup = subgroup.lower()
        if subgroup != 'none' and subgroup != 'ug' and subgroup != 'grad':
            raise ValueError("The subgroup argument is not a recognized value. Possible values are \'none\',\'ug\', and \'grad\'")
        self.subgroup = subgroup 
        #This slot will allow the same individuals to be sampled every time a sample is requested (if lock = True)
        self.sample       = np.array([])

    def create(self,sim):
        if not (self.lock and len(self.sample) > 0):
            if self.subgroup == 'none':
                nonResPop  = sim['pop_size'] - sim.dorm_offsets[-1]
                self.sample   = np.random.choice(np.arange(sim.dorm_offsets[-1],sim['pop_size']),int(round(nonResPop * self.samplingRate)),replace = False)
            if self.subgroup == 'ug':
                nonResPop  = sim.nonResidentEndIndex - sim.dorm_offsets[-1]
                self.sample   = np.random.choice(np.arange(sim.dorm_offsets[-1],sim.nonResidentEndIndex),int(round(nonResPop * self.samplingRate)),replace = False)
            else:
                nonResPop  = sim['pop_size'] - sim.nonResidentEndIndex
                self.sample   = np.random.choice(np.arange(sim.nonResidentEndIndex,sim['pop_size']),int(round(nonResPop * self.samplingRate)),replace = False)
        return self.sample


class SampleAll:
    def __init__(self,subgroup = 'none'):
        #This data member can be used to specify a subgroup of non-resident students. Your options are:
        #   none   = (default) Sample all students
        #   ug     = Sample all residential and non-residential undergraduate students
        #   resid  = Sample all residential students 
        #   nonres = Sample all non-residential undergraduate students
        #   grad   = Sample all non-residential grad students
        subgroup = subgroup.lower()
        if subgroup != 'none' and subgroup != 'ug' and subgroup != 'resid' and subgroup != 'nonres' and subgroup != 'grad':
            raise ValueError("The subgroup argument is not a recognized value. Possible values are \'none\',\'ug\', and \'grad\'")
        self.subgroup = subgroup 
        #This slot will allow the same individuals to be sampled every time a sample is requested (if lock = True)
        self.sample       = np.array([])

    def create(self,sim):
        if len(self.sample) == 0:
            if self.subgroup == 'grad':
                self.sample = np.arange(sim.nonResidentEndIndex,sim['pop_size'])
            elif self.subgroup == 'ug':
                self.sample = np.arange(sim.nonResidentEndIndex)
            elif self.subgroup == 'resid':
                self.sample = np.arange(sim.dorm_offsets[-1])
            elif self.subgroup == 'nonres':
                self.sample = np.arange(sim.dorm_offsets[-1],sim.nonResidentEndIndex)
            else:
                self.sample = np.arange(sim['pop_size'])
        return self.sample


class FloorTargetedPools:
    '''
    Generate a sampling scheme in which a (roughly) set proportion of agents associated with a floor in a Dorm object are 
    included in a pool, where every agent in the same pool is from the same pool and floor, and one agent from every room
    is sampled before multiple agents from the same room (this latter criteria has not yet been tested).

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
        collapse (Boolean)      : Whether the pooled structure should be collapsed down to a single numpy.array (False; default). This allows
                                  this class to be used without pooled testing.
    '''
    def __init__(self,sampleProportion,assureCoverage = False,skipQuarantine = False,skipDiagnosed = False,lock = False,collapse = False):
        self.sampleProportion = sampleProportion
        self.lock = lock

        #Instructions on 
        self.assureCoverage = assureCoverage
        self.skipQuarantine = skipQuarantine
        self.skipDiagnosed  = skipDiagnosed
        self.collapse       = collapse

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
                    if "suite" in dorm.dormType:
                        uniqueRooms   = set(dorm['b'][dorm['f'] == i])
                    else:
                        uniqueRooms   = set(dorm['r'][dorm['f'] == i])

                    if samplesToTake < len(uniqueRooms):
                        sampledRooms = random.sample(uniqueRooms,samplesToTake)
                    else:
                        j = 1
                        while len(uniqueRooms) < samplesToTake - len(uniqueRooms) * j:
                            j += 1
                        sampledRooms  = list(uniqueRooms) * j
                        sampledRooms += random.sample(uniqueRooms,samplesToTake - len(uniqueRooms)*j)

                    poolCounter = 0
                    for room in sampledRooms:
                        resample = True
                        if "suite" in dorm.dormType:
                            targetedAgents = np.array(list(set(cvu.true(dorm['b'] == room)) - set(newPool)))
                        else:
                            targetedAgents = np.array(list(set(cvu.true(dorm['r'] == room)) - set(newPool)))
                        breakOuter = False
                        while resample:
                            while len(targetedAgents) == 0:
                                #uniqueRooms doubles as a collection of rooms that could be sampled. If we know that 
                                #no agent associated with a room fits the desired criteria, the room is removed.
                                uniqueRooms -= set([room])
                                #If there are no more rooms that fit the desired criterion, quit searching. 
                                if len(uniqueRooms) == 0:
                                    breakOuter = True
                                    candidate = -1
                                    break

                                if len(uniqueRooms) > len(sampledRooms):
                                    possibleRooms = uniqueRooms - set(sampledRooms) 
                                    room = random.sample(possibleRooms,1)[0]
                                else:
                                    room = random.sample(uniqueRooms,1)[0]

                                if "suite" in dorm.dormType:
                                    targetedAgents = np.array(list(set(cvu.true(dorm['b'] == room)) - set(newPool)))
                                else:
                                    targetedAgents = np.array(list(set(cvu.true(dorm['r'] == room)) - set(newPool)))

                            if breakOuter:
                                break

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
        if self.collapse:
            self.pools = np.concatenate(list(self.pools.values()))

        return self.pools




#These are Intervention classes meant to be used in conjunction with 
class SuperShedderEvent(cvi.Intervention):
    '''
    This intervention adds contacts to the community layer beyond what was initially sampled. The intent is to simulate super shedder event.

    Args:
    eventSize      (int) or (dict): If an int, the number of individuals that should be incorporated into each contact event. If a dict, each 
                                    key-value pair should correspond to one argument in utils.sample; in other words, the dict should specify
                                    the distribution from which to pull sizes. 
    eventFrequency (int) or (list): If an int, the number of events on each day between start_day and end_day (see below). If a list, every 
                                    element should be the rate parameter for a Poisson distribution. The number events in one day will be pulled
                                    from the distribution. There should be seven elements, one for each day of the week, starting with the day the
                                    simulation begins. For example, if the simulation begins on a Monday,the first element provides the expected 
                                    number of events on a Monday. 
    start_day                (int): When to start the intervention
    end_day                  (int): When to end the intervention
    kwargs                  (dict): passed to Intervention()

    **Examples**::

    '''
    def __init__(self, eventSize, eventFrequency,start_day=0, end_day=None, **kwargs):
        super().__init__(**kwargs)
        self._store_args()
        self.eventSize      = eventSize
        self.eventFrequency = eventFrequency
        self.start_day      = start_day
        self.end_day        = end_day
        return


    def initialize(self,sim):
        '''This is borrowed from interventions.test_prob'''
        #Make sure that the eventFrequency member meets requirements
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.initialized = True
        return

    def apply(self, sim):
        '''Some of this code is also borrowed from interventions.test_prob'''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        #Get the number of events that will occur today
        if isinstance(self.eventFrequency,int):
            numberOfEvents = self.eventFrequency
        else:
            currentRate    = self.eventFrequency[t%len(self.eventFrequency)]
            numberOfEvents = cvu.n_poisson(currentRate, 1)[0]

        #Draw the number of individuals to incorporate into the supershedder event
        for i in range(numberOfEvents):
            if isinstance(self.eventSize,int):
                currentSize = self.eventSize
            elif isinstance(self.eventSize,dict):
                currentSize = cvu.sample(**self.eventSize)

            #This is necessary if the user requests a Poisson distribution for the size of the event
            if isinstance(currentSize,np.ndarray):
                currentSize = currentSize[0]

            #Select the individuals to include in the event
            agentsInEvent = cvu.choose_r(sim['pop_size'],currentSize)
            print(agentsInEvent)

            #Add the sampled individuals into the network
            newContacts = cvpc.create_mutual_contacts(sim,agentsInEvent,layer = 'c')
            sim.people.add_contacts(newContacts)

        return 



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
        symp_notMitigate = sim.people.symptomatic * ~sim.people.diagnosed
        testRecorded     = cvu.false(np.isnan(sim.people.date_tested))
        pendingTest      = testRecorded[sim.t - sim.people.date_tested[testRecorded] < self.test_delay]
        symp_notMitigate[pendingTest] = False
        if self.ili_prev is None:
            covidLikeInds = cvu.true(symp_notMitigate)
        else:
            rel_t = t - self.start_day
            ili_indices      = cvu.n_binomial(self.ili_prev[rel_t],sim['pop_size'])
            covidLikeInds    = cvu.true(np.logical_or(ili_indices,symp_notMitigate))

        reportedInds  = cvu.binomial_filter(self.symp_prob,covidLikeInds)


        #Quarantine and test the selected indices
        sim.people.quarantine(reportedInds)
        sim.people.test(reportedInds, self.test_sensitivity, 0.0, self.test_delay, True)

        sim.results['new_tests'][t] += int(len(reportedInds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

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

        nTests = 0

        #Implement a pooled test, if one is scheduled
        if check_schedule(t,self.pooledSchedule,False):
            print("Implementing pooled testing.")
            if callable(self.poolGenerator):
                currentPools = self.poolGenerator(sim)
            else:
                currentPools = self.poolGenerator

            newSchedule, newPools    = sim.people.test_pooled(currentPools,self.test_sensitivity, self.test_delay)
            print(len(currentPools))
            nTests += len(currentPools)
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
            nTests += len(currentPool)
            self.individualSchedule.pop(0)

        #This line taken from covasim.interventions.test_prob
        sim.results['new_tests'][t] += int(nTests*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec
        return 


class TestScheduler(cvi.Intervention):
    '''
    This is a relatively simple Intervention class. It works comparably to the PooledTesting class in that it can handle detailed 
    schedules for when to implement tests and facilitates detailed targeted testing. However, it focuses on individual testing rather
    than pooled testing. 


    Args:
        sampleGenerator(numpy.array or function): Either a numpy.array listing the individuals that should be tested according to the
                                                  schedule or a function that returns such an array. If a function, there should be one 
                                                  argument, the Sim object.
        schedule (int or []String or []int)     : If a list, the days in the simulation where a pooled test should be implemented.
                                                  Dates can be specified as in-simulation dates (i.e., int) or as Strings (run 
                                                  sciris.readdate() for acceptable formats). If an int, a pooled tests will be 
                                                  implemented serially this number of days apart.
        start_date (int or String)              : The first day that testing should be implemented, either via an in-simulation date 
                                                  (i.e., int) or as a String (run sciris.readdate() for acceptable formats)  (0; default). 
                                                  This argument is only used if schedule is a single int; otherwise, the first test is 
                                                  run on the first day specified in schedule.
        end_date (int or String)                : The last day that testing *could* be implemented, either via an in-simulation date 
                                                  (i.e., int) or as a String (run sciris.readdate() for acceptable formats) (last simulated 
                                                  day; default). This argument is only used if schedule is a single int; otherwise, the 
                                                  last test is run on the last day specified in schedule.
        test_sensitivity (float)                : Probability of a true positive
        test_delay (int)                        : How long testing takes
        loss_prob  (float)                      : Probability that an individual with a positive test will be lossed to follow up (i.e.,
                                                  will not have a record of their diagnosis and will not enter isolation)
        end_quarantine (Boolean)                : Whether a negative test result should end an individual's quarantine (False; default). 
                                                  This is only possible using a SimCampus object.
        kwargs (dict)                           : passed to Intervention()

    **Examples**::
    '''

    def __init__(self, sampleGenerator,schedule,start_date = 0,end_date = None,test_sensitivity=1.0,
                    test_delay=0, loss_prob = 0., end_quarantine = False,**kwargs):
        super().__init__(**kwargs)
        self._store_args()
        self.sampleGenerator   = sampleGenerator
        self.schedule          = schedule
        self.start_date        = start_date
        self.end_date          = end_date
        self.test_sensitivity  = test_sensitivity
        self.test_delay        = test_delay
        self.loss_prob         = loss_prob
        self.end_quarantine    = end_quarantine

        return

    def initialize(self,sim):
        if isinstance(self.schedule,list):
            #The schedule has a stack-like structure for efficiency.
            if len(self.schedule) == 1:
                self.schedule = [sim.day(self.schedule)]

            else:
                self.schedule = sim.day(self.schedule)
                self.schedule.sort(reverse = True)
        elif isinstance(self.schedule,int):
            tempStorage = [sim.day(self.start_date)]
            if not self.end_date:
                self.end_date = sim['n_days']
            else:
                self.end_date = sim.day(self.end_date)
            while tempStorage[-1] + self.schedule <= self.end_date:
                tempStorage += [tempStorage[-1] + self.schedule]
            tempStorage.sort(reverse = True)
            self.schedule = tempStorage
        self.days           = self.schedule
        self.initialized    = True
        return


    def apply(self, sim):
        '''Some of this code is also borrowed from interventions.test_prob'''
        t = sim.t

        #Implement tests, if they are scheduled
        if check_schedule(t,self.schedule,False):
            if callable(self.sampleGenerator):
                currentSample = self.sampleGenerator(sim)
            else:
                currentSample = self.sampleGenerator

            #The decision to end an individual's quarantine is decided via an independent control statement so that this
            #   Intervention class will work with the original Covasim
            if self.end_quarantine:
                sim.people.test(currentSample,self.test_sensitivity,self.loss_prob,self.test_delay,True)
            else:
                sim.people.test(currentSample,self.test_sensitivity,self.loss_prob,self.test_delay)
            self.schedule.pop()

            #This line taken from covasim.interventions.test_prob
            sim.results['new_tests'][t] += int(len(currentSample)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return 
