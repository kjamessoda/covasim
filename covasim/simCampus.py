'''
Defines the SimCampus class, which extends the Sim class to make it more amendable to simulating dormitories
'''

import covasim.sim as cvs
import covasim.base as cvb
import covasim.baseCampus as cvbc
import covasim.utils as cvu
import covasim.defaults as cvd
import covasim.populationCampus as cvpc
import covasim.misc as cvm
import numpy as np
import sciris as sc

class SimCampus(cvs.Sim):

    def __init__(self, pars=None, datafile=None, datacols=None, label=None, simfile=None, popfile=None, load_pop=False, save_pop=False, 
                    dorms = None, 
                    nonResident = 0, nonResContacts = None, n_importsNonRes = None,
                    gradStudents = 0, gradContactScale = 1., gradTransmissionScale = 1.,
                    age_dist = {'dist':'uniform', 'par1':18, 'par2':22},initialRecovered = [],debug= False, **kwargs):
        super().__init__(pars, datafile, datacols, label, simfile, popfile, load_pop, save_pop,**kwargs)
        #super().__init__(**kwargs)
        self['pop_type'] = 'campus' #This is just bookkeeping right now
        self.age_dist = age_dist #This new parameter provides a function for the age distribution of People objects
        self.debug = debug #This data member communicates whether the simulation is being used for software testing

        if not isinstance(initialRecovered,list):
            raise ValueError("The argument initialRecovered needs to be a list of dictionaries.")
        else:
            for instruction in initialRecovered:
                if not isinstance(instruction,dict):
                    raise ValueError("The argument initialRecovered needs to be a list of dictionaries.")
                currentKeys = instruction.keys()
                if not ('nAgents' in currentKeys and 'subpop' in currentKeys):
                    raise ValueError("One of the dict objects in initialRecovered does not contain the necessary keys, \'nAgent\' and \'subpop\'.") 
        self.initialRecovered = initialRecovered


        if dorms:
            self.dorms = dorms
        else:
            self.dorms = cvb.FlexDict({'a':cvbc.Dorm(2,[2,2,2],[2,2,2]),
                            'b':cvbc.Dorm(3,[3,3],[2,2])}) 
        self.validate_dorms()

        self.dorm_offsets = np.array([0] * (len(self.dorms) + 1)) #This array records the first agent ID for each Dorm object in self.dorms
        self['pop_size'] = 0 #Update the population size to match the number of People in self.dorms
        counter = 0
        for i in range(len(self.dorms)):
            new = len(self.dorms[i])
            self['pop_size'] += new
            self.dorm_offsets[i] = counter 
            counter += new

        self.dorm_offsets[-1] = self['pop_size'] #The population size is added as a convenience for later functions
        self['pop_size'] += nonResident    # Add non-residential students to the population
        self.nonResidentEndIndex = self['pop_size'] #This will record the first id that does not refer to a non-residential undergraduate; it also provides the number of undergraduates in the simulation
        self['pop_size'] += gradStudents    # Add graduate students to the population
        self.update_pars(pars, **kwargs)   # We have to update the parameters again in case any of the above overwrote a user provided value


        self['beta_layer']  = {'r':  1,'b': 1,'f': 1,'c': 1} #Set values for SimCampus-specific features
        self['contacts']    = {'r': -1,'b': 3,'f': 3,'c': 3}
        self['iso_factor']  = {'r':  0,'b': 0,'f': 0,'c': 0}
        self['quar_factor'] = {'r':  0,'b': 0,'f': 0,'c': 0}
        self['dynam_layer'] = {'r': False ,'b': True,'f': True,'c': True}
        #This state allows the user to provide a different expected value for the number of imported cases for non-residential students 
        #   (this value) than for residential students (which is provided in the original n_imports parameter). If the value is None,
        #   then the expected value for the number of imported cases is set via n_imports, and the imported cases are assigned without
        #   regard to residential status.
        self.n_importsNonRes = n_importsNonRes 

        #Check if the user provided values for any of the layer-centric parameters or to alter prognoses. 
        for key,value in kwargs.items():
            if key == 'beta_layer':
                self['beta_layer']  = value
            if key == 'contacts':
                self['contacts']  = value
            if key == 'iso_factor':
                self['iso_factor']  = value
            if key == 'quar_factor':
                self['quar_factor']  = value
            if key == 'dynam_layer':
                self['dynam_layer']  = value

        #This parameter sets the average number of contacts that a non-residential (undergraduate) student has, if provided. If not provide, it defaults to the residential rate
        if nonResContacts:
            self.nonResContacts = nonResContacts
        else:
            self.nonResContacts = self['contacts']['c']

        #Grad students act like non-residential students, but you can scale down (or up) the number of contacts they have on average and probability 
        #   that they will acquire/spread COVID using these scaling parameters
        self.gradContactScale = gradContactScale 
        self.gradTransmissionScale = gradTransmissionScale


        if debug:
            self.watcher = open("watcher.csv",'w')
            self.watcher.write("Label," + "Data" + '\n')
            self.nonGradDiff = 0
            self.nonResContactsCounter = 0
        else:
            self.watcher = None



    def step(self):
        '''
        This is a modification of the superclass function. Mostly it calls the super class function, but
        it logs an additional stock that is not present in Sim objects.
        '''
        super().step()
        self.results["n_quarantineDorm"][self.t - 1] = self.people.quarantined.sum() + np.logical_and(self.people.diagnosed,~self.people.recovered).sum()

        return


    def initialize(self, reset=False, **kwargs):
        super().initialize(reset, **kwargs)
        for instruction in self.initialRecovered:
            self.set_random_recovered(instruction['nAgents'],instruction['subpop'])


    def init_people(self, save_pop=False, load_pop=False, popfile=None, verbose=None, **kwargs):
        '''
        This is a modification of the superclass function.

        Args:
            save_pop (bool): if true, save the population dictionary to popfile
            load_pop (bool): if true, load the population dictionary from popfile
            popfile (str): filename to load/save the population
            verbose (int): detail to print
            kwargs (dict): passed to cv.make_people()
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if verbose:
            print(f'Initializing sim with {self["pop_size"]:0n} people for {self["n_days"]} days')
        if load_pop and self.popdict is None:
            self.load_population(popfile=popfile)

        # Actually make the people
        self.people = cvpc.make_students(self, save_pop=save_pop, popfile=popfile, verbose=verbose, **kwargs)
        self.people.initialize() # Fully initialize the people

        # Create the seed infections
        inds = cvu.choose(self['pop_size'], self['pop_infected'])
        self.people.infect(inds=inds, layer='seed_infection')
        return


    def set_random_recovered(self,nAgents,subpop = 'none'):
        '''
        This function will change some number of agents' states to recovered. The agents that are switched to
        recovered are randomly selected, but you can specify that the agents belong to a certain subpopulation.

        Args:
            nAgents = int; number of agents to switch to the recovered state.
            subpop  = str; what subpopulation of students should be sampled. The following options are available:
                none     = (default) any agent can be sampled 
                res      = only sample residential students
                nonres_ug = only sample nonresidential undergraduate students
                grad     = only sample graduate students
        '''
        subpop = subpop.lower()
        if subpop == "none":
            pool = np.arange(self['pop_size'])
        elif subpop == "res":
            pool = np.arange(self.dorm_offsets[-1])
        elif subpop == "nonres_ug":
            pool = np.arange(self.dorm_offsets[-1],self.nonResidentEndIndex)
        elif subpop == "grad":
            pool = np.arange(self.nonResidentEndIndex,self['pop_size'])
        else:
            raise ValueError("The value provided for subpop is not recognized. Possible values are \'none\', \'res\', \'nonres_ug\', and \'grad\'")

        #Select which agents will be sampled
        pool = pool[~self.people.recovered[pool]]
        sample = np.random.choice(pool,nAgents,replace = False)        

        #Make the necessary alterations
        self.people.susceptible[sample] = False
        self.people.recovered[sample] = True

        return


    def init_results(self):
        '''
        This is a modification of the superclass function. Mostly it calls the super class function, but
        it creates a new stock channel.        
        '''
        super().init_results()

        #Create a new stock channel to record the number of students in quaratine *or* isolation; the color is shared 
        #   with quarantined.
        self.results["n_quarantineDorm"] = cvb.Result(name="n_quarantineDorm", npts=self.npts, scale='dynamic', color='#5f1914')

        return


    def update_prognoses(self,addedInfo):
        '''
        This function allows the user to elegantly alter sim.prognosis. It also validates that the provided update will operate properly.

        Args:
            addedInfo = dict; the keys in the dict should be a subset of the keys in the default prognoses (see parameters.py). The values
                        should be the desired np.array for that key. If 'age_cutoff' is present, all other keys must also be present; otherwise,
                        keys can be missing so long as the value has the correct length for the default age_cutoff value.  
        '''
        prognoses = self.pars['prognoses']
        if 'age_cutoffs' in addedInfo.keys():
             prognoses = addedInfo
        else:
            #Convert the health outcome probabilities back to joint probabilities for later consistency
            prognoses['severe_probs'] *= prognoses['symp_probs']   
            prognoses['crit_probs']   *= prognoses['severe_probs'] 
            prognoses['death_probs']  *= prognoses['crit_probs']   

            #Update the prognoses
            for key, val in addedInfo.items():
                prognoses[key] = np.array(val)

        #Convert the health outcome probabilities back to conditional probabilities
        prognoses['death_probs']  /= prognoses['crit_probs']   # Conditional probability of dying, given critical symptoms
        prognoses['crit_probs']   /= prognoses['severe_probs'] # Conditional probability of symptoms becoming critical, given severe
        prognoses['severe_probs'] /= prognoses['symp_probs']   # Conditional probability of symptoms becoming severe, given symptomatic

        #Validate the new prognoses
        for key, val in prognoses.items():
            if len(val) != len(prognoses['age_cutoffs']):
                raise ValueError("The provided value for " + key + " has an improper length.")

            if key in ['symp_probs','severe_probs','crit_probs','death_probs']:
                if (val < 0.).any() or (val > 1.).any():
                    raise ValueError("The provided joint probability for " + key + "does not lead to a valid conditional probability.")

        self.pars['prognoses'] = prognoses

        return


    def validate_pars(self, validate_layers=True):
        '''
        This is a slight modification of the superclass function to account for alterations in this subclass.

        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle types
        for key in ['pop_size', 'pop_infected', 'pop_size']:
            try:
                self[key] = int(self[key])
            except Exception as E:
                errormsg = f'Could not convert {key}={self[key]} of {type(self[key])} to integer'
                raise ValueError(errormsg) from E

        # Handle start day
        start_day = self['start_day'] # Shorten
        if start_day in [None, 0]: # Use default start day
            start_day = '2020-03-01'
        self['start_day'] = cvm.date(start_day)

        # Handle end day and n_days
        end_day = self['end_day']
        n_days = self['n_days']
        if end_day:
            self['end_day'] = cvm.date(end_day)
            #As I see it, we need to add one more day b/c the simulation will include both the start_day and end_day, but sciris.daydiff does not
            #   (cvm.daydiff is just a wrapper for sciris.daydiff)
            n_days = cvm.daydiff(self['start_day'], self['end_day'])
            if n_days <= 0:
                errormsg = f"Number of days must be >0, but you supplied start={str(self['start_day'])} and end={str(self['end_day'])}, which gives n_days={n_days}"
                raise ValueError(errormsg)
            else:
                self['n_days'] = int(n_days)
        else:
            if n_days:
                self['n_days'] = int(n_days)
                self['end_day'] = self.date(n_days) # Convert from the number of days to the end day
            else:
                errormsg = f'You must supply one of n_days and end_day, not "{n_days}" and "{end_day}"'
                raise ValueError(errormsg)

        # Handle population data
        popdata_choices = ['campus']
        choice = self['pop_type']
        if choice and choice not in popdata_choices:
            choicestr = ', '.join(popdata_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle interventions and analyzers
        self['interventions'] = sc.promotetolist(self['interventions'], keepnone=False)
        for i,interv in enumerate(self['interventions']):
            if isinstance(interv, dict): # It's a dictionary representation of an intervention
                self['interventions'][i] = cvi.InterventionDict(**interv)
        self['analyzers'] = sc.promotetolist(self['analyzers'], keepnone=False)

        # Optionally handle layer parameters
        if validate_layers:
            self.validate_layer_pars()

        return

    def validate_dorms(self):
        '''This function will validate that self.dorms has the correct types'''
        if not isinstance(self.dorms,cvb.FlexDict):
            raise TypeError("SimCampus.dorms must be class covasim.base.FlexDict")

        for element in self.dorms.values():
            if not isinstance(element,cvbc.Dorm):
                raise TypeError("The elements of SimCampus.dorms must be class covasim.baseCampus.Dorm")

        return

    def finalize(self, verbose=None, restore_pars=True):
        ''' This function largely calls the corresponding function in the parent class, but it allows additional debugging steps '''
        super().finalize(verbose,restore_pars)
        if self.debug:
            self.watcher.write("TotalNonResContacts," + str(self.nonResContactsCounter) + '\n')
            self.watcher.write("DiffContacts_NonResGrad," + str(self.nonGradDiff) + '\n')
            self.watcher.close()
