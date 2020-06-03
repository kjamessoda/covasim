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

    def __init__(self, pars=None, datafile=None, datacols=None, label=None, simfile=None, popfile=None, load_pop=False, save_pop=False, debug= False, **kwargs):
        super().__init__(kwargs)
        self['pop_type'] = 'campus' #This is just bookkeeping right now
        self.age_dist = {'dist':'uniform', 'par1':18, 'par2':22} #This new parameter provides a function for the age distribution of People objects
        self.debug = debug #This data member communicates whether the simulation is being used for software testing

        if self.debug:
            self.dorms = cvb.FlexDict({'a':cvbc.Dorm(2,[2,2,2],[2,2,2]),
                            'b':cvbc.Dorm(3,[3,3],[2,2])}) #For now, I am hard coding a Dorm object into the class until I can do something more specific
        else:
            #For now, I am hard coding a Dorm object into the class until I can do something more specific
            self.dorms = cvb.FlexDict({'DanaEnglish':cvbc.Dorm(2,[10,15,15],[2,2,2]),
                            'MountainView':cvbc.Dorm(2,[4,4,4,4,4],[8,14,14,5,5]),
                            'Commons':cvbc.Dorm(4,[13,13,13],[3,3,3])}) #I feel particularly uncertain about this structure; there are 12 fewer people here than there should be

        self.dorm_offsets = np.array([0] * (len(self.dorms) + 1)) #This array records the first agent ID for each Dorm object in self.dorms
        self['pop_size'] = 0 #Update the population size to match the number of People in self.dorms
        counter = 0
        for i in range(len(self.dorms)):
            new = len(self.dorms[i])
            self['pop_size'] += new
            self.dorm_offsets[i] = counter 
            counter += new

        self.dorm_offsets[-1] = self['pop_size'] #The population size is added as a convenience for later functions
        self.update_pars(pars, **kwargs)   # We have to update the parameters again in case any of the above overwrote a user provided value

        #TODO: Change these from placeholders to true default values
        self['beta_layer']  = {'r': 1 ,'b': 1,'f': 1,'c': 1} #Set defaults for the layer-specific parameters and establish what layers are present 
        self['contacts']    = {'r': -1,'b': 3,'f': 5,'c': 10}
        self['iso_factor']  = {'r': 0 ,'b': 0,'f': 0,'c': 0}
        self['quar_factor'] = {'r': 1 ,'b': 1,'f': 1,'c': 1}
        self['dynam_layer'] = {'r': False ,'b': True,'f': True,'c': True}



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
