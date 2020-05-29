'''
Defines the SimCampus class, which extends the Sim class to make it more amendable to simulating dormitories
'''

import covasim.sim as cvs
import covasim.base as cvb
import covasim.baseCampus as cvbc
import covasim.defaults as cvd
import numpy as np

class SimCampus(cvs.Sim):

    def __init__(self, pars=None, datafile=None, datacols=None, label=None, simfile=None, popfile=None, load_pop=False, save_pop=False, debug= True, **kwargs):
        super().__init__()
        #self['pop_type'] = 'campus' #This is just bookkeeping right now
        self.age_dist = {'dist':'uniform', 'par1':18, 'par2':22} #This new parameter provides a function for the age distribution of People objects
        self.debug = debug #This data member communicates whether the simulation is being used for software testing

        if self.debug:
            self.dorms = cvb.FlexDict({'a':cvbc.Dorm(2,[2,2,2],[2,2,2]),
                            'b':cvbc.Dorm(3,[3,3],[2,2])}) #For now, I am hard coding a Dorm object into the class until I can do something more specific
        else:
            self.dorms = cvb.FlexDict({'DanaEnglish':cvbc.Dorm(2,[10,15,15],[2,2,2]),
                            'MountainView':cvbc.Dorm(2,[4,4,4,4,4],[8,14,14,5,5])}) #For now, I am hard coding a Dorm object into the class until I can do something more specific

        #TODO: Change these from placeholders to true default values
        self['beta_layer']  = {'r': 1 ,'b': 1,'f': 1,'c': 1} #Set defaults for the layer-specific parameters and establish what layers are present 
        self['contacts']    = {'r': -1,'b': 3,'f': 5,'c': 20}
        self['iso_factor']  = {'r': 1 ,'b': 1,'f': 1,'c': 1}
        self['quar_factor'] = {'r': 1 ,'b': 1,'f': 1,'c': 1}

        self.dorm_offsets = np.array([0] * (len(self.dorms) + 1)) #This array records the first agent ID for each Dorm object in self.dorms
        self['pop_size'] = 0 #Update the population size to match the number of People in self.dorms
        counter = 0
        for i in range(len(self.dorms)):
            new = len(self.dorms[i])
            self['pop_size'] += new
            self.dorm_offsets[i] = counter 
            counter += new

        self.dorm_offsets[-1] = self['pop_size'] #The population size is added as a convenience for later functions
        #self.update_pars(self.pars, **kwargs)   # We have to update the parameters again in case any of the above overwrote a user provided value


    def init_students(self, save_pop=False, load_pop=False, popfile=None, verbose=None, **kwargs):
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
        self.people = cvpop.make_people(self, save_pop=save_pop, popfile=popfile, verbose=verbose, **kwargs)
        self.people.initialize() # Fully initialize the people

        # Create the seed infections
        inds = cvu.choose(self['pop_size'], self['pop_infected'])
        self.people.infect(inds=inds, layer='seed_infection')
        return

