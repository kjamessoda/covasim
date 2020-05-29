'''
Defines the SimCampus class, which extends the Sim class to make it more amendable to simulating dormitories
'''

import covasim.sim as cvs


class SimCampus(cvs.Sim):

    def __init__(self, pars=None, datafile=None, datacols=None, label=None, simfile=None, popfile=None, load_pop=False, save_pop=False, **kwargs):
        super().__init__()                
