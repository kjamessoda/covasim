'''
Base classes to support the SimCampus class, analogous to the purpose of base.py
'''

import covasim.base as cvb
import covasim.defaults as cvd
import numpy as np

class Dorm(cvb.FlexDict):
    '''
    The Dorm class is a relatively simple class for recording the spatial structure in one residence hall.

    Args:
        room     (int)  : number of residents in each room of the residence hall
        bathroom (int[]): number of rooms that share a bathroom on each floor of the residence hall
        floor    (int[]): number of bathrooms on each floor of the residence hall
    '''

    def __init__(self,room,bathroom,floor):
        if len(bathroom) != len(floor):
            raise RuntimeError("Dorm objects must have bathroom and floor arguments of equal length.")

        self['r'] = np.concatenate([[i] * room for i in range(np.inner(bathroom,floor))])        

        i = 0 #This strategy is not very efficient but it works for now
        bathroomValue = np.array([],dtype = cvd.default_int)
        for j in range(len(floor)):
            for k in range(floor[j]):
                bathroomValue = np.append(bathroomValue,[i] * room * bathroom[j])
                i += 1
        self['b'] = bathroomValue

        self['f'] = np.concatenate([[i] * floor[i] * bathroom[i] * room for i in range(len(floor))])

    def validate(self):
        '''This function will verify that the Dorm object has a nested structure'''
        for i in set(self['r']):
            if len(set(self['b'][self['r'] == i])) != 1:
                raise RuntimeError("Two agents in the same Dorm room are assigned to different bathrooms.")
            if len(set(self['f'][self['r'] == i])) != 1:
                raise RuntimeError("Two agents in the same Dorm room are assigned to different floors.")

        for i in set(self['b']):
            if len(set(self['f'][self['b'] == i])) != 1:
                raise RuntimeError("Two agents in assigned to the same Dorm bathroom are assigned to different floors.")

        return
