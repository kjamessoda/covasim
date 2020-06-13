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

        if len(self['r']) != len(self['b']) or len(self['r']) != len(self['b']):
            raise RuntimeError("Dorm object has values of inconsistent length across keys.")

        return

    def __len__(self):
        return len(self['r'])

    def __eq__(self,otherDorm):
        if not isinstance(otherDorm,Dorm):
            raise ValueError("One of the objects is not a Dorm object")

        result = None
        if len(self['r']) != len(otherDorm['r']):
            result = False
        else:
            result = (self['r'] == otherDorm['r']).all() and (self['b'] == otherDorm['b']).all() and (self['f'] == otherDorm['f']).all()

        return result

def autoCreateDorms(nDorms,room,bathroom,floor,dormName = "dorm"):
    '''
    Auto-Generate a dict of Dorm objects with identical room-bathroom-floor structures. 

    Args:
        nDorms    (int)  : number of Dorm objects to place into the list
        room      (int)  : number of residents in each room of the residence hall
        bathroom  (int[]): number of rooms that share a bathroom on each floor of the residence hall
        floor     (int[]): number of bathrooms on each floor of the residence hall
        dormName (String): (optional) base name for each Dorm object's key. The actual key name will be this String
                            with an underscore and an integer (e.g., dorm_1)

    Returns
    dict (Dorm); a dictionary of Dorm objects, each with the same structure.
    '''
    returnDict = {}

    for i in range(nDorms):
        returnDict[dormName + '_' + str(i)] = Dorm(room,bathroom,floor) 
    return returnDict
