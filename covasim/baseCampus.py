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
        room        (int)  : number of residents in each room of the residence hall
        bathroom    (int[]): number of rooms that share a bathroom on each floor of the residence hall
        floor       (int[]): number of bathrooms on each floor of the residence hall
        dormType (str[]): allows the user to place tag(s) on the dorm object that Intervention classes can then interpret.
                             For example, interventionsCampus.FloorTargetedPools can use a "communal" or "suite" flag (referring
                             to the type of bathroom that the dorm has) to determine how to sample agents.
    '''

    def __init__(self,room,bathroom,floor,dormType = []):
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

        if isinstance(dormType,list):
            for tag in dormType:
                if not isinstance(tag,str):
                    raise ValueError("Every element in dormType must be a str value")
            self.dormType = dormType
        elif isinstance(dormType,str):
            self.dormType = [dormType]
        else:
            raise ValueError("The dormType argument must be either a str or a list of str values.")

    def validate(self):
        '''This function will verify that the Dorm object has a nested structure'''
        for i in set(self['r']):
            if len(set(self['b'][self['r'] == i])) != 1:
                raise RuntimeError("Two agents in the same Dorm room are assigned to different bathrooms.")
            if len(set(self['f'][self['r'] == i])) != 1:
                raise RuntimeError("Two agents in the same Dorm room are assigned to different floors.")

        for i in set(self['b']):
            if len(set(self['f'][self['b'] == i])) != 1:
                raise RuntimeError("Two agents that are assigned to the same Dorm bathroom are assigned to different floors.")

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


class FlexDorm(Dorm):
    '''
    The FlexDorm class provides added flexibility in describing a dorm's structure. Now bathrooms can be shared between different sized rooms,
    and a single floor can contain bathrooms associated with different numbers of rooms. This is useful, for example, when a dorm contains 
    multiple types of floor plans.

    Args:
        bathroom (int[][][]): The layers of the nested list work as follows:
                                Outer List : Elements correspond to a floor
                                Middle List: Elements corresponds to one "type" of bathroom on that floor
                                Inner List : Elements provide the number of occupants in each room for that bathroom type
        floor    (int[][]): The layers of the nested list work as follows:
                                Outer List : Elements corresponds to a floor
                                Inner List : Elements specify the number of each type of bathroom on that floor, as specified in the middle
                                             layer of bathrooms
        dormType (str[]): allows the user to place tag(s) on the dorm object that Intervention classes can then interpret.
                             For example, interventionsCampus.FloorTargetedPools can use a "communal" or "suite" flag (referring
                             to the type of bathroom that the dorm has) to determine how to sample agents.

    **Example**::
    FlexDorm([[[1,2],[2,3]],[[2,3],[3,3],[3,4]]],[[5,4],[3,2,1]])

    This would be consistent with a dorm that has five styles of bathroom and two floors. The first floor has five bathrooms where each is shared
    between a single and  a double room, and the first floor also has four bathrooms where each is shared by a single and a triple room. The second
    floor has three bathrooms where each is shared with a double and a triple, two bathrooms where each is shared by two triples, and one bathroom
    shared between a triple and a quadruple.
    '''
    def __init__(self,bathroom,floor,dormType = []):
        #Make sure the specification makes sense
        if len(bathroom) != len(floor):
            raise RuntimeError("The bathroom and floor arguments specify different numbers of floors.")

        for i in range(len(bathroom)):
            if len(bathroom[i]) > len(floor[i]):
                raise ValueError("The bathroom argument specifies a bathroom structure not used in the floor argument.")
            if len(bathroom[i]) < len(floor[i]):
                raise ValueError("The floor argument references a bathroom structure not present in the bathroom argument.")

        self['r'] = np.array([],dtype = cvd.default_int)
        self['b'] = np.array([],dtype = cvd.default_int)
        self['f'] = np.array([],dtype = cvd.default_int)

        roomCounter = 0
        bathroomCounter = 0
        floorCounter = 0
        #This mass of loops is assuredly inefficient, but it will do for now
        for f in range(len(floor)):
            for suiteType in range(len(floor[f])):
                for j in range(floor[f][suiteType]):
                    for r in bathroom[f][suiteType]:
                        self['r'] = np.append(self['r'],[roomCounter] * r)
                        roomCounter += 1
                    self['b'] = np.append(self['b'],[bathroomCounter] * sum(bathroom[f][suiteType]))
                    bathroomCounter += 1
            self['f'] = np.append(self['f'],[floorCounter] * np.inner([sum(i) for i in bathroom[f]],floor[f]))
            floorCounter += 1


        if isinstance(dormType,list):
            for tag in dormType:
                if not isinstance(tag,str):
                    raise ValueError("Every element in dormType must be a str value")
            self.dormType = dormType
        elif isinstance(dormType,str):
            self.dormType = [dormType]
        else:
            raise ValueError("The dormType argument must be either a str or a list of str values.")

#TODO: Create a testing scenario

    def validateFlexDorm(self,bathroom,floor):
        '''
        Verify that the key-value pairs in the new FlexDorm object have the current number of agents on each floor, in each bathroom
        and in each room. The nested structure of rooms, bathrooms, and floors is validated in Dorm.validate()
        '''
        #Verify the floors have the correct number of agents
        floorCounts = [np.inner([sum(j) for j in bathroom[i]],floor[i]) for i in range(len(floor))]
        if len(floorCounts) != len(set(self['f'])):
            raise RuntimeError("The number of floors in the FlexDorm object does not match the specification.")

        for i in  range(len(floorCounts)):
            if sum(self['f'] == i) != floorCounts[i]:
                raise RuntimeError("At least one floor in the FlexDorm object has an incorrect number of agents")

        #Verify the bathrooms have the correct number of agents
        bathroomCounts = [[sum(bathroom[f][b])] * floor[f][b]  for f in range(len(floor)) for b in range(len(floor[f]))]
        bathroomCounts = [i for j in bathroomCounts for i in j]

        if len(bathroomCounts) != len(set(self['b'])):
            raise RuntimeError("The number of bathrooms in the FlexDorm object does not match the specification.")

        for i in range(len(bathroomCounts)):
            if sum(self['b'] == i) != bathroomCounts[i]:
                raise RuntimeError("At least one bathroom in the FlexDorm object has an incorrect number of agents")

        #Verify the rooms have the correct number of agents
        roomCounts = [bathroom[f][b] * floor[f][b]  for f in range(len(floor)) for b in range(len(floor[f]))]
        roomCounts = [i for j in roomCounts for i in j]

        if len(roomCounts) != len(set(self['r'])):
            raise RuntimeError("The number of rooms in the FlexDorm object does not match the specification.")

        for i in range(len(roomCounts)):
            if sum(self['r'] == i) != roomCounts[i]:
                raise RuntimeError("At least one room in the FlexDorm object has an incorrect number of agents")


def autoCreateDorms(nDorms,room,bathroom,floor,dormName = "dorm",dormType = []):
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
        returnDict[dormName + '_' + str(i)] = Dorm(room,bathroom,floor,dormType) 
    return returnDict
