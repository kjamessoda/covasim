'''
Defines functions that work analogously to the functions in the population module but geared toward the goals of campusSim
'''

import covasim.defaults as cvd
import covasim.people as cvppl
import covasim.utils as cvu
import numpy as np
import sciris as sc

def make_students(sim, save_pop=False, popfile=None, verbose=None, die=True, reset=False):
    '''
    An analog to population.make_people. It borrows some code from this function to ensure the simulation runs smoothly.
    '''
    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    if verbose is None:
        verbose = sim['verbose']
    if popfile is None:
        popfile = sim.popfile

    if sim.people and not reset:
        return sim.people # If it's already there, just return
    elif sim.popdict and not reset:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove
    elif sim['pop_type'] == 'campus':
        popdict = make_campus(sim)
    else:
        raise RuntimeWarning("populationCampus.make_students only supports the value \'campus\' for \'pop_type\'")
        popdict = make_campus(sim)

    # Ensure prognoses are set
    if sim['prognoses'] is None:
        sim['prognoses'] = cvpars.get_prognoses(sim['prog_by_age'])

    # Actually create the people
    people = cvppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'], contacts=popdict['contacts']) # List for storing the people

    average_age = sum(popdict['age']/pop_size)
    sc.printv(f'Created {pop_size} people, average age {average_age:0.2f} years', 2, verbose)

    if save_pop:
        if popfile is None:
            errormsg = 'Please specify a file to save to using the popfile kwarg'
            raise FileNotFoundError(errormsg)
        else:
            filepath = sc.makefilepath(filename=popfile)
            sc.saveobj(filepath, people)
            if verbose:
                print(f'Saved population of type "{pop_type}" with {pop_size:n} people to {filepath}')

    return people


def make_campus(sim,sex_ratio = 0.5):
    '''
    This function is analogous to population.make_randpop, but it generates its output based on information in a SimCampus object's dorms member.
    It borrows some code from population.make_randpop to make everything run smoothly.
    '''
    pop_size = int(sim['pop_size']) # Number of people

    # Handle sexes and ages
    uids           = np.arange(pop_size, dtype=cvd.default_int)
    #TODO: Sex information should eventually be stored in sim.dorms, as residence halls are usually gender structured. Right now, the sex
    #   data member does not do anything so it does not matter
    sexes          = np.random.binomial(1, sex_ratio, pop_size)
    ages           = cvu.sample(**sim.age_dist,size = pop_size)
    layer_keys     = ['r','b','f','c']
    contacts       = make_dorm_contacts(sim,layer_keys)

    # Store output
    popdict = {}
    popdict['uid'] = uids
    popdict['age'] = ages
    popdict['sex'] = sexes
    popdict['contacts'] = contacts
    popdict['layer_keys'] = layer_keys

    return popdict


def make_dorm_contacts(sim,layers):
    '''
    This function is analogous to population.make_microstructured_contacts, but it incorporates the structure of SimCampus.dorms.
    '''
    pop_size = sim['pop_size'] #For convenience
    contacts_list = [{c:[] for c in layers} for p in range(pop_size)] # Pre-populate

    if 'b' in layers: #Determine the number of contacts for each person in each layer all at once. Room contacts always occur.
        bathroomContacts  = cvu.n_poisson(sim['contacts']['b'], pop_size)
    if 'f' in layers: 
        floorContacts     = cvu.n_poisson(sim['contacts']['f'], pop_size)
    if 'c' in layers:
        communityContacts = cvu.n_poisson(sim['contacts']['c'], pop_size)

    dormIndex = 0
    currentDorm = sim.dorms[dormIndex]
    for i in range(len(contacts_list)):
        if i >= sim.dorm_offsets[dormIndex + 1]:
            dormIndex += 1
            currentDorm = sim.dorms[dormIndex]

        if 'r' in layers:
            j = currentDorm['r'][i - sim.dorm_offsets[dormIndex]]
            contacts_list[i]['r'] = cvu.true(currentDorm['r'] == j) + sim.dorm_offsets[dormIndex] #This is really inefficient but it will do for now

        if 'b' in layers:
            j = currentDorm['b'][i - sim.dorm_offsets[dormIndex]]
            bathroomMates = cvu.true(currentDorm['b'] == j)
            subIndices = cvu.choose_r(len(bathroomMates),bathroomContacts[i])
            contacts_list[i]['b'] = bathroomMates[subIndices] + sim.dorm_offsets[dormIndex] 

        if 'f' in layers:
            j = currentDorm['f'][i - sim.dorm_offsets[dormIndex]]
            floorMates = cvu.true(currentDorm['f'] == j)
            subIndices = cvu.choose_r(len(floorMates),floorContacts[i])
            contacts_list[i]['f'] = floorMates[subIndices] + sim.dorm_offsets[dormIndex]

        if 'c' in layers:
            contacts_list[i]['c'] = create_community_contacts(sim,i,communityContacts[i])

    return(contacts_list)

def create_community_contacts(sim,individual,nContacts):
    '''
    Select community contacts for an individual. Right now, this is just a placeholder. It will in time incorporate demographic info about the agent.
    '''
    return cvu.choose_r(sim['pop_size'],nContacts)
