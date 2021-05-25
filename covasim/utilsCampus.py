'''
    A module of helper functions for certain simCampus features. Generally, it holds routines that are based on or fill similar roles to those
    in utils.py
'''

import numpy as np

def importation_immunity(inf_inds,immunity_factors):
    '''
       This routine will take a list of proposed agents to receive an imported infection and decide which ones should receive an infection based on
       their current nab level. It's pretty simple right now; the intention is to use the relevant entries in sim.people.sus_imm to establish the 
       the probability that an attempted imported infection leads to an actual infection.
    '''
    #Ignore this. It is a first draft that I am keeping in case I want to rework it. 
    #importation_immunity(inf_inds, beta, rel_sus, sus, quar, quar_factor, immunity_factors)
    #Establish the susceptibility
    #f_quar      = ~quar +  quar * quar_factor
    #adjustedSus = rel_sus * sus * f_quar * (1-immunity_factors)
    #print(immunity_factors)

    #Establish the infections
    #betas = beta * adjustedSus[inf_inds] # Calculate the raw transmission probabilities
    #nonzero_inds     = betas.nonzero()[0] # Find nonzero entries
    #nonzero_inf_inds = inf_inds[nonzero_inds] # Map onto original indices
    #nonzero_betas    = betas[nonzero_inds] # Remove zero entries from beta
    #transmissions    = (np.random.random(len(nonzero_betas)) < nonzero_betas).nonzero()[0] # Compute the actual infections!

    transmissions    = (immunity_factors[inf_inds] < np.random.random(len(inf_inds))).nonzero()[0] # Compute the actual infections!

    return inf_inds[transmissions]
