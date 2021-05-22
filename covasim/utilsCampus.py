'''
    A module of helper functions for certain simCampus features. Generally, it holds routines that are based on or fill similar roles to those
    in utils.py
'''

def compute_importations(inf_inds, beta, rel_sus, sus, quar, quar_factor, immunity_factors):
    '''
       This routine will take a list of proposed agents to receive an imported infection and decide which ones should receive an infection based on
       their rel_susc value. 
    '''
    #Establish the susceptibility
    f_quar      = ~quar +  quar * quar_factor
    adjustedSus = rel_sus * sus * f_quar * (1-immunity_factors)

    #Establish the infections
    betas = beta * adjustedSus[targets[inf_inds]]

    

    return
