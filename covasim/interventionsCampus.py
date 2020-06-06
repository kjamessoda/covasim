'''
These are additional interventions and helper functions that are useful for SimCampus objects
'''


def weekly_testing(sim):
    '''This helper function will implement a test_num or test_prob intervention on a weekly basis'''
    if sim.t%7 == 0:
        indices = []
    else:
        indices = [True] * sim['pop_size']

    return indices
