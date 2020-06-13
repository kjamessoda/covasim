'''
Defines the Student class, which is a subclass of People. It adds and modifies features for use in SimCampus runs. 
'''

import covasim.people as cvppl
import covasim.populationCampus as cvpc
import numpy as np

class Students(cvppl.People):
    def __init__(self, sim, strict=True, **kwargs):
        super().__init__(sim.pars,strict,**kwargs)
        self.sim = sim
        return

    def update_contacts(self):
        ''' This function overwrites the parent function so that the dynamic contacts will continue to reference the SimCampus object's dorms array'''

        # Figure out if anything needs to be done -- e.g. {'h':False, 'c':True}
        dynam_keys = [lkey for lkey,is_dynam in self.pars['dynam_layer'].items() if is_dynam]

        # Loop over dynamic keys
        for lkey in dynam_keys:
            # Remove existing contacts
            self.contacts.pop(lkey)

        # Add to contacts
        new_contacts = cvpc.make_dorm_contacts(self.sim,dynam_keys)
        self.add_contacts(new_contacts)

        for lkey in dynam_keys:
            self.contacts[lkey].validate()

        if self.sim.watcher:
            self.sim.watcher.write(str(len(self.contacts)/self.sim['pop_size'])+",")

        return self.contacts

    def test(self, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0,end_quarantine = False):
        '''
        This function is identical to the superclass function, but it provides the option to remove an agent from quarantine upon a negative result
        using the end_quarantine argument.
        '''
        super().test(inds, test_sensitivity, loss_prob, test_delay)

        if end_quarantine:
            if loss_prob != 0:
                raise ValueError("Currently, end_quarantine cannot be True if loss_prob is non-zero.")
            neg_indices =  np.logical_and(self.date_tested == self.t,self.date_pos_test != self.t)
            self.date_end_quarantine[neg_indices] = self.t + test_delay

        return
