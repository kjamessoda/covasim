'''
Defines the Student class, which is a subclass of People. It adds and modifies features for use in SimCampus runs. 
'''

import covasim.people as cvppl
import covasim.populationCampus as cvpc
import covasim.utils as cvu
import numpy as np

class Students(cvppl.People):
    def __init__(self, sim, strict=True, **kwargs):
        super().__init__(sim.pars,strict,**kwargs)
        self.sim = sim
        return

    def initialize(self):
        ''' Perform initializations. The Students class overwrites its parent class's implementation so grad students can have scaled sus./trans. '''
        #NOTE: Most of this code originated in the parent class.
        self.set_prognoses()
        if self.sim['pop_size'] > self.sim.nonResidentEndIndex:
            self.rel_sus[self.sim.nonResidentEndIndex:] *= self.sim.gradTransmissionScale
            self.rel_trans[self.sim.nonResidentEndIndex:] *= self.sim.gradTransmissionScale 
        self.validate()
        self.initialized = True
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


    def test_pooled(self,allInds,test_sensitivity = 1.0, test_delay = 0):
        '''
        This function simulates a pooled test. See Lohse, S et al. 2020. Lancet Infect Dis 20: https://doi.org/10.1016/S1473-3099(20)30362-5.
        It borrows some code from People.test. 

        IMPORTANT: The function only returns which pools in allInds tested positive and on what simulated day the test results will be available
            It does not take any action based on these results (e.g., it does not quarantine the pool). The caller needs to incorporate such
            actions.

        Args:
            allInds (dict): each value in the dict should be a numpy array containing all the agents in one pool. Currently, the keys are not
                            used.
            test_sensitivity (float): probability of a true positive
            test_delay (int): number of days before test results are ready
        '''
        positiveSchedule = []
        positivePools    = []
        for pool in allInds.values():
            self.tested[pool] = True
            self.date_tested[pool] = self.t
            if (self.infectious[pool]).sum() > 0 and cvu.sample("uniform",par1=0, par2=1, size = 1) < test_sensitivity:
                positiveSchedule += [self.t + test_delay]
                positivePools    += [pool]

        return positiveSchedule, positivePools
                

    def quarantine(self, inds):
        '''
        NOTE: This is legacy code from covasim 1.0. It is needed for certain Intervention classes in the campus simulation type.
        Quarantine selected people starting on the current day. If a person is already
        quarantined, this will extend their quarantine.

        Args:
            inds (array): indices of who to quarantine, specified by check_quar()
        '''

        self.quarantined[inds] = True
        self.date_quarantined[inds] = self.t
        self.date_end_quarantine[inds] = self.t + self.pars['quar_period']
        return

