'''
Defines the Student class, which is a subclass of People. It adds and modifies features for use in SimCampus runs. 
'''

import covasim.people as cvppl
import covasim.populationCampus as cvpc

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

        return self.contacts

