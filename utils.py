import torch
import numpy as np
import random
import os
import pickle
from config import Config
config = Config()



class Memory:
    def __init__(self, max_len=None):
        # What are the current targets for the model?
        self.experience_targets = []
        # What are the inputs for the saved experiences?
        self.experience_data    = []
        # What are the correct options (given the context)
        self.experience_context = []
        # What was the outcome (P or N)
        self.experience_outcome = []
        # What is the group (this is used for CL)
        self.experience_group   = []
        # What is the id of the experience
        self.experience_ids     = []
        # How many memories do we have?
        self.memories_stored = 0
    
    def __len__(self):
        return self.memories_stored
    
    def __add__(self, other_memory):
        assert type(other_memory) == Memory
        if len(other_memory):
            if not len(self):
                self.experience_targets = other_memory.experience_targets
                self.experience_data    = other_memory.experience_data
                self.experience_context = other_memory.experience_context
                self.experience_ids     = other_memory.experience_ids
                self.experience_outcome = other_memory.experience_outcome
                self.experience_group   = other_memory.experience_group
                self.memories_stored    = other_memory.memories_stored
            else:
                self.experience_targets = torch.cat((self.experience_targets,other_memory.experience_targets))
                self.experience_data = torch.vstack((self.experience_data,other_memory.experience_data))
                self.experience_context = np.concatenate((self.experience_context,other_memory.experience_context))
                self.experience_ids.extend(list(range(self.memories_stored, self.memories_stored + other_memory.memories_stored)))
                self.experience_outcome = np.concatenate((self.experience_outcome, other_memory.experience_outcome)) 
                self.experience_group   = np.concatenate((self.experience_group, other_memory.experience_group))
                self.memories_stored += other_memory.memories_stored
        return self
        
    
    def add_memories(self, experience_data, experience_targets, experience_context, experience_outcome, experience_group):
        if len(experience_targets):
            if not len(self):
                self.experience_targets = experience_targets
                self.experience_data    = experience_data
                self.experience_context = experience_context
                self.experience_ids     = list(range(len(experience_targets)))
                self.experience_outcome = experience_outcome
                self.experience_group   = experience_group
                self.memories_stored    += len(experience_targets)
            else:
                self.experience_targets = torch.cat((self.experience_targets,experience_targets))
                self.experience_data = torch.vstack((self.experience_data,experience_data))
                self.experience_context = np.concatenate((self.experience_context,experience_context))
                self.experience_ids.extend(list(range(self.memories_stored, self.memories_stored + len(experience_targets))))
                self.experience_outcome = np.concatenate((self.experience_outcome, experience_outcome)) 
                self.experience_group   = np.concatenate((self.experience_group, experience_group))
                self.memories_stored += len(experience_targets)
    
    def shuffle(self):
        if len(self):
            indices = list(range(len(self)))
            random.shuffle(indices)
            # shuffle the keys
            self.experience_targets = self.experience_targets[indices]
            self.experience_data    = self.experience_data[indices]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[indices]
                self.experience_ids     = [self.experience_ids[i] for i in indices]
                self.experience_outcome = [self.experience_outcome[i] for i in indices]
                self.experience_group   = [self.experience_group[i] for i in indices]

        
    def unshuffle(self):
        unshuffle_ids = [i[0] for i in sorted(enumerate(self.experience_ids), key=lambda x:x[1])]
        if len(self):
            self.experience_targets = self.experience_targets[unshuffle_ids]
            self.experience_data    = self.experience_data[unshuffle_ids]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[unshuffle_ids]
                self.experience_outcome = [self.experience_outcome[i] for i in unshuffle_ids]
                self.experience_group   = [self.experience_group[i] for i in unshuffle_ids]
                self.experience_ids     = [self.experience_ids[i] for i in unshuffle_ids]

    def write(self, save_dir, num_written=""):
        with open(save_dir + f'classifier_memory_{num_written}.pkl', 'wb') as handle:
            pickle.dump(self, handle)
    
    def read(self, save_dir):
        with open(save_dir +  'classifier_memory.pkl', 'rb') as handle:
            loaded_content = pickle.load(self, handle)
            self.experience_targets = loaded_content.experience_targets
            self.experience_data    = loaded_content.experience_data
            self.experience_context = loaded_content.experience_context
            self.experience_outcome = loaded_content.experience_outcome
            self.experience_group   = loaded_content.experience_group
            self.experience_ids     = loaded_content.experience_ids
            self.memories_stored = loaded_content.memories_stored
    
    def from_file(self, save_dir, memory_id):
        with open(save_dir + f'classifier_memory_{memory_id}.pkl', 'rb') as handle:
            obj = pickle.load(handle)
        self.experience_targets = obj.experience_targets
        self.experience_data    = obj.experience_data
        self.experience_context = obj.experience_context
        self.experience_outcome = obj.experience_outcome
        self.experience_group   = obj.experience_group
        self.experience_ids     = obj.experience_ids
        self.memories_stored    = obj.memories_stored
