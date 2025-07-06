import torch
import torch.nn as nn


class CombMemory:
    """
    CombMemory manages storage of style-component features indexed by style and sample.
    """
    def __init__(self):
        self.memory = {}
        self.reset()

    def write(self, style_ids, style_sample_index, sc_feats): 

        assert len(style_ids) == len(style_sample_index) == len(sc_feats), "Input sizes are different"

        for style_id, sample_index, sc_feat in zip(style_ids, style_sample_index, sc_feats): 
            self.write_point(style_id, sample_index, sc_feat)


    def write_point(self, style_id, sample_index, sc_feat):
        """
        Write a single style-component feature to memory.
        """
        sc_feat = sc_feat.squeeze() 
        self.memory.setdefault(style_id.item(), {}) \
                   .setdefault(sample_index.item(), []) \
                   .append(sc_feat)  

    
    def read_point(self, style_id, sample_index, reduction='mean'):
        """
        Read all stored features for a given style and character.
        """

        style_id = int(style_id)  
        sample_index = int(sample_index) 
        sc_feats = self.memory[style_id][sample_index]  

        return torch.stack(sc_feats) 


    def read_char(self, style_id, sample_index, reduction='mean'):
        """
        Read and stack features corresponding to a single character.
        """
        char_feats = []
        comp_feat = self.read_point(style_id, sample_index, reduction)
        char_feats.append(comp_feat)

        return torch.stack(char_feats).squeeze(0)

    def reset(self):
        """
        Reset the memory to empty state.
        """
        self.memory = {}


class Memory(nn.Module):
    """
    Memory wrapper for managing style-component features.
    """

    STYLE_id = -1    #  Reserved or special-purpose style index

    def __init__(self):
        super().__init__()
        self.comb_memory = CombMemory()

    def write_comb(self, style_ids, style_sample_index, sc_feats):
        """
        Batch write component features to memory.
        """
        self.comb_memory.write(style_ids, style_sample_index, sc_feats)

    def write_point_comb(self, style_id, sc_feat):
        self.comb_memory.write_point(style_id, sc_feat)

    def read_chars(self, style_ids, trg_sample_index, reduction='mean', type="both"):
        sc_feats = []
        read_char = self.comb_memory.read_char
        for style_id, sample_index in zip(style_ids, trg_sample_index):
            sc_feat = read_char(style_id, sample_index, reduction)
            sc_feats.append(sc_feat.cuda())
        return sc_feats 

    def read_comb(self, style_ids, reduction='mean'):
        sc_feats = []
        for style_id in zip(style_ids):
            sc_feat = self.comb_memory.read_char(style_id, reduction)
            sc_feats.append(sc_feat.cuda())
        return sc_feats

    def reset_memory(self):
        """
        reset_memory
        """
        self.comb_memory.reset()
