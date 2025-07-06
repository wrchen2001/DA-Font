import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder import dec_builder, Integrator
from model.content_encoder import content_enc_builder
from model.references_encoder import comp_enc_builder
from model.dual_attention_hybrid_module import ComponentAttentionBlock, RelationAttentionBlock
from model.memory import Memory

class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, C_in, C, C_out, cfg, comp_enc, dec, content_enc, integrator_args):
        super().__init__()
        self.num_heads = cfg.num_heads  
        self.kshot = cfg.kshot 

        self.component_encoder = comp_enc_builder(C_in, C, **comp_enc)  
        self.mem_shape = self.component_encoder.out_shape  
        assert self.mem_shape[-1] == self.mem_shape[-2]  

        self.Get_style_components = ComponentAttentionBlock()
        self.Get_style_components_1 = ComponentAttentionBlock()
        self.Get_style_components_2 = ComponentAttentionBlock()  
        self.rab = RelationAttentionBlock()  

        self.memory = Memory()

        self.shot = cfg.kshot
        
    
        C_content = content_enc['C_out']
        C_reference = comp_enc['C_out']
        self.content_encoder = content_enc_builder(C_in, C, **content_enc)  

        self.decoder = dec_builder(
            C, C_out, **dec
        )  

        self.Integrator = Integrator(C * 8, **integrator_args, C_content=C_content, C_reference=C_reference)  

        self.Integrator_local = Integrator(C * 8, **integrator_args, C_content=C_content, C_reference=0)   


    def reset_memory(self):  
        """
        reset memory
        """
        self.memory.reset_memory()


    def read_decode(self, target_style_ids, trg_sample_index, content_imgs, learned_components, trg_unis, ref_unis,
                    chars_sim_dict, reset_memory=True, reduction='mean'):
        """
        decode
        :param target_style_ids:
        :param trg_sample_index:
        :param content_imgs:
        :param reset_memory:
        :param reduction:
        :return:
        """

        reference_feats = self.memory.read_chars(target_style_ids, trg_sample_index, reduction=reduction)
        reference_feats = torch.stack([x for x in reference_feats])  

        content_feats = self.content_encoder(content_imgs)  

        try:
            style_components = self.Get_style_components(learned_components, reference_feats)  
            style_components = self.Get_style_components_1(style_components, reference_feats)
            style_components = self.Get_style_components_2(style_components, reference_feats)
        except Exception as e:
            traceback.print_exc()

        sr_features = self.rab(content_feats, learned_components, style_components)  


        global_style_features = self.Get_style_global(trg_unis, ref_unis, reference_feats, chars_sim_dict)
        all_features = self.Integrator(sr_features, content_feats, global_style_features)  
        out = self.decoder(all_features)

        if reset_memory:
            self.reset_memory()

        return out, style_components   



    def encode_write_comb(self, style_ids, style_sample_index, style_imgs, reset_memory=True):
        """
        encode && memory component features
        :param style_ids:
        :param style_sample_index:
        :param style_imgs:
        :param reset_memory:
        :return:
        """

        if reset_memory:
            self.reset_memory()

        feats = self.component_encoder(style_imgs)
        feat_scs = feats["last"]    
        self.memory.write_comb(style_ids, style_sample_index, feat_scs)

        return feat_scs

    def CosineSimilarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

    def Get_style_global(self, trg_unis, ref_unis, reference_feats, chars_sim_dict):

        list_trg_chars = list(trg_unis)
        list_ref_unis = list(ref_unis)   

        B, K, C, H, W = reference_feats.shape 
        global_feature = torch.zeros([B, C, H, W]).cuda()  

        for i in range(0, B):
            distance_0 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][0]]   

            distance_1 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][1]]

            distance_2 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][2]]

            distance_3 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][3]]

            weight = torch.tensor([distance_0, distance_1, distance_2, distance_3])
            t = 1
            weight = F.softmax(weight / t, dim=0)

            global_feature[i] = reference_feats[i][0] * weight[0] + reference_feats[i][1] * weight[1] \
                               + reference_feats[i][2] * weight[2] + reference_feats[i][3] * weight[3]
                               
        return global_feature


    def infer(self, in_style_ids, in_imgs, style_sample_index, trg_style_ids,
              content_imgs, learned_components, trg_unis=None, ref_unis=None, chars_sim_dict=None,
              reduction="mean", k_shot_tag=False):

        in_style_ids = in_style_ids.cuda() 
        in_imgs = in_imgs.cuda()  

        infer_size = content_imgs.size()[0]
        learned_components = learned_components[:infer_size]  

        content_imgs = content_imgs.cuda() 

        reference_feats = self.encode_write_comb(in_style_ids, style_sample_index, in_imgs)  


        if not k_shot_tag:
            reference_feats = reference_feats.unsqueeze(1)  
        else:
            KB, C, H, W = reference_feats.size()
            reference_feats = torch.reshape(reference_feats, (KB // self.shot, self.shot, C, H, W))

        content_feats = self.content_encoder(content_imgs)  
        content_feats = content_feats.cuda()

        style_components = self.Get_style_components(learned_components, reference_feats)  
        style_components = self.Get_style_components_1(style_components, reference_feats)
        style_components = self.Get_style_components_2(style_components, reference_feats)

        sr_features = self.rab(content_feats, learned_components, style_components) 


        if k_shot_tag:
            global_style_features = self.Get_style_global(trg_unis, ref_unis, reference_feats, chars_sim_dict)
            all_features = self.Integrator(sr_features, content_feats, global_style_features)

        else:
            all_features = self.Integrator(sr_features, content_feats, reference_feats.squeeze())

        out = self.decoder(all_features)  

        return out, style_components, sr_features

