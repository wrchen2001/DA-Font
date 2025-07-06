import torch
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
import cv2 as cv
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading truncated images to avoid errors


class CombTrainDataset(Dataset):
    """
    CombTrainDataset
    """
    def __init__(self, env, env_get, avails, all_content_json, content_font, transform=None):
        self.env = env  # LMDB environment
        self.env_get = env_get  # Function to read image from LMDB

        self.num_Positive_samples = 2  # Number of positive samples per batch
        self.k_shot = 4  # Number of reference images per style

        with open(all_content_json, 'r') as f:     
            self.all_characters = json.load(f)  # Load all available character list

        self.avails = avails  # Available characters per font
        self.unis = sorted(self.all_characters)  # Global sorted unicode list
        self.fonts = list(self.avails)  # List of font names
        self.n_fonts = len(self.fonts)  # Total number of fonts
        self.n_unis = len(self.unis)  # Total number of unique characters

        self.content_font = content_font  # Content font name
        self.transform = transform  # Optional transform for image processing

    def random_get_trg(self, avails, font_name):
        target_list = list(avails[font_name])  # Available characters for given font
        trg_uni = np.random.choice(target_list, self.num_Positive_samples * 5)  # Randomly select extra characters
        return [str(trg_uni[i]) for i in range(0, self.num_Positive_samples * 5)]  # Return unicode strings


    def sample_pair_style(self, font, ref_unis):
        # Sample style reference images from LMDB
        try:
            imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in ref_unis])  
        except:
            return None
        return imgs
    

    def __getitem__(self, index):
        font_idx = index % self.n_fonts  # Loop index to get font
        font_name = self.fonts[font_idx] 
        while True:
            style_unis = self.random_get_trg(self.avails, font_name)  # Randomly select characters
            trg_unis = style_unis[:self.num_Positive_samples]  # Positive target characters
            sample_index = torch.tensor([index])  

            avail_unis = self.avails[font_name]
            ref_unis = style_unis[self.num_Positive_samples:]  # Reference characters for style extraction

            style_imgs = torch.stack([self.sample_pair_style(font_name, ref_unis[i*4:(i+1)*4]) for i in range(0, self.num_Positive_samples)], 0)
            if style_imgs is None:
                continue  # Re-sample if image loading failed

            trg_imgs = torch.stack([self.env_get(self.env, font_name, uni, self.transform)
                                  for uni in trg_unis], 0)  # Target style images

            trg_uni_ids = [self.unis.index(uni) for uni in trg_unis]  # Get unicode index for targets
            font_idx = torch.tensor([font_idx])  # Font index tensor

            content_imgs = torch.stack([self.env_get(self.env, self.content_font, uni, self.transform)
                                      for uni in trg_unis], 0)  # Content font images

            # Return data tuple
            ret = (
                torch.repeat_interleave(font_idx, style_imgs.shape[1]),  # Font indices for style images
                style_imgs,  # Style images
                torch.repeat_interleave(font_idx, trg_imgs.shape[1]),  # Font indices for target images
                torch.tensor(trg_uni_ids),  # Unicode indices
                trg_imgs,  # Target images
                content_imgs,  # Content images
                trg_unis[0],  # First target unicode (string)
                torch.repeat_interleave(sample_index, style_imgs.shape[1]),  # Sample index repeated for style images
                sample_index,  # Sample index
                ref_unis[:self.k_shot]  # Reference unicode list
            )
            return ret

    def __len__(self):
        return sum([len(v) for v in self.avails.values()])  # Total number of samples

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader to process batch output
        """
        (style_ids, style_imgs,
         trg_ids, trg_uni_ids, trg_imgs, content_imgs, trg_unis, style_sample_index, trg_sample_index, ref_unis) = zip(*batch)

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs, 1).unsqueeze_(2),  # Add channel dimension
            torch.cat(trg_ids),
            torch.cat(trg_uni_ids),
            torch.cat(trg_imgs, 1).unsqueeze_(2),
            torch.cat(content_imgs, 1).unsqueeze_(2),
            trg_unis,
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            ref_unis
        )
        return ret


class CombTestDataset(Dataset):
    """
    CombTestDataset
    """

    def __init__(self, env, env_get, target_fu, avails, all_content_json, content_font, language="chn",
                 transform=None, ret_targets=True):

        self.fonts = list(target_fu)  # Font names
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]])  # Number of characters per font
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis]  # List of (font, character) pairs

        self.env = env
        self.env_get = env_get
        self.avails = avails  # Available characters per font

        self.transform = transform  # Optional transforms
        self.ret_targets = ret_targets  # Whether to return ground truth target image
        self.content_font = content_font  # Content font name

        # Language-specific unicode conversion
        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }
        self.to_int = to_int_dict[language.lower()]


    def sample_pair_style(self, avail_unis):
        # Randomly sample 4 reference characters for style
        style_unis = random.sample(avail_unis, 4)
        return list(style_unis)


    def __getitem__(self, index):
        font_name, trg_uni = self.fus[index]  
        font_idx = self.fonts.index(font_name)  
        sample_index = torch.tensor([index])  
        avail_unis = self.avails[font_name]  
        style_unis = self.sample_pair_style(avail_unis)  # Sample reference unis

        try:
            a = [self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis]  # Load style images
        except:
            print(font_name, style_unis)

        style_imgs = torch.stack(a)  # Stack style images
        font_idx = torch.tensor([font_idx])  
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])  # Convert target unicode

        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)  # Load content image

        ret = (
            torch.repeat_interleave(font_idx, len(style_imgs)),
            style_imgs,
            font_idx,
            trg_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img,
            trg_uni,
            style_unis
        )

        if self.ret_targets:
            try:
                trg_img = self.env_get(self.env, font_name, trg_uni, self.transform)  # Load target image
            except:
                trg_img = torch.ones(size=(1, 128, 128))  # Fallback for missing images
            ret += (trg_img,)

        return ret


    def __len__(self):
        return len(self.fus)  # Total samples

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for test batch
        """
        style_ids, style_imgs, trg_ids, trg_unis, style_sample_index, trg_sample_index, content_imgs, trg_uni, style_unis, *left = list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            trg_uni,
            style_unis
        )

        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret


class CombTrain_VQ_VAE_dataset(Dataset):
    """
    Dataset for training VQ-VAE using character images
    """
    def __init__(self, root, transform=None):
        self.img_path = root  # Path to image folder
        self.transform = transform  
        self.imgs = self.read_file(self.img_path)  # Load all image paths


    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list  # Sorted image list


    def __getitem__(self, index):
        img_name = self.imgs[index]  
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)  # Apply optional transforms
        return img

    def __len__(self):
        return len(self.imgs)  # Total images
    

class FixedRefDataset(Dataset):
    '''
    FixedRefDataset
    '''
    def __init__(self, env, env_get, target_dict, ref_unis, k_shot,
                 all_content_json, content_font, language="chn",  transform=None, ret_targets=True):
        '''
        ref_unis: target unis
        target_dict: {style_font: [uni1, uni2, uni3]}
        '''
        self.target_dict = target_dict  
        self.ref_unis = sorted(ref_unis)  
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]  
        with open(all_content_json, 'r') as f:
            self.cr_mapping = json.load(f)  # Load character mapping

        self.content_font = content_font
        self.fonts = list(target_dict)  

        self.env = env
        self.env_get = env_get

        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }
        self.to_int = to_int_dict[language.lower()]


    def sample_pair_style(self, font, style_uni):
        style_unis = random.sample(style_uni, 4)
        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis]) 
        return imgs, list(style_unis)


    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])

        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))  # Exclude target uni from refs
        style_imgs, style_unis = self.sample_pair_style(fname, self.ref_unis)  

        fidces = torch.tensor([fidx]) 
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])  
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis])  

        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)  

        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)),
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img,
            trg_uni,
            style_unis
        )

        if self.ret_targets:
            trg_img = self.env_user_get(self.env_user, fname, trg_uni, self.transform)  # Target image retrieval
            ret += (trg_img,)

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod  
    def collate_fn(batch):
        """
        Collate function for fixed reference dataset
        """
        style_ids, style_imgs, trg_ids, trg_unis, style_uni, style_sample_index, trg_sample_index, content_imgs, trg_uni, style_unis, *left = list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_uni),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            trg_uni,
            style_unis
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret









