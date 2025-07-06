import cv2
import numpy as np

import copy
import random
import torch
import torch.nn.functional as F
import utils
from pathlib import Path
from .trainer_utils import *
try:
    from apex import amp
except ImportError:
    print('failed to import apex')


class BaseTrainer:
    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):
        """
        BaseTrainer initialization for GAN-based font generation training.
        """
        self.gen = gen  # Generator model
        self.gen_ema = copy.deepcopy(self.gen)  # EMA generator

        self.g_optim = g_optim  # Generator optimizer
        self.g_scheduler = g_scheduler  # Generator scheduler
        self.disc = disc  # Discriminator model
        self.d_optim = d_optim  # Discriminator optimizer
        self.d_scheduler = d_scheduler  # Discriminator scheduler
        self.cfg = cfg  # Configuration dictionary

        self.cross_entropy_loss = nn.CrossEntropyLoss()  # Contrastive loss

        self.batch_size = cfg.batch_size   
        self.num_component_object = cfg.num_embeddings    
        self.num_channels = cfg.num_channels     
        self.num_postive_samples = cfg.num_positive_samples   

        # Model parallel initialization
        [self.gen, self.gen_ema, self.disc], [self.g_optim, self.d_optim] = self.set_model(
            [self.gen, self.gen_ema, self.disc],
            [self.g_optim, self.d_optim],
        )

        self.logger = logger
        self.evaluator = evaluator
        self.cv_loaders = cv_loaders

        self.step = 1  # Training step counter

        self.g_losses = {}  # Generator loss dictionary
        self.d_losses = {}  # Discriminator loss dictionary

        self.projection_style = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        ).cuda()   


    def set_model(self, models, opts):
        return models, opts


    def clear_losses(self):
        """ Integrate and clear generator and discriminator loss dictionaries. """
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())

        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})

        self.g_losses = {}
        self.d_losses = {}

        return loss_dic
    

    def accum_g(self, decay=0.999):
        """ Exponential moving average (EMA) for generator. """
        par1 = dict(self.gen_ema.named_parameters())
        par2 = dict(self.gen.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))


    def train(self):
        return
    
    def get_codebook_detach(self, component_embeddings):
        """ Detach codebook embeddings for current batch. """
        component_objects = torch.zeros(self.num_component_object, self.num_channels).cuda()
        component_objects = component_objects + component_embeddings
        component_objects = component_objects.unsqueeze(0)
        component_objects = component_objects.repeat(self.batch_size, 1, 1)
        return component_objects.detach()

    def add_pixel_loss(self, out, target, self_infer):
        """ Add pixel-level L1 reconstruction loss. """
        loss1 = F.l1_loss(out, target, reduction="mean") * self.cfg['pixel_w']
        loss2 = F.l1_loss(self_infer, target, reduction="mean") * self.cfg['pixel_w']
        self.g_losses['pixel'] = loss1 + loss2
        return loss1 + loss2

    
    def add_corner_loss(self, out, target, max_corners = 100, quality_level = 0.01, min_distance = 10):
        """ Add corner-based geometric consistency loss using OpenCV corner detection.  """
        
        batch_size = out.shape[0]
        total_loss = 0.0
        valid_samples = 0

        for i in range(batch_size):
            out_image = out[i]
            tg_image = target[i]

            if isinstance(tg_image, torch.Tensor):
                tg_image = tg_image.cpu().detach().numpy()
            if isinstance(out_image, torch.Tensor):
                out_image = out_image.cpu().detach().numpy()
            
            if tg_image.ndim == 3 and tg_image.shape[0] == 1:
                tg_image = tg_image.squeeze(0)
            if out_image.ndim == 3 and out_image.shape[0] == 1:
                out_image = out_image.squeeze(0)
            
            tg_image = (tg_image*255).astype(np.uint8) if tg_image.max() <=1 else tg_image.astype(np.uint8)
            out_image = (out_image*255).astype(np.uint8) if out_image.max() <=1 else out_image.astype(np.uint8)

            tg_corners = cv2.goodFeaturesToTrack(tg_image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
            out_corners = cv2.goodFeaturesToTrack(out_image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
            
            if tg_corners is None or out_corners is None:
                continue
            
            tg_corners = np.squeeze(tg_corners) / np.array([tg_image.shape[1], tg_image.shape[0]])    
            out_corners = np.squeeze(out_corners) / np.array([out_image.shape[1], out_image.shape[0]])

            distances = []
            for tg_corner in tg_corners:
                min_dist = np.min(np.linalg.norm(out_corners-tg_corner, axis=1))
                distances.append(min_dist)
            for out_corner in out_corners:
                min_dist = np.min(np.linalg.norm(tg_corners - out_corner, axis=1))
                distances.append(min_dist)

            distances = np.array(distances)
            sample_loss = np.mean(distances)
            
            total_loss = total_loss + sample_loss

            valid_samples +=1

        if valid_samples > 0:
            corner_loss = (total_loss / valid_samples) * self.cfg["corner_w"]

            self.g_losses["corner"] = corner_loss
        else:
            self.g_losses["corner"] = 0.0
        
        return corner_loss


    def add_grid_loss(self, out, target, grid_size = 16, patch_size = 5):
        """ Add grid-based local feature consistency loss. """

        def create_elastic_grid(image, grid_size):
            B, _, H, W = image.size()
            y, x = torch.meshgrid(
                torch.linspace(0, H-1, steps=grid_size),
                torch.linspace(0, W-1, steps=grid_size)
            )
            grid_points = torch.stack([x, y], dim = 1).view(-1, 2)
            grid_points = grid_points.repeat(B, 1, 1).to(image.device)
            return grid_points
        
        def extract_local_features(image, grid_points, patch_size):

            B, C, H, W = image.size()
            num_points = grid_points.size(1)
            half_patch = patch_size//2

            padded_image = F.pad(image, (half_patch, half_patch, half_patch, half_patch))
            local_features = []

            for i in range(num_points):
                x, y = grid_points[:, i, 0].long(), grid_points[:, i, 1].long()
                patches = []
                for b in range(B):
                    patch = padded_image[b, :, y[b]:y[b]+patch_size, x[b]:x[b]+patch_size]
                    patches.append(patch.contiguous().view(-1))
                local_features.append(torch.stack(patches))
            
            local_features = torch.stack(local_features, dim=1)
            features = F.normalize(local_features, p = 2, dim=-1)
            return features
    
        assert out.size() == target.size()

        grid_points = create_elastic_grid(out, grid_size)
        out_features = extract_local_features(out, grid_points, patch_size)
        target_features = extract_local_features(target, grid_points, patch_size)

        grid_loss = F.mse_loss(out_features, target_features) * self.cfg["grid_w"]

        self.g_losses['grid'] = grid_loss

        return grid_loss
    

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        """
        Compute contrastive loss for feature vectors.
        """
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))

        return loss


    def take_contrastive_feature(self, input):
        """ Project and normalize input feature vectors for contrastive learning. """
        out = self.projection_style(input)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out


    def style_contrastive_loss(self, style_components_1, style_components_2, batch_size):
        """ Compute contrastive loss between style components. """

        _, N, _ = style_components_1.shape   
        style_contrastive_loss = 0   
        if self.cfg['contrastive_w'] == 0: 
            return style_contrastive_loss

        style_up = self.take_contrastive_feature(style_components_1)     
        style_down = self.take_contrastive_feature(style_components_2)

        for s in range(batch_size):

            if s == 0:
                negative_style_up = style_up[1:].transpose(0, 1)
                negative_style_down = style_down[1:].transpose(0, 1)

            if s == batch_size - 1:
                negative_style_up = style_up[:batch_size - 1].transpose(0, 1)
                negative_style_down = style_down[:batch_size - 1].transpose(0, 1)

            else: 
                negative_style_up = torch.cat([style_up[0:s], style_up[s + 1:]], 0).transpose(0, 1)
                negative_style_down = torch.cat([style_down[0:s], style_down[s + 1:]], 0).transpose(0, 1)

            index_up = torch.LongTensor(random.sample(range(batch_size - 1), 5))
            index_down = torch.LongTensor(random.sample(range(batch_size - 1), 5))

            negative_style_up = torch.index_select(negative_style_up, 1, index_up.cuda())
            negative_style_down = torch.index_select(negative_style_down, 1, index_down.cuda())


            for i in range(N):
                style_comparisons_up = torch.cat([style_down[s][i:i + 1], negative_style_up[i]], 0) 

                style_contrastive_loss += self.compute_contrastive_loss(style_up[s][i:i + 1],
                                                                        style_comparisons_up, 0.2, 0)

                style_comparisons_down = torch.cat([style_up[s][i:i + 1], negative_style_down[i]], 0)

                style_contrastive_loss += self.compute_contrastive_loss(style_down[s][i:i + 1],
                                                                        style_comparisons_down, 0.2, 0)

        style_contrastive_loss /= N 

        style_contrastive_loss *= self.cfg['contrastive_w']   
        self.g_losses['contrastive'] = style_contrastive_loss  

        return style_contrastive_loss


    def add_gan_g_loss(self, real_font, real_uni, fake_font, fake_uni):
        """
        Add generator adversarial loss.
        """

        if self.cfg['gan_w'] == 0.:
            return 0.

        g_loss = -(fake_font.mean() + fake_uni.mean())
        g_loss *= self.cfg['gan_w']
        self.g_losses['gen'] = g_loss

        return g_loss
    

    def add_gan_d_loss(self, real_font, real_uni, fake_font, fake_uni):
        """
        Add discriminator adversarial loss. 
        """

        if self.cfg['gan_w'] == 0.:
            return 0.

        d_loss = (F.relu(1. - real_font).mean() + F.relu(1. + fake_font).mean()) + \
                 F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()

        d_loss *= self.cfg['gan_w']
        self.d_losses['disc'] = d_loss

        return d_loss


    def d_backward(self):
        """
        Backward pass for discriminator.
        """
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()
    

    def g_backward(self):
        """
        Backward pass for generator.
        """
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()

    
    def save(self, cur_loss, method, save_freq=None):
        """
        Save model checkpoint.

        Args:
            method: all / last
                all: save checkpoint by step
                last: save checkpoint to 'last.pth'
                all-last: save checkpoint by step per save_freq and
                          save checkpoint to 'last.pth' always
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method == 'last' or method == 'all-last':
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,
            'loss': cur_loss
        }

        if self.disc is not None:
            save_dic['discriminator'] = self.disc.state_dict()
            save_dic['d_optimizer'] = self.d_optim.state_dict()
            save_dic['d_scheduler'] = self.d_scheduler.state_dict()

        ckpt_dir = self.cfg['work_dir'] / "checkpoints" / self.cfg['unique_name']
        step_ckpt_name = "{:06d}-{}.pth".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pth"
        step_ckpt_path = Path.cwd() / ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            if last_save:
                utils.rm(last_ckpt_path)
                last_ckpt_path.symlink_to(step_ckpt_path)
                log += " and symlink to {}".format(last_ckpt_path)

        if not step_save and last_save:
            utils.rm(last_ckpt_path)  
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))



    def baseplot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,
            "train/grid_loss": losses.grid.val,
            "train/corner_loss": losses.corner.val,
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_loss': losses.disc.val,
                'train/g_loss': losses.gen.val,
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,
            })


    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  Cor {L.corner.avg: 7.3f}  Grid {L.grid.avg: 7.3f} D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_uni {D.real_uni_acc.avg:7.3f}  F_uni {D.fake_uni_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
