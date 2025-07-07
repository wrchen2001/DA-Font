from .base_trainer import BaseTrainer
import utils
from datasets import cyclize
import torch

torch.autograd.set_detect_anomaly = True

class CombinedTrainer(BaseTrainer):
    """
    CombinedTrainer
    Trainer class that combines generator and discriminator training for font generation
    """

    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):  # cls_char
        super().__init__(gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                         logger, evaluator, cv_loaders, cfg)
    

    def train(self, loader, st_step=1, max_step=100000, component_embeddings=None, chars_sim_dict=None):
        """
        Main training loop for the combined generator and discriminator.
        """

        self.gen.train()   
        if self.disc is not None:
            self.disc.train()    

        # Initialize loss trackers
        losses = utils.AverageMeters("g_total", "pixel", "corner", "grid", "disc", "gen", "contrastive")
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni")
        stats = utils.AverageMeters("B_style", "B_target")
        self.step = st_step   
        self.clear_losses()    
        self.logger.info("Start training FewShot ...")

        while True:
            # Main training loop cycling through the data loader
            for (in_style_ids, in_imgs, trg_style_ids, trg_uni_ids, trg_imgs,
                 content_imgs, trg_unis, style_sample_index, trg_sample_index, ref_unis) in cyclize(loader):
                epoch = self.step // len(loader)
                B = trg_imgs.shape[0]  
                stats.updates({
                    "B_style": in_imgs.shape[0],
                    "B_target": B
                })

                # Move data to GPU
                in_style_ids = in_style_ids.cuda()  
                in_imgs = in_imgs.cuda() 
                content_imgs = content_imgs.cuda()  
                trg_uni_disc_ids = trg_uni_ids.cuda()
                trg_style_ids = trg_style_ids.cuda()
                trg_imgs = trg_imgs.cuda()

                # Get detached component embeddings for current batch
                bs_component_embeddings = self.get_codebook_detach(component_embeddings)

                # First pass through generator
                self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs[0])  

                out_1, style_components_1 = self.gen.read_decode(trg_style_ids, trg_sample_index,
                                                                 content_imgs[0],
                                                                 bs_component_embeddings,
                                                                 trg_unis,
                                                                 ref_unis,
                                                                 chars_sim_dict)

                # Second pass through generator with different input 

                self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs[1]) 

                _, style_components_2 = self.gen.read_decode(trg_style_ids, trg_sample_index,
                                                             content_imgs[1],
                                                             bs_component_embeddings,
                                                             trg_unis,
                                                             ref_unis,
                                                             chars_sim_dict)  
                
                # Self-reconstruction inference
                self_infer_imgs, style_components, feat_recons = self.gen.infer(trg_style_ids, trg_imgs[0],
                                                                                trg_style_ids,
                                                                                trg_sample_index, content_imgs[0],
                                                                                bs_component_embeddings)

                # Discriminator real/fake evaluation
                real_font, real_uni = self.disc(trg_imgs[0], trg_style_ids,
                                                trg_uni_disc_ids[0::self.num_postive_samples])

                fake_font, fake_uni = self.disc(out_1.detach(), trg_style_ids,
                                                trg_uni_disc_ids[0::self.num_postive_samples])

                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs.detach(), trg_style_ids,
                                                            trg_uni_disc_ids[0::self.num_postive_samples])
                
                # Update discriminator losses
                self.add_gan_d_loss(real_font, real_uni, fake_font + fake_font_recon,
                                    fake_uni + fake_uni_recon)

                # Discriminator backward pass
                self.d_backward()  
                self.d_optim.step()  
                self.d_scheduler.step()  
                self.d_optim.zero_grad() 

                # Generator evaluation by discriminator
                fake_font, fake_uni = self.disc(out_1, trg_style_ids, trg_uni_disc_ids[0::self.num_postive_samples])

                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs, trg_style_ids,
                                                            trg_uni_disc_ids[0::self.num_postive_samples])
                
                # Update generator losses
                self.add_gan_g_loss(real_font, real_uni, fake_font + fake_font_recon, fake_uni + fake_uni_recon)
                self.add_pixel_loss(out_1, trg_imgs[0], self_infer_imgs)
                self.style_contrastive_loss(style_components_1, style_components_2, self.batch_size)
                if self.step % self.cfg['print_freq'] == 0:
                    self.add_corner_loss(out_1, trg_imgs[0])
                    self.add_grid_loss(out_1, trg_imgs[0])


                # Generator backward pass
                self.g_backward() 
                self.g_optim.step()  
                self.g_scheduler.step()  
                self.g_optim.zero_grad() 

                # Update metrics
                discs.updates({
                    "real_font": real_font.mean().item(),
                    "real_uni": real_uni.mean().item(),
                    "fake_font": fake_font.mean().item(),
                    "fake_uni": fake_uni.mean().item(),
                }, B)

                loss_dic = self.clear_losses()
                losses.updates(loss_dic, B) 

                self.accum_g()   

                # Logging and validation
                if self.step % self.cfg['tb_freq'] == 0:    
                    self.baseplot(losses, discs, stats)

                if self.step % self.cfg['print_freq'] == 0:  
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_reserved() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                if self.step % self.cfg['val_freq'] == 0:  
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                    self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step,
                                                 bs_component_embeddings, chars_sim_dict)
                    self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))

                if self.step >= max_step:  
                    break

                self.step += 1

            if self.step >= max_step:
                break

        self.logger.info("Iteration finished.")


    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f} Cor {L.corner.avg: 7.3f} Grid {L.grid.avg: 7.3f} Contrastive {L.contrastive.avg:7.4f}"
            "  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
