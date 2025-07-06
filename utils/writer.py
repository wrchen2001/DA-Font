from pathlib import Path
import torch.nn.functional as F
from . import save_tensor_to_image


class Writer:
    """Abstract base class for writing training logs and images"""
    def add_scalars(self, tag_scalar_dic, global_step):
        """Add scalar values to log
        Args:
            tag_scalar_dic: Dictionary of tag names and scalar values
            global_step: Current training step
        """
        raise NotImplementedError()

    def add_image(self, tag, img_tensor, global_step):
        """Add image to log
        Args:
            tag: Identifier for the image
            img_tensor: Image tensor to log
            global_step: Current training step
        """
        raise NotImplementedError()


class DiskWriter(Writer):
    """Writer that saves images to disk"""
    def __init__(self, img_path, scale=None):
        """
        Args:
            img_path: Path to directory where images will be saved
            scale: Optional scaling factor for images
        """
        self.img_dir = Path(img_path)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale

    def add_scalars(self, tag_scalar_dic, global_step):
        """DiskWriter doesn't support scalar logging"""
        pass
        #  raise Exception("DiskWriter supports add_image only")

    def add_image(self, tag, img_tensor, global_step):
        """Save image tensor to disk as PNG file
        Args:
            tag: Used as part of the filename
            img_tensor: Image tensor to save
            global_step: Used in the filename for ordering
        """
        path = self.img_dir / "{:07d}-{}.png".format(global_step, tag)
        save_tensor_to_image(img_tensor, path, self.scale)


class TBWriter(Writer):
    """Writer that logs to TensorBoard"""
    def __init__(self, dir_path, scale=None):
        """
        Args:
            dir_path: Directory for TensorBoard logs
            scale: Optional scaling factor for images
        """
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(dir_path, flush_secs=30)
        self.scale = scale

    def add_scalars(self, tag_scalar_dic, global_step):
        """Log multiple scalar values to TensorBoard"""
        for tag, scalar in tag_scalar_dic.items():
            self.writer.add_scalar(tag, scalar, global_step)

    def add_image(self, tag, img_tensor, global_step):
        """Log image to TensorBoard with optional scaling"""
        if self.scale:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), scale_factor=self.scale, mode='bilinear',
                align_corners=False
            ).squeeze(0)
        self.writer.add_image(tag, img_tensor, global_step)


class TBDiskWriter(TBWriter):
    """Writer that logs to both TensorBoard and disk"""
    def __init__(self, dir_path, img_path, scale=None):
        """
        Args:
            dir_path: Directory for TensorBoard logs
            img_path: Directory for image files
            scale: Optional scaling factor for images
        """
        super().__init__(dir_path)
        self._disk_writer = DiskWriter(img_path, scale)

    def add_image(self, tag, img_tensor, global_step):
        """Save image to disk while maintaining TensorBoard scalar logging"""
        return self._disk_writer.add_image(tag, img_tensor, global_step)