import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze


class Model(BaseMethod):
    """
    An  implementation of modified version of iCaRL that doesn't use the nearest class mean during inference
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], tasks, config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, tasks, config)

        self.old_net = copy_freeze(self.net)

        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]):
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def _prepare_model_for_new_task(self, **kwargs):
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It copies the old network and freezes it's gradients.
        """
        self.old_net = copy_freeze(self.net)
        self.old_net.eval()

    def _preprocess_target(self, x: torch.Tensor, y: torch.Tensor):
        """Replaces the labels on the older classes with the distillation targets produced by the old network"""
        offset1, offset2 = self._compute_offsets(self.cur_task_id)
        y = y.clone()
        if self.cur_task_id > 0:
            distill_model_output, _ = self.old_net(x)
            distill_model_output = distill_model_output.detach()
            distill_model_output = torch.sigmoid(distill_model_output / self.temperature)
            y[:, :offset1] = distill_model_output[:, :offset1]
        return y

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True, super_class_index=None, sub_class_index=None):
        """
        The method used for training and validation, returns a tensor of model predictions and the loss
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images
            y (torch.Tensor): A 2-d batch indicator tensor of shape (number of samples x number of classes)
            in_buffer (Optional[torch.Tensor]): A 1-d boolean tensor which indicates which sample is from the buffer.
            train (bool): Whether this is training or validation/test

        Returns:
            Tuple[torch.Tensor, float]:
            predictions (torch.Tensor) : a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
            loss (float): the value of the loss
        """
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = self._preprocess_target(x, y)
        assert target.shape[1] == offset_2
        output, _ = self.forward_net(x)
        output = output[:, :offset_2]
        loss = self.bce(output / self.temperature, target)

        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        predictions = output.ge(0.0)
        return predictions, loss.item()

    def forward(self, x: torch.Tensor, return_output=False) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        output, _ = self.forward_net(x)
        output = output[:, :num_seen_classes]
        predictions = output.ge(0.0)
        if return_output:
            return predictions, output
        else:
            return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        pass

    def consolidate_task_knowledge(self, **kwargs) -> None:
        """Takes place after training on each task"""
        pass


class Buffer(BufferBase):
    def __init__(self,
                 config: Dict,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    def _reduce_exemplar_set(self, **kwargs) -> None:
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[dict] = None,
                                model: torch.nn.Module = None, batch_size=1, **kwargs):
        """
        Update the buffer with the new task samples using herding

        Args:
            task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            model (BaseMethod): The current method object to calculate the latent variables
            batch_size (int): The minibatch size
        """
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
            rank = dist_args['rank']
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0
        new_class_labels = task_data.cur_task
        model.eval()

        with task_data.disable_augmentations(): # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    if distributed:
                        device = torch.device(f"cuda:{dist_args['gpu']}")
                        class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                        dist.broadcast(class_data_indices_to_broadcast, 0)
                        class_data_indices = class_data_indices_to_broadcast.cpu().numpy()
                    sampler = SubsetSampler(class_data_indices)
                    class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler)
                    latent_vectors = []
                    for minibatch in class_loader:
                        images = minibatch[0].to(device)
                        output, out_latent = model.forward_net(images)
                        out_latent = out_latent.detach()
                        out_latent = F.normalize(out_latent, p=2, dim=-1)
                        latent_vectors.append(out_latent)
                    latent_vectors = torch.cat(latent_vectors, dim=0)
                    class_mean = torch.mean(latent_vectors, dim=0)

                    chosen_exemplars_ind = []
                    exemplars_mean = torch.zeros_like(class_mean)
                    while len(chosen_exemplars_ind) < min(self.n_mems_per_cla, len(class_data_indices)):
                        potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(chosen_exemplars_ind) + latent_vectors) \
                                                   / (len(chosen_exemplars_ind) + 1)
                        distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
                        shuffled_index = torch.argmin(distance).item()
                        exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
                        exemplar_index = class_data_indices[shuffled_index]
                        chosen_exemplars_ind.append(exemplar_index)
                        latent_vectors[shuffled_index, :] = float("inf")

                    for image_index in chosen_exemplars_ind:
                        image, label1, label2 = task_data.get_item(image_index)
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2), rank=rank)
