import torch.nn as nn
import torch.distributed as dist
import torch
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBaseSAL
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import get_optimizer
from lifelong_methods.utils import SubsetSampler, copy_freeze



class Model(BaseMethod):
    """
    A finetuning (Experience Replay) baseline.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], tasks, config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, tasks, config)

        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.criterion=nn.CrossEntropyLoss()
        self.T = 2
        self.old_net = copy_freeze(self.net)
        self.alpha = 1

        ## RRR
        self.saliency_loss = "l1"
        self.l1_reg = True
        self.saliency_method = "gc"
        self.saliency_regularizer = 100

        self.get_optimizer_explanations()

        if self.saliency_loss == "l1":
            self.sal_loss = torch.nn.L1Loss()
        elif self.saliency_loss == "l2":
            self.sal_loss = torch.nn.MSELoss()

        if self.l1_reg:
            self.l1_reg = torch.nn.L1Loss(reduction='sum')


        # XAI
        if self.saliency_method == 'gc':
            print ("Using GradCAM to obtain saliency maps")
            from explanations import GradCAM as Explain
        elif self.saliency_method == 'smooth':
            print ("Using SmoothGrad to obtain saliency maps")
            from explanations import SmoothGrad as Explain
        elif self.saliency_method == 'bp':
            print ("Using BackPropagation to obtain saliency maps")
            from explanations import BackPropagation as Explain
        elif self.saliency_method == 'gbp':
            print ("Using Guided BackPropagation to obtain saliency maps")
            from explanations import GuidedBackPropagation as Explain
        elif self.saliency_method == 'deconv':
            from explanations import Deconvnet as Explain

        self.explainer = Explain(self.args)

    def kaiming_normal_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

    def get_optimizer_explanations(self):

        self.optimizer_explanations,self.scheduler_exp_opt =  get_optimizer(
            model_parameters=self.net.parameters(), optimizer_type=self.optimizer_type, lr=self.lr,
            lr_gamma=self.lr_gamma, lr_schedule=self.lr_schedule, reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            weight_decay=self.weight_decay,patience=self.patience,
        )

        # elif (self.args.train.optimizer=="radam"):
        #     self.optimizer_explanations = RAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def _prepare_model_for_new_task(self, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
            prepare_model_for_task function)
        """
        self.old_net = copy_freeze(self.net)
        self.old_net.eval()
        if self.cur_task_id > 0:
            in_features = self.net.model.output_layer.in_features
            offset_1_o, offset_2_o = self._compute_offsets(self.cur_task_id-1)
            out_features = offset_2_o         

            # Old weight/bias of the last FC layer
            weight = self.net.model.output_layer.weight.data
            bias = self.net.model.output_layer.bias.data
            # New number of output channel of the last FC layer in new model
            # Creat a new FC layer and initial it's weight/bias
            self.net.model.output_layer.weight.data[:out_features] = weight[:out_features]
            self.net.model.output_layer.bias.data[:out_features] = bias[:out_features]

    def _preprocess_target(self, x: torch.Tensor, z: torch.Tensor):
        """Replaces the labels on the older classes with the distillation targets produced by the old network"""
        offset1, offset2 = self._compute_offsets(self.cur_task_id)
        y = z.clone()
        if self.cur_task_id > 0:
            distill_model_output, _ = self.old_net(x)
            distill_model_output = distill_model_output.detach()
            distill_model_output = torch.sigmoid(distill_model_output / self.temperature)
            y[:, :offset1] = distill_model_output[:, :offset1]
        return y

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True, super_class_index=None, sub_class_index=None,verified_super_key_dict=None):
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
        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)

        target = self._preprocess_target(x, y)
        # target=y.clone()

        if verified_super_key_dict is not None:
            for item_id in range(len(target)):
                for key in verified_super_key_dict.keys():
                    if target[item_id][key] ==1:
                    #     target[item_id]*=0

                        for j in range(len(target[item_id])):
                            target[item_id][j]=0

                        target[item_id][verified_super_key_dict[key]]=1
                        target[item_id][key] = 1
                        ##############


        if self.cur_task_id > 0:
            offset_1_o, offset_2_o = self._compute_offsets(self.cur_task_id-1)

            assert target.shape[1] == offset_2
            output, _ = self.forward_net(x)
            output = output[:, :offset_2]

            
            soft_target,_ = self.old_net(x) # output from old model
            
            loss1 = self.bce(output,target) #Binary Cross entropy between output of the new task and label
            
            outputs_S = nn.functional.softmax(output[:,:offset_2_o]/self.T,dim=1) # Using the new softmax to handle outputs
            outputs_T = nn.functional.softmax(soft_target[:,:offset_2_o]/self.T,dim=1)
            # Cross entropy between output of the old task and output of the old model
            loss2 = outputs_T.mul(-1*torch.log(outputs_S))
            loss2 = loss2.sum(1)
            loss2 = loss2.mean()**self.T
            loss = loss1*self.alpha+loss2*(1-self.alpha)
        else:
            assert target.shape[1] == offset_2
            output, _ = self.forward_net(x)
            output = output[:, :offset_2]
            
            loss = self.bce(output / self.temperature, target)

        # TODO weigh the buffer loss by the self.memory_strength before getting the loss mean (use in_buffer)
        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        predictions = output > 0.0
        return predictions, loss.item()
    

    def observe_sal(self, x: torch.Tensor, s: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True):
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
        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)

        explanations, _, _, _ = self.explainer(x, self.model, task_id=self.cur_task_id)

        self.saliency_size = explanations.size()

        # to make predicted explanations (Bx7x7) same as ground truth ones (Bx1x7x7)
        sal_loss = self.sal_loss(explanations.view_as(s), s)
        sal_loss *= self.saliency_regularizer

        try:
            sal_loss.requires_grad = True
            self.optimizer_explanations.zero_grad()
            sal_loss.backward()
            self.optimizer_explanations.step()
        except:
            pass

        

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
        outputs = output[:, :num_seen_classes]
        predictions = outputs > 0.0 # icarl ptm uses greater than equals to here and before for predictions
        if return_output:
            return predictions,output
        else:
            return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        # Add Scheduler for optimizer_explanation
        pass

    def consolidate_task_knowledge(self, **kwargs) -> None:
        """Takes place after training on each task"""
        pass


class Buffer(BufferBaseSAL):
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

    def _construct_exemplar_set(self, task_data: Dataset, model, dist_args: Optional[Dict] = None,**kwargs) -> None:
        """
        update the buffer with the new task exemplars, chosen randomly for each class.

        Args:
            new_task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
        """
        distributed = dist_args is not None
        explainer = kwargs["model"].explainer if kwargs["model"] is not None else None
        if distributed:
            rank = dist_args['rank']
        else:
            rank = 0
        new_class_labels = task_data.cur_task

        for class_label in new_class_labels:
            num_images_to_add = min(self.n_mems_per_cla, self.max_mems_pool_size)
            class_images_indices = task_data.get_image_indices_by_cla(class_label, num_images_to_add)
            if distributed:
                device = torch.device(f"cuda:{dist_args['gpu']}")
                class_images_indices_to_broadcast = torch.from_numpy(class_images_indices).to(device)
                torch.distributed.broadcast(class_images_indices_to_broadcast, 0)
                class_images_indices = class_images_indices_to_broadcast.cpu().numpy()

            for image_index in class_images_indices:
                image, label1, label2 = task_data.get_item(image_index)
                if label2 != NO_LABEL_PLACEHOLDER:
                    warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                self.add_sample(class_label, image, explainer(image, model, task_id=model.cur_task_id), (label1, label2), rank=rank)
