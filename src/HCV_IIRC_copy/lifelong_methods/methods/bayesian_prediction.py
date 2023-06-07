import torch.nn as nn
import torch.distributed as dist
import torch
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from sklearn.mixture import GaussianMixture
from torchvision.models import ResNet101_Weights

from lifelong_methods.utils import transform_labels_names_to_vector
from lifelong_methods.utils import get_optimizer


class Model(BaseMethod):
    """
    A finetuning (Experience Replay) baseline.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], tasks, config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, tasks, config)

        # setup model
        self.net = resnet101() # ImageNet pretrained feature extractor
        self.net.fc = nn.Linear(2048, 8) # new fc layer (prev)

        self.net.load_state_dict(torch.load("./feature_extractor/isic_feature_extractor.pth"))
        self.net.fc = nn.Linear(2048, self.num_classes) # new fc layer
        # freeze feature extractor
        self.set_parameter_requires_grad(True)

        # fine-tune the last layer
        # self.net.fc = nn.Linear(2048, self.num_classes) # new fc layer

        # params_to_update = self.net.parameters()
        # params_to_update = []
        # for name,param in self.net.named_parameters():
        #     if param.requires_grad == True:
        #         params_to_update.append(param)

        # self.opt, self.scheduler = get_optimizer(
        #     model_parameters=params_to_update, optimizer_type=self.optimizer_type, lr=self.lr,
        #     lr_gamma=self.lr_gamma, lr_schedule=self.lr_schedule, reduce_lr_on_plateau=self.reduce_lr_on_plateau,
        #     weight_decay=self.weight_decay,patience=self.patience,
        # )

        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

        self.class_names_samples = {class_: 0 for class_ in self.class_names_to_idx.keys()}

        # gmm for each class
        # self.gmms = np.array([GaussianMixture(n_components=2, covariance_type="full") for _ in range(self.num_classes)]) gives 97% acc
        self.gmms = np.array(    [[GaussianMixture(n_components=2, covariance_type="full") for _ in range(2048)] for _ in range(self.num_classes) ]  ) 

        self.cur_task_class_outputs = [[] for _ in range(len(self.seen_classes))] # zik for a given class for all samples in the buffer


    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.net.parameters():
                param.requires_grad = False

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) :
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
            prepare_model_for_task function)
        """
        task_data = kwargs["task_data"]
        self.cur_task_class_outputs = [[] for _ in range(self.num_classes)]
        
        for idx in range(len(task_data)):
            labels = task_data.get_labels(idx)
            for label in labels:
                if label in self.class_names_samples.keys():
                    self.class_names_samples[label] += 1
        
        # for class_ in task_data.cur_task:
        #     self.cur_task_class_outputs.append([])

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True, super_class_index=None, sub_class_index=None, task_end: bool = True):
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
        target = y
        assert target.shape[1] == offset_2
        output, _ = self.forward_net(x)
        output = output[:, :offset_2]
        loss = self.bce(output / self.temperature, target)

        # TODO weigh the buffer loss by the self.memory_strength before getting the loss mean (use in_buffer)
        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        predictions = output.ge(0.0)
        return predictions, loss.item()

    def forward(self, x: torch.Tensor, return_output=False) :
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """

        num_seen_classes = len(self.seen_classes)
        output = self.forward_net(x)

        # get output only from feature extractor
        new_classifier = nn.Sequential(*list(self.net.children())[:-1])

        output_copy = new_classifier(x)
        output_copy = output_copy.view(output_copy.shape[0],-1)


        # TODO weigh the buffer loss by the self.memory_strength before getting the loss mean (use in_buffer)

        num_samples = 0 
        for i in self.seen_classes:
            num_samples += self.class_names_samples[i]
        
        prob_list = [[] for _ in range(len(self.seen_classes))] # p_c | z_j for all classes
        for i in self.seen_classes:
            p_c = np.log(self.class_names_samples[i] / num_samples)
            p_zj_c = 0
            for k in range(2048):
                p_zjk_c = self.gmms[self.class_names_to_idx[i]][k].score_samples(output_copy[:,k].detach().cpu().numpy().reshape(-1, 1)) # for all samples in the output, i.e., batch_size
                p_zj_c += p_zjk_c # batch_size, 1
            p_zj_c += p_c
            prob_list[self.class_names_to_idx[i]] = p_zj_c # batch_size, 1
        
        prob_list = np.array(prob_list)
        prob_list = torch.from_numpy(prob_list).float()
        prob_list = prob_list.T
        # prob_list = torch.stack(prob_list, dim=1) # batch_size, num_seen_classes
        # prob_list = prob_list[:, :num_seen_classes]
        # for each sample, find the class with the highest probability
        predictions_index = torch.argmax(prob_list, dim=1) # dimensions = batch_size, 1
        predictions = torch.zeros_like(output)
        for i in range(predictions.shape[0]):
            predictions[i][predictions_index[i]] = 1
            
        
        # prob_list is 115, 128 dims
        # for ind, sample in enumerate(output_copy):
        #     prob_list = [[] for _ in range(len(self.seen_classes))] # p_c | z_j for all classes
        #     for i in  self.seen_classes:
        #         p_c_num = self.class_names_samples[i]
        #         p_c_denom = 0
        #         for j in self.seen_classes:
        #             p_c_denom += self.class_names_samples[j]
        #         log_p_c = np.log(p_c_num / p_c_denom)
        #         # log_p_zj_c = self.gmms[self.class_names_to_idx[i]].predict(sample.detach().cpu().numpy().reshape(1, -1)) 
        #         log_p_zj_c = 0
        #         for num in range(2048):
        #             log_p_zj_c += self.gmms[self.class_names_to_idx[i]][num].predict(sample[num].detach().cpu().numpy().reshape(1, -1))    
        #         prob_list[self.class_names_to_idx[i]].append(log_p_c + log_p_zj_c)

        #     prediction = torch.argmax(torch.tensor(np.array(prob_list)))

        #     predictions[ind][prediction] = 1             

        # max two values are the classes

        # if not task_end:
        # predictions = output.ge(0.0)
        # else:
            # get idx of max two values in prob_list

        predictions= predictions.to(torch.int32)

        if return_output:
            return predictions[:, :num_seen_classes],output[:, :num_seen_classes]
        else:
            return predictions[:, :num_seen_classes]


    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        pass

    def consolidate_task_knowledge(self, dataloader, config, metadata, cur_task_classes, gpu=None, **kwargs) -> None:
        """Takes place after training on each task"""
        # pass each sample through the network and get the output of the last layer
        class_names_to_idx = metadata["class_names_to_idx"]
        num_seen_classes = len(self.seen_classes)
        self.net.eval()
        new_classifier = nn.Sequential(*list(self.net.children())[:-1])

        with torch.no_grad():
            for minibatch in dataloader:
                labels_names = list(zip(minibatch[1], minibatch[2]))
                labels = transform_labels_names_to_vector(
                    labels_names, num_seen_classes, class_names_to_idx
                )
                if gpu is None:
                    images = minibatch[0].to(config["device"], non_blocking=True)
                    labels = labels.to(config["device"], non_blocking=True)
                else:
                    images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                    labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

                if len(minibatch) > 3:
                    if gpu is None:
                        in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                    else:
                        in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                else:
                    in_buffer = None

                outputs = new_classifier(images)

                # for each batch, sort the inputs classwise and append individual entries to cur_task_class_outputs
                
                # fconvert labels from one hot to class index
                labels = torch.argmax(labels,dim=1)
                # flatten outputs so that batch is not considered
                outputs = outputs.view(outputs.shape[0],-1)

                # sort outputs classwise
                # classwise_sort = np.argsort(labels.cpu().numpy(),axis=0)
                # outputs = outputs[classwise_sort,np.arange(outputs.shape[-1])]

                # append output for every sample to corresponding class
                # num_unique_classes_in_batch = np.unique(labels.cpu().numpy(),axis=0).shape[0]
                # num_samples_per_class_in_batch = []*num_unique_classes_in_batch
                # for i in labels:
                    # num_samples_per_class_in_batch[i]+=1

                # classwise_offset = np.cumsum(num_samples_per_class_in_batch)
                for i in range(len(labels.cpu().numpy())):
                    self.cur_task_class_outputs[labels.cpu().numpy()[i]].append(outputs[i]) # i-> class index, j-> sample index



        # for each class, fit a gmm to the outputs
        for index, class_ in enumerate(cur_task_classes):
            self.cur_task_class_outputs[class_] = torch.stack(self.cur_task_class_outputs[class_],dim=0)
            # 2048 gmms for each class
            # 2048 dimensions for each output sample, fit a gmm to each dimensopm over all samples
            for i in range(2048):
                self.gmms[class_][i].fit(self.cur_task_class_outputs[class_][:,i].cpu().numpy().reshape(-1,1)) # fit i th dimension of output of all samples of class_ to a gmm

            # fit a gmm to all dimensions of output of all samples of class_
            # self.gmms[class_] = np.apply_along_axis(fit_gmm, axis=0, arr=self.cur_task_class_outputs[class_].cpu().numpy())



        
def fit_gmm(x):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(x.reshape(-1,1))
    return gmm


class Buffer(BufferBase):
    def __init__(self,
                 config: Dict,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    def _reduce_exemplar_set(self, **kwargs):
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[Dict] = None, **kwargs) :
        """
        update the buffer with the new task exemplars, chosen randomly for each class.

        Args:
            new_task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
        """
        distributed = dist_args is not None
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
                self.add_sample(class_label, image, (label1, label2), rank=rank)
