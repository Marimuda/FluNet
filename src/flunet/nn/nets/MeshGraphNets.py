import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lightning import LightningModule
from einops import rearrange, reduce, repeat

from torchmetrics import MeanMetric

from flunet.nn.loss import L2Loss
from flunet.utils.common import NodeType
from flunet.utils.Normalization import Normalizer
from flunet.utils.types import *

# Notes: I don't need get_network_config, it is taken from hydra specifications.
#   add_noise_to_mesh_nodes is handled in _extract_features -> _add_noise, which is called within training_step
# TODO: Compare calculate_loss_normalizer and self.output_normalizer, are they doing the same thing?

# Notes: Color is only used for cloud points. TODO: REMOVE COLOR FROM MESH GRAPH NETS
# Notes: poisson (only 2D): Tensor containing poisson ratio of the current data sample. TODO: REMOVE POISSON.
# Notes: add_static_tissue_info_batched: Is datset specific, not of interest.
# Notes: According to convert_to_mgn_hetro: It looks like hetrogeneous = Hypergraph with two edge types and one node type. According to MeshGraphNets. TODO: Implement relevant Hetro functionality

# Notes: I think it looks too clumsy to control internally whether everything is homo or hetro.
#   I think composition Base -> Homo / Hetro overwriting differences is the best approach
#   That means, I don't need to track Homo / Hetro all the time.

# additional_eval is only for the imputation parameters -> No use at the moment.

# NOTE: self_extract_features has to be implemented in both HOMO / HETRO
# TODO: Examine meshgraphnets-torch closer for multigraph - kernels - encoder - processor - decoder, DataSet


class MeshGraphNetsLitModule(LightningModule):
    def __init__(
        self, net: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, config: ConfigDict
    ):
        super(MeshGraphNetsLitModule, self).__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        #self.automatic_optimization = False
        self.field = config.get("field")
        self.target_field = "target_" + self.field

        self.net = net
        
        # loss function
        self.criterion = L2Loss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.node_normalizer = Normalizer(size=self.net.node_feat_dim + NodeType.SIZE, name="node_normalizer")
        self.edge_normalizer = Normalizer(size=self.net.edge_feat_dim, name="edge_normalizer")
        self.output_normalizer = Normalizer(size=self.net.out_node_feat_dim, name="output_normalizer")


    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def _extract_features(
        self, graph, is_training=False, add_noise=False, use_world_edges=None, use_mesh_coordinates=None, mgn_hetro=None
    ):
        if add_noise:
            # Add noise like in original implementation
            self._add_noise(graph)

        # build feature vectors for each node and edge
        length_trajectory = graph[self.field].shape[1]  # should be 598
        node_type = F.one_hot(graph.node_type[:, 0].to(torch.int64), NodeType.SIZE)
        # n: number of node; f: feature dims; l: trajectory length
        node_type = repeat(node_type, "n f -> n l f", l=length_trajectory)
        node_features = torch.cat([graph[self.field], node_type], dim=-1)  # (num_nodes, length_traj, feat_dim)

        senders, receivers = graph.edge_index
        relative_mesh_pos = graph.mesh_pos[senders] - graph.mesh_pos[receivers]
        edge_features = torch.cat([relative_mesh_pos, torch.norm(relative_mesh_pos, dim=-1, keepdim=True)], dim=-1)
        # (num_edges, length_traj, feat_dim)
        edge_features = repeat(edge_features, "n f -> n l f", l=length_trajectory)

        # normalization
        node_features = self.node_normalizer(node_features, is_training)
        edge_features = self.edge_normalizer(edge_features, is_training)
        return node_features, edge_features

    def _add_noise(self, graph):
        length_trajectory = graph[self.field].shape[1]
        mask = torch.eq(graph.node_type, NodeType.NORMAL)
        mask = repeat(mask, "n f -> n l f", l=length_trajectory)

        noise = torch.normal(
            mean=0.0, std=self.hparams.config.noise_scale, size=graph[self.field].shape, dtype=torch.float32
        ).to(mask.device)

        noise = torch.where(mask, noise, torch.zeros_like(noise).to(mask.device))
        graph[self.field] += noise
        graph[self.target_field] += (1.0 - self.hparams.config.noise_gamma) * noise

#    def l2_loss(self, prediction, graph, start, end, is_training=True):
#        # build target velocity change
#        cur_field = graph[self.field][:, start:end]
#        target_field = graph[self.target_field][:, start:end]
#        field_change = target_field - cur_field
#        target_normalized = self.output_normalizer(field_change, accumulate=is_training)

#        # build loss
#        node_type = rearrange(graph.node_type, "n 1 -> n")
#        loss_mask = torch.logical_or(torch.eq(node_type, NodeType.NORMAL), torch.eq(node_type, NodeType.OUTFLOW))
#        error = reduce((target_normalized - prediction) ** 2, "n l f -> n l", "sum")
#        loss = torch.mean(error[loss_mask])
#        return loss

    def _build_target_velocity_change(self, graph, start, end, is_training=True):
        # build target velocity change
        cur_field = graph[self.field][:, start:end]
        target_field = graph[self.target_field][:, start:end]
        field_change = target_field - cur_field
        target_normalized = self.output_normalizer(field_change, accumulate=is_training)
        return target_normalized

    def model_step(self, batch: Any, batch_idx: Any, is_training: bool = True):
        # Extract Features
        in_node_features, in_edge_features = self._extract_features(batch, is_training=is_training, add_noise=True)

        traj_length = batch[self.field].shape[
            1
        ]  # TODO: Check what alternative there is to self.field, what does MMGN use?
        small_step = self.hparams.config.accumulate_step_size
        num_steps = int(math.ceil(traj_length / small_step))

        accumulate_loss = 0.0
        #optimizer = self.optimizers()

        for i in range(0, num_steps):  # solve the issue of out of memory
            edge_index = batch.edge_index
            start = i * small_step
            end = (i + 1) * small_step
            prediction, _, _ = self.net(edge_index, in_node_features[:, start:end], in_edge_features[:, start:end])

            target_normalized = self._build_target_velocity_change(batch, start, end, is_training)
            loss = self.criterion(prediction, batch, target_normalized)

            if self.current_epoch == 0 and is_training:  # First epoch to accumulate data for normalization terms
                continue
            #loss.backward()  # NOTE: Is this working or do I need to manually call self.manual_backward(loss)
            accumulate_loss += loss.item()

        
        return loss, accumulate_loss, traj_length
        

    def training_step(self, batch, batch_idx):
        loss, accumulate_loss, traj_length = self.model_step(batch, batch_idx)

        if self.current_epoch == 0:# First epoch to accumulate data for normalization terms
            return

        self.train_loss(loss) # accumulate multibatch if needed
        self.log("train/loss", accumulate_loss / traj_length, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
#        in_node_features, in_edge_features = self._extract_features(batch, is_training=False, add_noise=False)
#        traj_length = batch[self.field].shape[
#            1
#        ]  # TODO: Check what alternative there is to self.field, what does MMGN use?
#        small_step = self.hparams.config.accumulate_step_size
#        num_steps = int(math.ceil(traj_length / small_step))
#        accumulate_loss = 0.0
#        for i in range(0, num_steps):
#            edge_index = batch.edge_index
#            start = i * small_step
#            end = (i + 1) * small_step
#
#            prediction, _, _ = self.net(edge_index, in_node_features[:, start:end], in_edge_features[:, start:end])
#            loss = self.criterion(prediction, batch, start, end, is_training=False)
#            accumulate_loss += loss.item()
        loss, accumulate_loss, traj_length = self.model_step(batch, batch_idx, is_training=False)
        self.val_loss(loss)
        self.log("val/loss", accumulate_loss / traj_length, on_step=True, on_epoch=True, logger=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Integrate model outputs."""
        node_features, edge_features = self._extract_features(batch, is_training=False, add_noise=False)

        traj_length = batch[self.field].shape[1]
        small_step = self.hparams.config.accumulate_step_size
        num_steps = int(math.ceil(traj_length / small_step))
        predictions = []
        for i in range(0, num_steps):  # solve the issue of out of memory
            edge_index = batch.edge_index
            start = i * small_step
            end = (i + 1) * small_step
            prediction, _, _ = self.net(edge_index, node_features[:, start:end], edge_features[:, start:end])

            field_update = self.output_normalizer.inverse(prediction)
            predict = batch[self.field] + field_update
            predictions.append(predict)

        predictions = rearrange(predictions, "b n l f -> n (b l) f")
        return predictions.cpu()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
