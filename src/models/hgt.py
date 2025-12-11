import logging

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from dgl.nn.pytorch.conv import HGTConv
from torch import nn

from src.config import ProtonConfig, conf
from src.samplers.fixed_sampler import FixedSampler
from src.utils import calculate_metrics

from .bilinear_decoder import BilinearDecoder

_logger = logging.getLogger(__name__)


class HGT(pl.LightningModule):
    def __init__(
        self,
        kg: dgl.DGLHeteroGraph,
        hparams: ProtonConfig | dict,
    ):
        super().__init__()

        self.num_nodes = kg.num_nodes()
        self.num_ntypes = len(kg.ntypes)
        self.num_etypes = len(kg.canonical_etypes)

        self.cached_kg = None
        self.cached_degree_threshold = None

        total_edges = kg.num_edges()
        self.edge_type_weights = {
            edge_type: total_edges / (kg.num_edges(etype=edge_type) + 1) for edge_type in kg.canonical_etypes
        }

        if not isinstance(hparams, ProtonConfig):
            hparams = ProtonConfig(**hparams)

        self.save_hyperparameters(hparams.model_dump())
        self.num_feat = hparams.tunable_params.num_feat
        self.num_heads = hparams.tunable_params.num_heads
        self.hidden_dim = hparams.tunable_params.hidden_dim
        self.output_dim = hparams.tunable_params.output_dim
        self.num_layers = hparams.fixed_params.num_layers
        self.dropout_prob = hparams.tunable_params.dropout_prob
        self.pred_threshold = hparams.fixed_params.pred_threshold
        self.lr = hparams.tunable_params.lr
        self.wd = hparams.tunable_params.wd
        self.lr_factor = hparams.fixed_params.lr_factor
        self.lr_total_iters = hparams.fixed_params.lr_total_iters

        _logger.debug("Model hyperparameters:")
        _logger.debug(f"- num_feat: {self.num_feat}")
        _logger.debug(f"- num_heads: {self.num_heads}")
        _logger.debug(f"- hidden_dim: {self.hidden_dim}")
        _logger.debug(f"- output_dim: {self.output_dim}")
        _logger.debug(f"- num_layers: {self.num_layers}")
        _logger.debug(f"- dropout_prob: {self.dropout_prob}")
        _logger.debug(f"- pred_threshold: {self.pred_threshold}")

        _logger.debug("Learning rate parameters:")
        _logger.debug(f"- lr: {self.lr}")
        _logger.debug(f"- wd: {self.wd}")
        _logger.debug(f"- lr_factor: {self.lr_factor}")
        _logger.debug(f"- lr_total_iters: {self.lr_total_iters}")

        self.h_dim_1 = self.hidden_dim * 2
        self.h_dim_2 = self.hidden_dim

        self.emb = nn.Embedding(self.num_nodes, self.num_feat)

        self.conv1 = HGTConv(
            in_size=self.num_feat,
            head_size=self.h_dim_1,
            num_heads=self.num_heads,
            num_ntypes=self.num_ntypes,
            num_etypes=self.num_etypes,
            dropout=self.dropout_prob,
            use_norm=False,
        )

        self.norm1 = nn.LayerNorm(self.h_dim_1 * self.num_heads)

        if self.num_layers == 2:
            self.conv2 = HGTConv(
                in_size=self.h_dim_1 * self.num_heads,
                head_size=self.output_dim,
                num_heads=self.num_heads,
                num_ntypes=self.num_ntypes,
                num_etypes=self.num_etypes,
                dropout=self.dropout_prob,
                use_norm=True,
            )

        elif self.num_layers == 3:
            self.conv2 = HGTConv(
                in_size=self.h_dim_1 * self.num_heads,
                head_size=self.h_dim_2,
                num_heads=self.num_heads,
                num_ntypes=self.num_ntypes,
                num_etypes=self.num_etypes,
                dropout=self.dropout_prob,
                use_norm=False,
            )

            self.norm2 = nn.LayerNorm(self.h_dim_2 * self.num_heads)

            self.conv3 = HGTConv(
                in_size=self.h_dim_2 * self.num_heads,
                head_size=self.output_dim,
                num_heads=self.num_heads,
                num_ntypes=self.num_ntypes,
                num_etypes=self.num_etypes,
                dropout=self.dropout_prob,
                use_norm=True,
            )

        else:
            raise ValueError("Number of layers must be 2 or 3.")

        self.decoder = BilinearDecoder(self.num_etypes, self.output_dim * self.num_heads)

        self.val_step_metrics = []
        self.test_step_metrics = []

    def forward(self, subgraph):
        """
        This function performs a forward pass of the model. Note that the subgraph must be converted to from a
        heterogeneous graph to homogeneous graph for efficiency.

        Args:
            subgraph (dgl.DGLGraph): Subgraph containing the nodes and edges for the current batch.
        """

        global_node_indices = subgraph.ndata["node_index"]

        x = self.emb(global_node_indices)

        x = self.conv1(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])

        if self.num_layers == 3:
            x = self.norm2(x)
            x = F.leaky_relu(x)
            x = self.conv3(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])

        return x

    def _step(self, input_nodes, pos_graph, neg_graph, subgraph, mode):
        """Defines the step that is run on each batch of data. PyTorch Lightning handles steps including:
            - Moving data to the correct device.
            - Epoch and batch iteration.
            - optimizer.step(), loss.backward(), optimizer.zero_grad() calls.
            - Calling of model.eval(), enabling/disabling grads during evaluation.
            - Logging of metrics.

        Args:
            input_nodes (torch.Tensor): Input nodes.
            pos_graph (dgl.DGLHeteroGraph): Positive graph.
            neg_graph (dgl.DGLHeteroGraph): Negative graph.
            subgraph (dgl.DGLHeteroGraph): Subgraph.
            mode (str): The mode of the step (train, val, test).
        """
        batch_size = sum([x.shape[0] for x in input_nodes.values()])

        # Convert heterogeneous graph to homogeneous graph for efficiency
        # See https://docs.dgl.ai/en/latest/generated/dgl.to_homogeneous.html
        subgraph = dgl.to_homogeneous(subgraph, ndata=["node_index"])

        neg_graph.ndata["node_index"] = pos_graph.ndata["node_index"]

        node_embeddings = self.forward(subgraph)

        neg_graph.ndata["node_index"] = pos_graph.ndata["node_index"]

        pos_scores = self.decoder(subgraph, pos_graph, node_embeddings)
        neg_scores = self.decoder(subgraph, neg_graph, node_embeddings)

        loss, metrics, edge_type_metrics = self.compute_loss(pos_scores, neg_scores)

        return loss, metrics, edge_type_metrics, batch_size

    def training_step(self, batch, batch_idx):
        """Defines the step that is run on each batch of training data."""
        input_nodes, pos_graph, neg_graph, subgraph = batch

        loss, metrics, edge_type_metrics, batch_size = self._step(
            input_nodes, pos_graph, neg_graph, subgraph, mode="train"
        )

        values = {
            "train/loss": loss.detach(),
            "train/accuracy": metrics["accuracy"],
            "train/ap": metrics["ap"],
            "train/f1": metrics["f1"],
            "train/auroc": metrics["auroc"],
        }
        self.log_dict(values, batch_size=batch_size)

        for edge_type, metric in edge_type_metrics.items():
            edge_type_label = [label.replace("/", "_") for label in edge_type]
            edge_type_label = "-".join(edge_type_label)
            values = {
                f"edge_type_metrics/train/{edge_type_label}/accuracy": metric["accuracy"],
                f"edge_type_metrics/train/{edge_type_label}/ap": metric["ap"],
                f"edge_type_metrics/train/{edge_type_label}/f1": metric["f1"],
                f"edge_type_metrics/train/{edge_type_label}/auroc": metric["auroc"],
            }
            self.log_dict(values, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        """Defines the step that is run on each batch of validation data."""

        input_nodes, pos_graph, neg_graph, subgraph = batch

        loss, metrics, edge_type_metrics, batch_size = self._step(
            input_nodes, pos_graph, neg_graph, subgraph, mode="val"
        )

        values = {
            "val/loss": loss.detach(),
            "val/accuracy": metrics["accuracy"],
            "val/ap": metrics["ap"],
            "val/f1": metrics["f1"],
            "val/auroc": metrics["auroc"],
        }
        self.log_dict(values, batch_size=batch_size)

        for edge_type, metric in edge_type_metrics.items():
            edge_type_label = [label.replace("/", "_") for label in edge_type]
            edge_type_label = "-".join(edge_type_label)
            values = {
                f"edge_type_metrics/val/{edge_type_label}/accuracy": metric["accuracy"],
                f"edge_type_metrics/val/{edge_type_label}/ap": metric["ap"],
                f"edge_type_metrics/val/{edge_type_label}/f1": metric["f1"],
                f"edge_type_metrics/val/{edge_type_label}/auroc": metric["auroc"],
            }
            self.log_dict(values, batch_size=batch_size)

        self.val_step_metrics.append(metrics)

    def on_validation_epoch_end(self):
        """Defines the step that is called at the end of the validation epoch."""

        all_preds = np.concatenate([output["pred"] for output in self.val_step_metrics], axis=0)
        all_targets = np.concatenate([output["target"] for output in self.val_step_metrics], axis=0)

        binary_preds = np.zeros((all_preds.shape[0], 2))
        binary_preds[:, 0] = 1 - all_preds
        binary_preds[:, 1] = all_preds

        thresholded_preds = (all_preds > self.pred_threshold).astype(int)
        self._logger.experiment.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_targets,
                preds=thresholded_preds,
                class_names=["Negative", "Positive"],
            )
        })

        self._logger.experiment.log({
            "val/roc_curve": wandb.plot.roc_curve(
                y_true=all_targets,
                y_probas=binary_preds,
                labels=["Negative", "Positive"],
            )
        })

        self._logger.experiment.log({
            "val/pr_curve": wandb.plot.pr_curve(
                y_true=all_targets,
                y_probas=binary_preds,
                labels=["Negative", "Positive"],
            )
        })

        self.val_step_metrics.clear()

    def test_step(self, batch, batch_idx):
        """Defines the step that is run on each batch of test data."""
        input_nodes, pos_graph, neg_graph, subgraph = batch

        loss, metrics, edge_type_metrics, batch_size = self._step(
            input_nodes, pos_graph, neg_graph, subgraph, mode="test"
        )

        values = {
            "test/loss": loss.detach(),
            "test/accuracy": metrics["accuracy"],
            "test/ap": metrics["ap"],
            "test/f1": metrics["f1"],
            "test/auroc": metrics["auroc"],
        }
        self.log_dict(values, batch_size=batch_size)

        for edge_type, metric in edge_type_metrics.items():
            edge_type_label = [label.replace("/", "_") for label in edge_type]
            edge_type_label = "-".join(edge_type_label)
            values = {
                f"edge_type_metrics/test/{edge_type_label}/accuracy": metric["accuracy"],
                f"edge_type_metrics/test/{edge_type_label}/ap": metric["ap"],
                f"edge_type_metrics/test/{edge_type_label}/f1": metric["f1"],
                f"edge_type_metrics/test/{edge_type_label}/auroc": metric["auroc"],
            }
            self.log_dict(values, batch_size=batch_size)

        self.test_step_metrics.append(metrics)

    def on_test_epoch_end(self):
        """Defines the step that is called at the end of the test epoch."""

        all_preds = np.concatenate([output["pred"] for output in self.test_step_metrics], axis=0)
        all_targets = np.concatenate([output["target"] for output in self.test_step_metrics], axis=0)

        binary_preds = np.zeros((all_preds.shape[0], 2))
        binary_preds[:, 0] = 1 - all_preds
        binary_preds[:, 1] = all_preds

        thresholded_preds = (all_preds > self.pred_threshold).astype(int)
        self._logger.experiment.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_targets,
                preds=thresholded_preds,
                class_names=["Negative", "Positive"],
            )
        })

        self._logger.experiment.log({
            "test/roc_curve": wandb.plot.roc_curve(
                y_true=all_targets,
                y_probas=binary_preds,
                labels=["Negative", "Positive"],
            )
        })

        self._logger.experiment.log({
            "test/pr_curve": wandb.plot.pr_curve(
                y_true=all_targets,
                y_probas=binary_preds,
                labels=["Negative", "Positive"],
            )
        })

        self.test_step_metrics.clear()

    def compute_loss(self, pos_scores, neg_scores):
        """
        This function computes the loss and metrics for the current batch.
        """

        pos_pred = torch.cat(list(pos_scores.values()))
        neg_pred = torch.cat(list(neg_scores.values()))
        raw_pred = torch.cat((pos_pred, neg_pred))

        pred = raw_pred

        pos_target = torch.ones(pos_pred.shape[0])
        neg_target = torch.zeros(neg_pred.shape[0])
        target = torch.cat((pos_target, neg_target)).to(self.device)  # .to(device)

        pos_edge_types = [[edge_type] * scores.shape[0] for edge_type, scores in pos_scores.items()]
        pos_edge_types = [item for sublist in pos_edge_types for item in sublist]
        neg_edge_types = [[edge_type] * scores.shape[0] for edge_type, scores in neg_scores.items()]
        neg_edge_types = [item for sublist in neg_edge_types for item in sublist]
        edge_types = pos_edge_types + neg_edge_types

        weights = [self.edge_type_weights[etype] for etype in edge_types]
        weights = torch.tensor([weight / sum(weights) for weight in weights]).to(self.device)

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        loss = torch.dot(loss, weights)

        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        metrics = calculate_metrics(pred, target, self.pred_threshold)

        edge_type_metrics = {}
        for edge_type in set(edge_types):
            mask = torch.tensor([item == edge_type for item in edge_types]).cpu().detach().numpy()

            if mask.sum() == 0:
                continue

            edge_type_metrics[edge_type] = calculate_metrics(pred[mask], target[mask], self.pred_threshold)

        return loss, metrics, edge_type_metrics

    def configure_optimizers(self):
        """
        This function is called by PyTorch Lightning to get the optimizer and scheduler.
        We reduce the learning rate by a factor of lr_factor if the validation loss does not improve for lr_patience epochs.

        Returns:
            dict: Dictionary containing the optimizer and scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=self.lr_factor, total_iters=self.lr_total_iters
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "name": "curr_lr",
            },
        }

    def cache_graph(self, kg, overwrite=False, degree_threshold=None):
        """
        This function caches the knowledge graph or a subset thereof.

        Args:
            kg (dgl.DGLHeteroGraph): Knowledge graph.
            overwrite (bool): Whether to overwrite the cache.
            degree_threshold (int): Maximum degree of nodes to cache.
        """

        if self.cached_kg is not None and not overwrite:
            raise ValueError("Cached knowledge graph already exists. Set overwrite = True to overwrite.")

        if degree_threshold is not None:
            degree_by_etype = {etype: kg.out_degrees(etype=etype) for etype in kg.canonical_etypes}

            degrees = {}
            for (src_type, _rel_type, _dst_type), degree in degree_by_etype.items():
                if src_type not in degrees:
                    degrees[src_type] = degree
                else:
                    degrees[src_type] += degree

            degree_mask = {ntype: degree <= degree_threshold for ntype, degree in degrees.items()}
            self.cached_kg = dgl.node_subgraph(kg, degree_mask, relabel_nodes=False)
            self.cached_degree_threshold = degree_threshold

        else:
            self.cached_kg = kg

    def clear_graph_cache(self):
        """
        This function clears the cached knowledge graph.
        """
        self.cached_kg = None
        self.cached_degree_threshold = None

    def subsample_graph(self, query_kg, use_cache=True, degree_threshold=None, fixed_k=None):
        """
        This function subsamples the knowledge graph to prevent OOM at inference time. Three strategies are available:
            1. Use cached knowledge graph. Run cache_graph() first, then set use_cache = True.
            2. Remove nodes with high degree, then sample all nodes. This strategy is used by cache_graph() if
                degree_threshold is provided.
            3. Use fixed number of neighbors within sampler.

        Args:
            query_kg (dgl.DGLHeteroGraph): Query graph.
            use_cache (bool): Whether to use cached knowledge graph. Note, if degree threshold is given and use_cache = True,
                then self.cache_graph() must be called first and self.cached_degree_threshold must = degree_threshold.
            degree_threshold (int): Maximum degree of nodes to include in MFG.
            fixed_k (int): Fixed number of neighbors to sample INSTEAD of using degree threshold.

        Returns:
            query_kg (dgl.DGLHeteroGraph): Subsampled query graph.
            query_sampler (dgl.dataloading.Sampler): Node sampler.
        """

        if query_kg is None and not use_cache:
            raise ValueError("Either query_kg must be provided or use_cache must be True.")

        elif use_cache:
            if self.cached_kg is None:
                raise ValueError("Cached knowledge graph does not exist. Call cache_graph() first.")
            if degree_threshold is not None and self.cached_degree_threshold != degree_threshold:
                raise ValueError("Cached degree threshold does not match provided degree threshold.")

            query_kg = self.cached_kg
            query_sampler = dgl.dataloading.ShaDowKHopSampler([-1, -1, -1])

        elif degree_threshold is not None:
            degree_by_etype = {etype: query_kg.out_degrees(etype=etype) for etype in query_kg.canonical_etypes}

            degrees = {}
            for (src_type, _, _), degree in degree_by_etype.items():
                if src_type not in degrees:
                    degrees[src_type] = degree
                else:
                    degrees[src_type] += degree

            degree_mask = {ntype: degree <= degree_threshold for ntype, degree in degrees.items()}
            query_kg = dgl.node_subgraph(query_kg, degree_mask, relabel_nodes=False)
            query_sampler = dgl.dataloading.ShaDowKHopSampler([-1, -1, -1])

        elif fixed_k is not None:
            query_sampler = FixedSampler(
                conf.proton.fixed_params.sampler_fanout,
                fixed_k,
                upsample_rare_types=False,
            )

        else:
            raise ValueError("Either use_cache or degree_threshold or fixed_k must be provided.")

        return query_kg, query_sampler

    @torch.no_grad()
    def get_embeddings(
        self,
        query_indices,
        query_kg=None,
        use_cache=True,
        degree_threshold=None,
        fixed_k=None,
    ):
        """
        This function returns the node embeddings for a set of global query indices.

        Args:
            query_indices (list): List of global (not reindexed!) node indices.
            hparams (dict): Dictionary of model hyperparameters.
            query_kg (dgl.DGLHeteroGraph): Query graph.
            use_cache (bool): See subsample_graph().
            degree_threshold (int): See subsample_graph().
            fixed_k (int): See subsample_graph().
        """

        query_kg, query_sampler = self.subsample_graph(query_kg, use_cache, degree_threshold, fixed_k)

        query_indices = torch.tensor(query_indices).unsqueeze(1)

        kg_indices = query_kg.ndata["node_index"]
        query_nodes = {key: torch.where(value == query_indices)[1] for key, value in kg_indices.items()}

        # Sample subgraph, and convert heterogeneous graph to homogeneous graph for efficiency
        _, _, query_subgraph = query_sampler.sample(query_kg, query_nodes)
        query_subgraph = dgl.to_homogeneous(query_subgraph, ndata=["node_index"])

        node_embeddings = self.forward(query_subgraph)

        query_subgraph_nodes = query_subgraph.ndata["node_index"]
        query_subgraph_index = torch.where(query_subgraph_nodes == query_indices)[1]

        return node_embeddings[query_subgraph_index]

    @torch.no_grad()
    def get_scores(
        self,
        src_indices,
        dst_indices,
        query_edge_type,
        hparams,
        query_kg=None,
        use_cache=True,
        degree_threshold=None,
        fixed_k=None,
    ):
        """
        For a set of edges described by paired source nodes and destination nodes, this function
        computes the likelihood score of each edge. Note that `src_indices` must be valid global
        node IDs of type `query_edge_type[0]` (where the global node IDs are stored as a node
        attribute in `query_kg.ndata['node_index']`, and `dst_indices` must be valid global node
        IDs of type `query_edge_type[2]`.

        Args:
            src_indices (list): List of global (not reindexed!) source node indices.
            dst_indices (list): List of global (not reindexed!) destination node indices.
            query_edge_type (tuple): Edge type of query edges.
            hparams (dict): Dictionary of model hyperparameters.
            query_kg (dgl.DGLHeteroGraph): Query graph.
            use_cache (bool): See subsample_graph().
            degree_threshold (int): See subsample_graph().
            fixed_k (int): See subsample_graph().
        """

        query_kg, query_sampler = self.subsample_graph(query_kg, use_cache, degree_threshold, fixed_k)

        query_indices = torch.tensor(src_indices + dst_indices)  # .to(device)
        query_indices = torch.unique(query_indices).unsqueeze(1)

        if query_edge_type not in query_kg.canonical_etypes:
            raise ValueError("Edge type not in knowledge graph.")
        src_type = query_edge_type[0]
        dst_type = query_edge_type[2]

        kg_indices = query_kg.ndata["node_index"]
        query_nodes = {key: torch.where(value == query_indices)[1] for key, value in kg_indices.items()}

        if src_type != dst_type:
            src_set = list(set(src_indices))
            src_srt = list(range(len(src_set)))
            src_map = dict(zip(src_set, src_srt, strict=False))
            src_nodes = torch.tensor([src_map[x] for x in src_indices])

            dst_set = list(set(dst_indices))
            dst_srt = list(range(len(dst_set)))
            dst_map = dict(zip(dst_set, dst_srt, strict=False))
            dst_nodes = torch.tensor([dst_map[x] for x in dst_indices])

        else:
            src_dst_set = list(set(src_indices + dst_indices))
            src_dst_srt = list(range(len(src_dst_set)))
            src_dst_map = dict(zip(src_dst_set, src_dst_srt, strict=False))
            src_nodes = torch.tensor([src_dst_map[x] for x in src_indices])
            dst_nodes = torch.tensor([src_dst_map[x] for x in dst_indices])

        edge_graph_data = {etype: ([], []) for etype in query_kg.canonical_etypes}
        edge_graph_data[query_edge_type] = (src_nodes, dst_nodes)
        query_edge_graph = dgl.heterograph(edge_graph_data)
        assert query_kg.ntypes == query_edge_graph.ntypes
        assert query_kg.canonical_etypes == query_edge_graph.canonical_etypes

        if src_type != dst_type:
            src_map_rev = {y: x for x, y in src_map.items()}
            dst_map_rev = {y: x for x, y in dst_map.items()}
            global_src_nodes = torch.tensor([src_map_rev[x.item()] for x in query_edge_graph.nodes(src_type)])
            global_dst_nodes = torch.tensor([dst_map_rev[x.item()] for x in query_edge_graph.nodes(dst_type)])

            node_index_data = {ntype: torch.empty(0) for ntype in query_kg.ntypes}
            node_index_data[src_type] = global_src_nodes
            node_index_data[dst_type] = global_dst_nodes
            query_edge_graph.ndata["node_index"] = node_index_data

        else:
            src_dst_map_rev = {y: x for x, y in src_dst_map.items()}
            global_src_dst_nodes = torch.tensor([src_dst_map_rev[x.item()] for x in query_edge_graph.nodes(src_type)])

            node_index_data = {ntype: torch.empty(0) for ntype in query_kg.ntypes}
            node_index_data[src_type] = global_src_dst_nodes
            query_edge_graph.ndata["node_index"] = node_index_data

        _, _, query_subgraph = query_sampler.sample(query_kg, query_nodes)

        query_subgraph = dgl.to_homogeneous(query_subgraph, ndata=["node_index"])

        # Get node embeddings
        node_embeddings = self.forward(query_subgraph)

        scores = self.decoder(query_subgraph, query_edge_graph, node_embeddings)
        scores = torch.sigmoid(scores[query_edge_type])

        return scores

    @torch.no_grad()
    def get_scores_from_embeddings(
        self,
        src_indices,
        dst_indices,
        query_edge_type,
        query_kg=None,
        use_cache=True,
        embeddings=None,
        decoder=None,
    ):
        """
        For a set of edges described by paired source nodes and destination nodes, this function
        computes the likelihood score of each edge. Note that `src_indices` must be valid global
        node IDs of type `query_edge_type[0]` (where the global node IDs are stored as a node
        attribute in `query_kg.ndata['node_index']`, and `dst_indices` must be valid global node
        IDs of type `query_edge_type[2]`.

        This function differs from get_scores() in that it uses cached embeddings, and is therefore much faster.

        Args:
            src_indices (list): List of global (not reindexed!) source node indices.
            dst_indices (list): List of global (not reindexed!) destination node indices.
            query_edge_type (tuple): Edge type of query edges.
            query_kg (dgl.DGLHeteroGraph): Query graph.
            embeddings (torch.Tensor): Node embeddings saved by save_embeddings() in pretrain.py. If not provided, the
                embeddings are read from disk based on the values of conf.paths.checkpoint.embeddings_path.
        """

        if query_kg is None and not use_cache:
            raise ValueError("Either query_kg must be provided or use_cache must be True.")
        elif use_cache:
            if self.cached_kg is None:
                raise ValueError("Cached knowledge graph does not exist. Call cache_graph() first.")
            query_kg = self.cached_kg

        if embeddings is None:
            embeddings = torch.load(conf.paths.checkpoint.embeddings_path)

        src_embeddings = embeddings[src_indices]
        dst_embeddings = embeddings[dst_indices]

        src_embeddings = F.leaky_relu(src_embeddings)
        dst_embeddings = F.leaky_relu(dst_embeddings)

        edge_type_index = [i for i, etype in enumerate(query_kg.canonical_etypes) if etype == query_edge_type]
        if len(edge_type_index) == 0:
            raise ValueError(
                f"Edge type ({query_edge_type[0]}, {query_edge_type[1]}, {query_edge_type[2]}) not found in knowledge graph."
            )
        else:
            edge_type_index = edge_type_index[0]

        if decoder is None:
            decoder = self.decoder.relation_weights

        scores = torch.sum(
            src_embeddings.to(self.device) * decoder[edge_type_index].to(self.device) * dst_embeddings.to(self.device),
            dim=1,
        )
        scores = torch.sigmoid(scores)
        return scores
