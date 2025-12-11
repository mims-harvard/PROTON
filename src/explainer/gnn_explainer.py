import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch import nn
from tqdm import tqdm

from src.constants import TORCH_DEVICE

from .utils.general import get_node_name, get_original_score

_logger = logging.getLogger(__name__)


class GNNExplainer:
    def __init__(
        self,
        model,
        original_score,
        sg,
        query_edge_graph,
        query_edge_type,
        src_name,
        dst_name,
        explainer_hparams,
        kg,
        nodes,
        edges,
        save_path,
        relation,
    ):
        self.model = model
        self.original_score = original_score

        self.sg = sg.to(TORCH_DEVICE)
        self.sg_cpu = sg.cpu()

        self.query_edge_graph = query_edge_graph.to(TORCH_DEVICE)
        self.query_edge_graph_cpu = query_edge_graph.cpu()

        self.query_edge_type = query_edge_type

        self.src_name = src_name
        self.dst_name = dst_name
        self.explainer_hparams = explainer_hparams
        self.kg = kg
        self.nodes = nodes
        self.edges = edges
        self.save_path = save_path
        self.relation = relation

        self.relation_save_path = save_path / self.relation
        self.relation_save_path.mkdir(parents=True, exist_ok=True)

        self.mask = None
        self.mask_cpu = None

    def compute_score_at_percentile(self):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        shuffled_edge_mask = self.mask_cpu.numpy().copy()
        np.random.shuffle(shuffled_edge_mask)
        shuffled_edge_mask = torch.tensor(shuffled_edge_mask, device=TORCH_DEVICE)

        percentiles = np.arange(0, 101, 5)
        remove_less_to_most_important_scores = {}
        remove_most_to_least_important_scores = {}

        for per in percentiles:
            mask_thresh = np.percentile(self.mask_cpu, per)

            mask_to_remove_least_important_edges = (self.mask > mask_thresh).float()
            num_edges_removed = (
                mask_to_remove_least_important_edges.numel() - mask_to_remove_least_important_edges.sum().item()
            )
            _logger.info(f"Percentile: {per}, Threshold: {mask_thresh:.4f}, Edges removed: {num_edges_removed}")

            mask_to_remove_random_edges = (shuffled_edge_mask > mask_thresh).float()

            with torch.no_grad():
                embeddings_when_remove_least_important_edges = self.model.forward(
                    self.sg, eweight=mask_to_remove_least_important_edges
                )
                score_when_remove_least_important_edges = self.model.decoder(
                    self.sg,
                    self.query_edge_graph,
                    embeddings_when_remove_least_important_edges,
                )
                score_when_remove_least_important_edges = torch.sigmoid(
                    score_when_remove_least_important_edges[self.query_edge_type]
                )

                embeddings_when_remove_random_edges = self.model.forward(self.sg, eweight=mask_to_remove_random_edges)
                score_when_remove_random_edges = self.model.decoder(
                    self.sg, self.query_edge_graph, embeddings_when_remove_random_edges
                )
                score_when_remove_random_edges = torch.sigmoid(score_when_remove_random_edges[self.query_edge_type])

            remove_less_to_most_important_scores[per] = {
                "num_edges_removed": num_edges_removed,
                "score_when_remove_least_important_edges": score_when_remove_least_important_edges.cpu().numpy()[0],
                "score_when_remove_random_edges": score_when_remove_random_edges.cpu().numpy()[0],
            }

            mask_to_remove_most_important_edges = (self.mask < mask_thresh).float()
            num_edges_removed = (
                mask_to_remove_most_important_edges.numel() - mask_to_remove_most_important_edges.sum().item()
            )
            _logger.info(f"Percentile: {per}, Threshold: {mask_thresh:.4f}, Edges removed: {num_edges_removed}")

            mask_to_remove_random_edges = (shuffled_edge_mask < mask_thresh).float()

            with torch.no_grad():
                embeddings_when_remove_most_important_edges = self.model.forward(
                    self.sg, eweight=mask_to_remove_most_important_edges
                )
                score_when_remove_most_important_edges = self.model.decoder(
                    self.sg,
                    self.query_edge_graph,
                    embeddings_when_remove_most_important_edges,
                )
                score_when_remove_most_important_edges = torch.sigmoid(
                    score_when_remove_most_important_edges[self.query_edge_type]
                )

                embeddings_when_remove_random_edges = self.model.forward(self.sg, eweight=mask_to_remove_random_edges)
                score_when_remove_random_edges = self.model.decoder(
                    self.sg, self.query_edge_graph, embeddings_when_remove_random_edges
                )
                score_when_remove_random_edges = torch.sigmoid(score_when_remove_random_edges[self.query_edge_type])

            remove_most_to_least_important_scores[
                100 - per
            ] = {  # NOTE: 100-per because we are removing most important edges
                "num_edges_removed": num_edges_removed,
                "score_when_remove_most_important_edges": score_when_remove_most_important_edges.cpu().numpy()[0],
                "score_when_remove_random_edges": score_when_remove_random_edges.cpu().numpy()[0],
            }

        remove_less_to_most_important_df = pd.DataFrame(remove_less_to_most_important_scores).T
        remove_less_to_most_important_df = remove_less_to_most_important_df.reset_index()
        remove_less_to_most_important_df = remove_less_to_most_important_df.rename(columns={"index": "percentile"})

        remove_most_to_least_important_df = pd.DataFrame(remove_most_to_least_important_scores).T
        remove_most_to_least_important_df = remove_most_to_least_important_df.reset_index()
        remove_most_to_least_important_df = remove_most_to_least_important_df.rename(columns={"index": "percentile"})

        return remove_less_to_most_important_df, remove_most_to_least_important_df

    def save_important_edges_df(self, top_k=500):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        k = min(top_k, len(self.mask_cpu))
        important_edges_scores, important_edges_indices = torch.topk(self.mask_cpu, k)

        important_edges_ids = self.sg_cpu.edges(form="eid")[important_edges_indices]
        important_edges = self.sg_cpu.find_edges(important_edges_ids)

        important_edges_src = self.sg_cpu.ndata["node_index"][important_edges[0]]
        impt_src_names = [get_node_name(self.nodes, int(i)) for i in important_edges_src]

        important_edges_dst = self.sg_cpu.ndata["node_index"][important_edges[1]]
        impt_dst_names = [get_node_name(self.nodes, int(i)) for i in important_edges_dst]

        important_edge_types = self.sg_cpu.edata["_TYPE"][important_edges_indices]
        mapped_relations = [self.kg.canonical_etypes[r] for r in important_edge_types]

        important_edges_df = pd.DataFrame({
            "x_index": important_edges_src,
            "x_name": impt_src_names,
            "relation": [r[1] for r in mapped_relations],
            "y_index": important_edges_dst,
            "y_name": impt_dst_names,
            "x_type": [r[0] for r in mapped_relations],
            "y_type": [r[2] for r in mapped_relations],
            "score": important_edges_scores,
            "rank": range(1, len(important_edges_scores) + 1),
        })

        important_edges_df = pd.merge(
            important_edges_df,
            self.edges,
            on=["x_index", "relation", "y_index"],
            suffixes=("", "_impt"),
            sort=False,
        )

        wandb.log({f"important_edges ({self.relation})": wandb.Table(dataframe=important_edges_df)})
        important_edges_df.to_csv(self.relation_save_path / "edge_importance.csv", index=False)

    def save_mask_distribution_plot(self, top_k=500):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        sorted_mask_values, _ = torch.sort(self.mask_cpu, descending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            range(len(sorted_mask_values)),
            sorted_mask_values,
            s=10,
            color="b",
            label="Mask values (sorted, high→low)",
        )

        axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)
        top_values = sorted_mask_values[:top_k]
        top_indices = range(top_k)
        axins.scatter(top_indices, top_values, s=20, color="r")
        axins.set_title(f"Top {top_k} mask values for {self.relation}")
        axins.set_xlabel("Index")
        axins.set_ylabel("Mask value")

        axins.set_xlim(-1, top_k)
        axins.set_ylim(top_values.min() * 0.98, top_values.max() * 1.02)

        ax.set_ylabel("Mask value")

        ax.set_title(f"Sorted Edge Mask Values for {self.src_name} → {self.dst_name} ({self.relation})")
        plt.tight_layout()

        file_path = self.relation_save_path / "mask_distribution.png"
        plt.savefig(file_path)
        wandb.log({f"mask_distribution_plot ({self.relation})": wandb.Image(str(file_path))})

        plt.close(fig)

    def save_scores_least_to_most_important_plots(self, remove_less_to_most_important_df):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            remove_less_to_most_important_df["percentile"],
            remove_less_to_most_important_df["score_when_remove_least_important_edges"],
            label=f"Score without $k$ edges (least to most important) for {self.relation}",
            marker="o",
            color="b",
            zorder=3,
        )
        ax1.plot(
            remove_less_to_most_important_df["percentile"],
            remove_less_to_most_important_df["score_when_remove_random_edges"],
            label=f"Score without $k$ random edges for {self.relation}",
            marker="o",
            color="r",
            zorder=2,
        )
        ax1.axhline(y=self.original_score, color="g", linestyle="--", label="Original score")
        ax1.set_xlabel("Percent of Edges Removed (%)", color="black", fontweight="bold")
        ax1.set_ylabel("Score", color="black", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(remove_less_to_most_important_df["percentile"])
        ax2.set_xticklabels(
            [str(x) for x in remove_less_to_most_important_df["num_edges_removed"]],
            color="black",
        )

        ax2.set_xlabel("Number of Edges Removed ($k$)", color="black", fontweight="bold")
        ax2.tick_params(axis="x", labelcolor="black")

        plt.title(f"{self.src_name} → {self.dst_name} ({self.relation})", fontweight="bold")
        ax1.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        file_path = self.relation_save_path / "scores_without_least_to_most_important_edges.png"
        plt.savefig(file_path)
        wandb.log({f"scores_without_least_to_most_important_edges ({self.relation})": wandb.Image(str(file_path))})

        plt.close(fig)

    def save_scores_most_to_least_important_plots(self, remove_most_to_least_important_df):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            remove_most_to_least_important_df["percentile"],
            remove_most_to_least_important_df["score_when_remove_most_important_edges"],
            label=f"Score without $k$ edges (most to least important) for {self.relation}",
            marker="o",
            color="b",
            zorder=3,
        )
        ax1.plot(
            remove_most_to_least_important_df["percentile"],
            remove_most_to_least_important_df["score_when_remove_random_edges"],
            label=f"Score without $k$ random edges for {self.relation}",
            marker="o",
            color="r",
            zorder=2,
        )
        ax1.axhline(y=self.original_score, color="g", linestyle="--", label="Original score")
        ax1.set_xlabel("Percent of Edges Removed (%)", color="black", fontweight="bold")
        ax1.set_ylabel("Score", color="black", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(remove_most_to_least_important_df["percentile"])
        ax2.set_xticklabels(
            [str(x) for x in remove_most_to_least_important_df["num_edges_removed"]],
            color="black",
        )

        ax2.set_xlabel("Number of Edges Removed ($k$)", color="black", fontweight="bold")
        ax2.tick_params(axis="x", labelcolor="black")

        # Title and grid
        plt.title(f"{self.src_name} → {self.dst_name} ({self.relation})", fontweight="bold")
        ax1.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        file_path = self.relation_save_path / "scores_without_most_to_least_important_edges.png"
        plt.savefig(file_path)
        wandb.log({f"scores_without_most_to_least_important_edges ({self.relation})": wandb.Image(str(file_path))})

        plt.close(fig)

    def save_sufficiency_based_faithfulness_plot(self, remove_less_to_most_important_df):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        # Compute faithfulness and scrambled_faithfulness
        remove_less_to_most_important_df["faithfulness"] = (
            remove_less_to_most_important_df["score_when_remove_least_important_edges"] - self.original_score
        ) ** 2
        remove_less_to_most_important_df["scrambled_faithfulness"] = (
            remove_less_to_most_important_df["score_when_remove_random_edges"] - self.original_score
        ) ** 2

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot faithfulness vs. percentile
        ax1.plot(
            remove_less_to_most_important_df["percentile"],
            remove_less_to_most_important_df["faithfulness"],
            label=f"Faithfulness without $k$ edges (least to most important) for {self.relation}",
            marker="o",
            color="b",
            zorder=3,
        )
        ax1.plot(
            remove_less_to_most_important_df["percentile"],
            remove_less_to_most_important_df["scrambled_faithfulness"],
            label=f"Faithfulness without $k$ random edges for {self.relation}",
            marker="o",
            color="r",
            zorder=2,
        )
        ax1.set_xlabel("Percent of Edges Removed (%)", color="black", fontweight="bold")
        ax1.set_ylabel(
            "Unfaithfulness Loss ($[f(S) - f(S_k)]^2$)",
            color="black",
            fontweight="bold",
        )
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        ax2.set_xticks(remove_less_to_most_important_df["percentile"])
        ax2.set_xticklabels(
            [str(x) for x in remove_less_to_most_important_df["num_edges_removed"]],
            color="black",
        )
        ax2.set_xlabel("Number of Edges Removed ($k$)", color="black", fontweight="bold")
        ax2.tick_params(axis="x", labelcolor="black")

        plt.title(f"{self.src_name} → {self.dst_name} ({self.relation})", fontweight="bold")
        ax1.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        file_path = self.relation_save_path / "sufficiency_based_faithfulness_plot.png"
        plt.savefig(file_path)
        wandb.log({f"sufficiency_based_faithfulness_plot ({self.relation})": wandb.Image(str(file_path))})

        plt.close(fig)

    def save_necessity_based_faithfulness_plot(self, remove_most_to_least_important_df):
        assert self.mask is not None, "Mask is not trained yet"
        assert self.mask_cpu is not None, "Mask is not trained yet"

        remove_most_to_least_important_df["faithfulness"] = (
            remove_most_to_least_important_df["score_when_remove_most_important_edges"] - self.original_score
        ) ** 2
        remove_most_to_least_important_df["scrambled_faithfulness"] = (
            remove_most_to_least_important_df["score_when_remove_random_edges"] - self.original_score
        ) ** 2

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            remove_most_to_least_important_df["percentile"],
            remove_most_to_least_important_df["faithfulness"],
            label=f"Faithfulness without $k$ edges (least to most important) for {self.relation}",
            marker="o",
            color="b",
            zorder=3,
        )
        ax1.plot(
            remove_most_to_least_important_df["percentile"],
            remove_most_to_least_important_df["scrambled_faithfulness"],
            label=f"Faithfulness without $k$ random edges for {self.relation}",
            marker="o",
            color="r",
            zorder=2,
        )
        ax1.set_xlabel("Percent of Edges Removed (%)", color="black", fontweight="bold")
        ax1.set_ylabel(
            "Unfaithfulness Loss ($[f(S) - f(S_k)]^2$)",
            color="black",
            fontweight="bold",
        )
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        ax2.set_xticks(remove_most_to_least_important_df["percentile"])
        ax2.set_xticklabels(
            [str(x) for x in remove_most_to_least_important_df["num_edges_removed"]],
            color="black",
        )
        ax2.set_xlabel("Number of Edges Removed ($k$)", color="black", fontweight="bold")
        ax2.tick_params(axis="x", labelcolor="black")

        plt.title(f"{self.src_name} → {self.dst_name} ({self.relation})", fontweight="bold")
        ax1.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        file_path = self.relation_save_path / "necessity_based_faithfulness_plot.png"
        plt.savefig(file_path)
        wandb.log({f"necessity_based_faithfulness_plot ({self.relation})": wandb.Image(str(file_path))})

        plt.close(fig)

    def save_distr_plot_at_epoch(self, mask_at_epoch, epoch, top_k=500):
        fig, ax = plt.subplots(figsize=(8, 5))
        mask_np = mask_at_epoch.cpu().numpy()
        ax.hist(
            mask_np,
            bins=200,
            color="skyblue",
            edgecolor="black",
            label="Full Distribution",
        )
        ax.set_title(
            f"Edge weight distribution at epoch {epoch}, {self.src_name} → {self.dst_name} ({self.relation})",
            fontweight="bold",
        )
        ax.set_xlabel("Edge weight", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")

        axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)

        sorted_mask_values = np.sort(mask_np)[::-1]  # Sort descending
        top_values = sorted_mask_values[:top_k]

        axins.hist(top_values, bins=(top_k // 10), color="coral", edgecolor="black")
        axins.set_title(f"Top {top_k} mask values (epoch {epoch})")
        axins.set_xlabel("Edge weight")
        axins.set_ylabel("Frequency")

        if len(top_values) > 0:
            axins.set_xlim(top_values.min() * 0.98, top_values.max() * 1.02)

        ax.legend(loc="upper left")
        plt.tight_layout()

        file_path = self.relation_save_path / f"mask_distribution_epoch_{epoch}.png"
        plt.savefig(file_path)

        wandb.log({f"mask_distribution_epoch_{epoch} ({self.relation})": wandb.Image(str(file_path))})
        plt.close(fig)

    @staticmethod
    def calculate_entropy_loss(entropy_loss_alpha, edge_mask):
        eps = 1e-15
        edge_mask = edge_mask.sigmoid()
        ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        return entropy_loss_alpha * ent.mean()

    @staticmethod
    def calculate_sparsity_loss(sparsity_loss_alpha, edge_mask):
        return sparsity_loss_alpha * torch.sum(edge_mask.sigmoid())

    def train_gnnexplainer(
        self,
    ):
        wandb.define_metric(
            step_metric=f"explainer_epoch ({self.relation})",
            name=f"cross_entropy_loss_forward ({self.relation})",
        )
        wandb.define_metric(
            step_metric=f"explainer_epoch ({self.relation})",
            name=f"regularization_loss ({self.relation})",
        )
        wandb.define_metric(
            step_metric=f"explainer_epoch ({self.relation})",
            name=f"total_loss ({self.relation})",
            step_sync=True,
        )

        lr = self.explainer_hparams.lr
        num_epochs = self.explainer_hparams.num_epochs
        sparsity_loss_alpha = self.explainer_hparams.sparsity_loss_alpha
        entropy_loss_alpha = self.explainer_hparams.entropy_loss_alpha

        loss_target_scores = get_original_score(  # TODO: Replace this by self.original_score
            self.model, self.sg, self.query_edge_graph, self.query_edge_type
        )
        if self.relation == "positive":
            loss_target_scores = torch.ones_like(loss_target_scores)
        elif self.relation == "negative":
            loss_target_scores = torch.zeros_like(loss_target_scores)

        edge_mask = nn.Parameter(torch.empty(self.sg.num_edges(), device=TORCH_DEVICE))
        nn.init.xavier_normal_(edge_mask.unsqueeze(1))

        optimizer = torch.optim.Adam([edge_mask], lr=lr)

        pbar = tqdm(total=num_epochs)
        pbar.set_description(f"Explain edge {self.src_name} → {self.dst_name} ({self.relation})")

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            masked_node_embeddings = self.model.forward(self.sg, eweight=edge_mask.sigmoid())
            masked_scores = self.model.decoder(self.sg, self.query_edge_graph, masked_node_embeddings)
            masked_scores = torch.sigmoid(masked_scores[self.query_edge_type])

            cross_entropy_loss_forward = F.binary_cross_entropy(masked_scores, loss_target_scores)

            total_cross_entropy_loss = cross_entropy_loss_forward

            entropy_loss = self.calculate_entropy_loss(entropy_loss_alpha, edge_mask)
            sparsity_loss = self.calculate_sparsity_loss(sparsity_loss_alpha, edge_mask)

            loss = total_cross_entropy_loss + entropy_loss + sparsity_loss

            wandb.log(
                {
                    f"masked_score_forward ({self.relation})": masked_scores.item(),
                    f"total_cross_entropy_loss ({self.relation})": total_cross_entropy_loss.item(),
                    f"cross_entropy_loss_forward ({self.relation})": cross_entropy_loss_forward.item(),
                    f"entropy_loss ({self.relation})": entropy_loss.item(),
                    f"sparsity_loss ({self.relation})": sparsity_loss.item(),
                    f"total_loss ({self.relation})": loss.item(),
                    f"explainer_epoch ({self.relation})": epoch,
                    f"entropy_loss_alpha ({self.relation})": entropy_loss_alpha,
                    f"sparsity_loss_alpha ({self.relation})": sparsity_loss_alpha,
                },
            )

            loss.backward()
            optimizer.step()

            if (epoch + 1) % (num_epochs // 5) == 0:
                self.save_distr_plot_at_epoch(edge_mask.detach().sigmoid(), epoch)

            pbar.update(1)
        pbar.close()

        edge_mask = edge_mask.detach().sigmoid()
        self.mask = edge_mask
        self.mask_cpu = edge_mask.cpu()

    def explain(self):
        torch.cuda.empty_cache()
        self.train_gnnexplainer()

        self.save_important_edges_df(top_k=30_000)
        self.save_mask_distribution_plot()
        remove_less_to_most_important_df, remove_most_to_least_important_df = self.compute_score_at_percentile()

        self.save_scores_least_to_most_important_plots(remove_less_to_most_important_df)
        self.save_scores_most_to_least_important_plots(remove_most_to_least_important_df)
        self.save_sufficiency_based_faithfulness_plot(remove_less_to_most_important_df)
        self.save_necessity_based_faithfulness_plot(remove_most_to_least_important_df)
