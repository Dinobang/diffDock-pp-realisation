import itertools

import plotly.graph_objects as go
import torch
from torch_geometric.data import Data
import random


class PlotlyVis:
    @classmethod
    def create_figure(
        cls, graph: Data, figsize: tuple[int, int] = (900, 600)
    ) -> go.Figure:
        traces = cls.plot_graph(graph)
        width, height = figsize
        figure = go.Figure(
            data=traces,
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                showlegend=False,
                width=width,
                height=height,
            ),
        )
        return figure

    @classmethod
    def plot_graph(
        cls,
        graph: Data,
    ) -> go.Figure:
        assert graph.pos is not None
        assert graph.edge_index is not None
        data = [
            cls.draw_nodes(graph),
            cls.draw_edges(
                graph, add_annotation=True, color="lightgray", dash="dot", width=1
            ),
        ]
        return data

    @staticmethod
    def draw_nodes(graph: Data) -> go.Scatter3d:
        x, y, z = graph.pos.T
        residue_names = graph.x


        residue_colors = {}
        for res in residue_names:
            if res not in residue_colors:
                residue_colors[res] = f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'


        nodes = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            hoverinfo="text",
            text=residue_names,  
            marker=dict(
                size=5,
                color=[residue_colors[res] for res in residue_names],  
                cmin=0,
                cmax=1,
                opacity=0.8,
            ),
        )
        return nodes

    @staticmethod
    def draw_edges(
        graph: Data,
        add_annotation: bool = False,
        color: str = "lightgray",
        dash: str = "dot",
        width: int = 1.2,
    ) -> go.Scatter3d:
        edges = graph.edge_index.T
        coordinates = graph.pos
        # compute pairwise distances
        src, dst = graph.edge_index
        distances = torch.linalg.norm(
            coordinates[src] - coordinates[dst], dim=1
        ).flatten()

        edges_plot = go.Scatter3d(
            x=list(
                itertools.chain(
                    *((coordinates[i, 0], coordinates[j, 0], None) for i, j in edges)
                )
            ),
            y=list(
                itertools.chain(
                    *((coordinates[i, 1], coordinates[j, 1], None) for i, j in edges)
                )
            ),
            z=list(
                itertools.chain(
                    *((coordinates[i, 2], coordinates[j, 2], None) for i, j in edges)
                )
            ),
            mode="lines",
            line=dict(
                color=color,
                width=width,
                dash=dash,
            ),
            text=list(
                itertools.chain(
                    *((f"{d:.3f}Å", f"{d:.3f}Å", None) for d in distances.tolist())
                )
            )
            if add_annotation
            else None,
            hoverinfo="text",
        )
        return edges_plot
