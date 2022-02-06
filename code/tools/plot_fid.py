import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
import yaml


def read_canal_fid_scores(scores):
    """

    :param scores: (dict)
    :return:
    """
    canal_names = []
    canal_fids = []
    for key, fid_items in scores.items():
        print('fid items', fid_items)
        for fid_key, fid_values in fid_items.items():
            print('fid_values', fid_values)
            canal_names = [canal_name for canal_name in fid_values.keys()]
            canal_fids = [float(canal_fid[0]) for canal_fid in fid_values.values()]
    return canal_names, canal_fids


def plot_canal_fid_scores(canal_names, canal_fids, title=None):

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=("FID Scores"))

    # Delta positive rates
    fig.add_trace(go.Bar(marker_color='darkred',
                         y=canal_fids,
                         name="",
                         x=canal_names), row=1, col=1)

    fig.update_xaxes(title_text="Channels Suppressed", row=1, col=1)
    fig.update_yaxes(title_text="FID Score", row=1, col=1)

    title = "Layer 16"
    fig.update_layout(title_text=title,
                      title_font_size=30,
                      height=600, width=1000)
    fig.show()


if __name__ == '__main__':
    with open('../../evaluation/layer_16_evaluation_metrics_all.yaml', 'r') as yfile:
        scores = yaml.load(yfile)
    print(scores)
    canal_names, canal_fids = read_canal_fid_scores(scores)
    plot_canal_fid_scores(canal_names, canal_fids, title=None)