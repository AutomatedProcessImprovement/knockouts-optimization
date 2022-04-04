from pandas.plotting import parallel_coordinates
import seaborn as sns
import matplotlib.pyplot as plt

palette = sns.color_palette("bright", 10)


def display_parallel_coordinates_centroids(
    df, num_clusters, _figsize=(12, 5), _labelsize=15, _lw=4
):
    """Display a parallel coordinates plot for the centroids in df"""

    # Create the plot
    fig = plt.figure(figsize=_figsize)
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=24)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, "cluster", color=palette, lw=_lw)

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    ax = plt.gca()
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(_labelsize)


# random colors webpage: https://www.random.org/colors/hex
# get all color codes in js console: colors = $$("div.color-code-black").map(d => d.innerText)

colors32 = [
    "#FEFEBB",
    "#c7e9b4",
    "#41b6c4",
    "#74add1",
    "#4575b4",
    "#313695",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",  # until here, built-in colors
    "#7b0d60",
    "#305a47",
    "#d625a8",
    "#c537a2",
    "#694cb0",
    "#7badff",
    "#09b765",
    "#05206a",
    "#176b1f",
    "#2a77c1",
    "#b6ec1f",
    "#ba0174",
    "#1ef909",
    "#86d5dc",
    "#b9fa4c",
    "#da6065",
    "#56af01",
    "#b0eca5",
    "#94c041",
    "#8047cf",
    "#783d00",
    "#7817c3",
    "#671fb6",
    "#af0605",
    "#ad7fcc",
    "#52834c",
    "#c8cc6f",
]


def pseudo_powerset(iterable):
    res = [None, None]
    for i in range(2, len(iterable) + 1):
        tmp = []
        for j in range(0, i):
            tmp.append(iterable[j])
        res.append(tmp)
    return res


colors_powerset_32 = pseudo_powerset(colors32)


# obtained debugging library itself,
# classes are limited to 10, due to color array size!
# https://githubmemory.com/index.php/repo/parrt/dtreeviz/issues/151
DTREEVIZ_LIMIT = 10

dtreeviz_extended_colors = {
    "scatter_edge": "#444443",
    "scatter_marker": "#4575b4",
    "scatter_marker_alpha": 0.7,
    "class_boundary": "#444443",
    "warning": "#E9130D",
    "tile_alpha": 0.8,
    "tesselation_alpha": 0.3,
    "tesselation_alpha_3D": 0.5,
    "split_line": "#444443",
    "mean_line": "#f46d43",
    "axis_label": "#444443",
    "title": "#444443",
    "legend_title": "#444443",
    "legend_edge": "#444443",
    "edge": "#444443",
    "color_map_min": "#c7e9b4",
    "color_map_max": "#081d58",
    "classes": colors_powerset_32,
    "rect_edge": "#444443",
    "text": "#444443",
    "highlight": "#D67C03",
    "wedge": "#444443",
    "text_wedge": "#444443",
    "arrow": "#444443",
    "node_label": "#444443",
    "tick_label": "#444443",
    "leaf_label": "#444443",
    "pie": "#444443",
    "hist_bar": "#a6bddb",
    "categorical_split_left": "#FFC300",
    "categorical_split_right": "#4575b4",
}
