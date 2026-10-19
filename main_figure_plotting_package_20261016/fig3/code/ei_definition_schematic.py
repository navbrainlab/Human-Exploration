"""Native explanatory EI diagram; no behavioral values or predictor are computed."""

import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D


def draw_ei_definition(fig, width_mm, height_mm, *, ink, purple, gray):
    def xy(point):
        return point[0] / width_mm, 1 - point[1] / height_mm

    def label(x, y, text, *, size=6, va="center"):
        artist = fig.text(*xy((x, y)), text, ha="center", va=va,
                          fontsize=size, fontweight="normal", color=ink)
        artist.set_gid("ei-definition-text")
        return artist

    label(29, 87, "Exploration index (EI)")
    nodes = [
        dict(id="previous", text=r"$C_{t-1}$", anchor_mm=[10, 127], fill="#F4F5F6", edge=gray, linestyle="-"),
        dict(id="current", text=r"$C_t$", anchor_mm=[25, 115], fill="#E7EAED", edge=ink, linestyle="-"),
        dict(id="actual", text=r"$C_{t+1}$", anchor_mm=[44, 99], fill="#E7EAED", edge=ink, linestyle="-"),
        dict(id="predicted", text=r"$C^*_{t+1}$", anchor_mm=[44, 135], fill="#EFEAF5", edge=purple, linestyle=(0, (3, 2))),
    ]
    radius = 4.0
    lookup = {node["id"]: node for node in nodes}
    arrows = []
    for start, stop, predicted in [("previous", "current", False), ("current", "actual", False), ("current", "predicted", True)]:
        a = np.array(lookup[start]["anchor_mm"], float)
        b = np.array(lookup[stop]["anchor_mm"], float)
        direction = (b - a) / np.linalg.norm(b - a)
        a += (radius + .8) * direction
        b -= (radius + .8) * direction
        artist = FancyArrowPatch(xy(a), xy(b), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=6, linewidth=1.0,
            color=purple if predicted else ink,
            linestyle=(0, (3, 2)) if predicted else "-", shrinkA=0, shrinkB=0)
        artist.set_gid("ei-definition-predicted-path" if predicted else "ei-definition-observed-path")
        fig.add_artist(artist)
        arrows.append(dict(start=start, stop=stop, predicted=predicted, endpoints_mm=[a.tolist(), b.tolist()]))
    for node in nodes:
        circle = Ellipse(xy(node["anchor_mm"]), width=2 * radius / width_mm,
                         height=2 * radius / height_mm, transform=fig.transFigure,
                         facecolor=node["fill"], edgecolor=node["edge"],
                         linewidth=.8, linestyle=node["linestyle"])
        circle.set_gid("ei-definition-node-" + node["id"])
        fig.add_artist(circle)
        label(*node["anchor_mm"], node["text"])
        node["radius_mm"] = radius
    label(44, 89.5, "Actual\nchoice", va="top")
    label(44, 142.5, "Predicted\noptimum", va="top")
    for a, b in [((51, 105), (51, 129)), ((49.7, 105), (52.3, 105)), ((49.7, 129), (52.3, 129))]:
        line = Line2D([xy(a)[0], xy(b)[0]], [xy(a)[1], xy(b)[1]],
                      transform=fig.transFigure, color=ink, lw=.8)
        line.set_gid("ei-definition-distance")
        fig.add_artist(line)
    label(55, 117, "EI")
    formula = r"$\mathrm{EI}_{t+1}=d(C_{t+1},C^*_{t+1})$"
    label(30, 153, formula)
    fig._ei_definition = dict(role="illustrative schematic", nodes=nodes, arrows=arrows,
        formula=formula, metric="Distance between actual next choice and predicted optimum",
        source="User-supplied EI reference schematic, redrawn natively",
        predictor="Conceptual reference; not a DGEM fit or replay",
        no_new_predictor_calculated=True, schematic_coordinates_are_not_empirical=True,
        bounds_mm=[4, 84, 58, 156], distance_bracket_mm=[51, 105, 129])
