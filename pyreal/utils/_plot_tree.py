# Code adapted from scikit-learn.
# source: https://github.com/scikit-learn/scikit-learn/blob/582fa30a3/sklearn/tree/_export.py
# The TreeExporter class allows us to change the color or geometry of the plot
import numpy as np
from sklearn.tree._export import _MPLTreeExporter


def _color_brew(n):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


def hex2rgb(colors):
    if isinstance(colors, str):
        hex_code = colors.strip("# ")
        return [int(hex_code[:2], 16), int(hex_code[2:4], 16), int(hex_code[4:], 16)]
    elif isinstance(colors, list):
        color_list = []
        for code in colors:
            hex_code = code.strip("# ")
            color_list.append(
                [int(hex_code[:2], 16), int(hex_code[2:4], 16), int(hex_code[4:], 16)]
            )
        return color_list


class TreeExporter(_MPLTreeExporter):
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        positive_color=None,
        negative_color=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
    ):
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
        )
        self.positive_color = positive_color
        self.negative_color = negative_color

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            if self.positive_color is None and self.negative_color is None:
                self.colors["rgb"] = _color_brew(tree.n_classes[0])
            elif tree.n_classes == 1:
                self.colors["rgb"] = hex2rgb([self.positive_color])
            elif tree.n_classes == 2:
                self.colors["rgb"] = hex2rgb([self.negative_color, self.positive_color])
            else:
                self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    # TODO: edit the following functions to change the sizes of the elements
    # in the plot

    # def export(self, decision_tree, ax=None):
    #     import matplotlib.pyplot as plt
    #     from matplotlib.text import Annotation

    #     if ax is None:
    #         ax = plt.gca()
    #     ax.clear()
    #     ax.set_axis_off()
    #     my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
    #     draw_tree = buchheim(my_tree)

    #     # important to make sure we're still
    #     # inside the axis after drawing the box
    #     # this makes sense because the width of a box
    #     # is about the same as the distance between boxes
    #     max_x, max_y = draw_tree.max_extents() + 1
    #     ax_width = ax.get_window_extent().width
    #     ax_height = ax.get_window_extent().height

    #     scale_x = ax_width / max_x
    #     scale_y = ax_height / max_y
    #     self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y)

    #     anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

    #     # update sizes of all bboxes
    #     renderer = ax.figure.canvas.get_renderer()

    #     for ann in anns:
    #         ann.update_bbox_position_size(renderer)

    #     if self.fontsize is None:
    #         # get figure to data transform
    #         # adjust fontsize to avoid overlap
    #         # get max box width and height
    #         extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
    #         max_width = max([extent.width for extent in extents])
    #         max_height = max([extent.height for extent in extents])
    #         # width should be around scale_x in axis coordinates
    #         size = anns[0].get_fontsize() * min(
    #             scale_x / max_width, scale_y / max_height
    #         )
    #         for ann in anns:
    #             ann.set_fontsize(size)

    #     return anns

    # def recurse(self, node, tree, ax, max_x, max_y, depth=0):
    #     import matplotlib.pyplot as plt

    #     kwargs = dict(
    #         bbox=self.bbox_args.copy(),
    #         ha="center",
    #         va="center",
    #         zorder=100 - 10 * depth,
    #         xycoords="axes fraction",
    #         arrowprops=self.arrow_args.copy(),
    #     )
    #     kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

    #     if self.fontsize is not None:
    #         kwargs["fontsize"] = self.fontsize

    #     # offset things by .5 to center them in plot
    #     xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

    #     if self.max_depth is None or depth <= self.max_depth:
    #         if self.filled:
    #             kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
    #         else:
    #             kwargs["bbox"]["fc"] = ax.get_facecolor()

    #         if node.parent is None:
    #             # root
    #             ax.annotate(node.tree.label, xy, **kwargs)
    #         else:
    #             xy_parent = (
    #                 (node.parent.x + 0.5) / max_x,
    #                 (max_y - node.parent.y - 0.5) / max_y,
    #             )
    #             ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
    #         for child in node.children:
    #             self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)

    #     else:
    #         xy_parent = (
    #             (node.parent.x + 0.5) / max_x,
    #             (max_y - node.parent.y - 0.5) / max_y,
    #         )
    #         kwargs["bbox"]["fc"] = "grey"
    #         ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)
