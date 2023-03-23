import colorsys

import matplotlib.colors as mc
import seaborn as sns

NEGATIVE_COLOR = "#710627"
POSITIVE_COLOR = "#08415C"
NEUTRAL_COLOR = "#FEEBC3"


def _lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    """
    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


NEGATIVE_COLOR_LIGHT = _lighten_color(NEGATIVE_COLOR, 0.8)
POSITIVE_COLOR_LIGHT = _lighten_color(POSITIVE_COLOR, 0.8)

PALETTE = sns.blend_palette([NEGATIVE_COLOR_LIGHT, NEUTRAL_COLOR, POSITIVE_COLOR_LIGHT])
PALETTE_CMAP = sns.blend_palette(
    [NEGATIVE_COLOR_LIGHT, NEUTRAL_COLOR, POSITIVE_COLOR_LIGHT], as_cmap=True
)
