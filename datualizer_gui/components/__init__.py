"""GUI Components package for Datualizer."""

from datualizer_gui.components.data_grid import DataGridWidget, PolarsTableModel
from datualizer_gui.components.inspector import InspectorPanel
from datualizer_gui.components.plot_canvas import MultiChannelPlotCanvas
from datualizer_gui.components.recipe_panel import RecipePanel

__all__ = [
    "DataGridWidget",
    "PolarsTableModel",
    "InspectorPanel",
    "MultiChannelPlotCanvas",
    "RecipePanel",
]
