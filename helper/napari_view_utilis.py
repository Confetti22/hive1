from copy import deepcopy
from typing import Optional

import numpy as np

from packaging.version import parse as parse_version
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import qthrottled

import napari
from napari.utils.translations import trans

from napari.components.layerlist import Extent
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer, Vectors
from napari.qt import QtViewer
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info
from napari.layers.utils.layer_utils import dims_displayed_world_to_layer
import os

def _filter_layer_name_with_pattern(layers,name_patterns):
    layer_names=set()
    for pattern in name_patterns:
        lst=[layer.name for layer  in layers if pattern in layer.name]
        layer_names.update(lst)
    return layer_names

def sync_planes_pos(event, layers):
    # Get the plane position and normal from the triggering layer
    pos = event.source.position
    normal = event.source.normal
    # Update all other layers with the same position and normal
    for layer in layers:
        if layer != event.source:  # Skip the triggering layer
            layer.plane.position = pos
            layer.plane.normal = normal


def link_pos_of_plane_layers(layers):
    for layer in layers:
        # Connect both the position and normal events to sync_planes_pos
        layer.plane.events.position.connect(lambda event: sync_planes_pos(event, layers))
        layer.plane.events.normal.connect(lambda event: sync_planes_pos(event, layers))



def save_remove_layer_given_full_name(layers,name):
    for layer in layers:
        if layer.name == name:
            layers.remove(name)
    
def toggle_layer_visibility(layers,name_patterns):
    layer_names=_filter_layer_name_with_pattern(layers,name_patterns)

    # if not all the layer have the same visible status, set all the layer.visibity to False;
    # else set all the layer.visibity the negation

    # Check if all layers have the same visibility status
    all_visible_status = [layers[layer_name].visible for layer_name in layer_names]

    if all(all_visible_status) or not any(all_visible_status):
        # If all layers have the same visibility status, set all to the negation of their current state
        new_visibility = not all_visible_status[0]
        for layer_name in layer_names:
            layers[layer_name].visible = new_visibility
            print(f"visibility of layer '{layer_name}' is {new_visibility} now")
    else:
        # If not all layers have the same visibility, set all layers to False
        for layer_name in layer_names:
            layers[layer_name].visible = False
            print(f"visibility of layer '{layer_name}' is False now")


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: Optional[str] = None,
        layer_type: Optional[str] = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class ExampleWidget(QWidget):
    """
    Dummy widget showcasing how to place additional widgets to the right
    of the additional viewers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.btn = QPushButton('Perform action')
        self.spin = QDoubleSpinBox()
        layout = QVBoxLayout()
        layout.addWidget(self.spin)
        layout.addWidget(self.btn)
        layout.addStretch(1)
        self.setLayout(layout)

class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title='model1')
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)


        # self.tab_widget = QTabWidget()
        # w1 = ExampleWidget()
        # w2 = ExampleWidget()
        # self.tab_widget.addTab(w1, 'Sample 1')
        # self.tab_widget.addTab(w2, 'Sample 2')
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        viewer_splitter.addWidget(self.qt_viewer1)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        self.addWidget(viewer_splitter)
        
        # self.addWidget(self.tab_widget)

        # self.qt_viewer1.setFixedSize(1000, 1000)  # Set width and height as desired


def get_dims_displayed(layer):
    # layer._dims_displayed was removed in
    # https://github.com/napari/napari/pull/5003
    if hasattr(layer, "_slice_input"):
        return layer._slice_input.displayed
    return layer._dims_displayed


def point_in_layer_bounding_box(point, layer):
    """Determine whether an nD point is inside a layers nD bounding box.

    Parameters
    ----------
    point : np.ndarray
        (n,) array containing nD point coordinates to check.
    layer : napari.layers.Layer
        napari layer to get the bounding box from
    
    Returns
    -------
    bool
        True if the point is in the bounding box of the layer,
        otherwise False
    
    Notes
    -----
    For a more general point-in-bbox function, see:
        `napari_threedee.utils.geometry.point_in_bounding_box`
    """
    dims_displayed = get_dims_displayed(layer)
    bbox = layer._display_bounding_box(dims_displayed).T
    if np.any(point < bbox[0]) or np.any(point > bbox[1]):
        return False
    else:
        return True


def add_point_on_plane(
    viewer: napari.viewer.Viewer,
    points_layer: napari.layers.Points = None,
    image_layer: napari.layers.Image = None,
    replace_selected: bool = False,
):
    # Early exit if image_layer isn't visible
    if image_layer.visible is False or image_layer.depiction != 'plane':
        return

    # event.position and event.view_direction are in world (scaled) coordinates
    position_world = viewer.cursor.position
    view_direction_world = viewer.camera.view_direction
    ndim_world = len(position_world)
    dims_displayed_world = np.asarray(viewer.dims.displayed)[list(viewer.dims.displayed_order)]

    # use image layer data (pixel) coordinates, because that's what plane.position uses
    position_image_data_coord = np.asarray(image_layer.world_to_data(position_world))
    view_direction_image_data_coord = np.asarray(image_layer._world_to_data_ray(view_direction_world))

    dims_displayed_image_layer = np.asarray(dims_displayed_world_to_layer(
        dims_displayed_world,
        ndim_world=ndim_world,
        ndim_layer=image_layer.ndim,
    ))

    # Calculate 3d intersection of click with plane through data in image data (pixel) coordinates
    position_image_data_3d = position_image_data_coord[dims_displayed_image_layer]
    view_direction_image_data_3d = view_direction_image_data_coord[dims_displayed_image_layer]
    intersection_image_data_3d = image_layer.plane.intersect_with_line(
        line_position=position_image_data_3d,
        line_direction=view_direction_image_data_3d
    )

    # Check if click was on plane by checking if intersection occurs within image layer
    # data bounding box. If not, exit early.
    if not point_in_layer_bounding_box(intersection_image_data_3d, image_layer):
        return

    # Transform the intersection coordinate from image layer coordinates to world coordinates
    intersection_3d_world = np.asarray(image_layer.data_to_world(intersection_image_data_3d))[
        dims_displayed_image_layer]

    # convert to nd in world coordinates
    intersection_nd_world = np.asarray(viewer.dims.point)
    intersection_nd_world[dims_displayed_image_layer] = intersection_3d_world

    # Transform the the point in world coordinates to point layer data coordinates
    intersection_3d_points = points_layer.world_to_data(intersection_3d_world)
    intersection_nd_points = points_layer.world_to_data(intersection_nd_world)

    if replace_selected:
        points_layer.remove_selected()
    if points_layer.data.shape[-1] < len(intersection_nd_points):
        intersection_nd_points = intersection_3d_points
    points_layer.add(intersection_nd_points)

