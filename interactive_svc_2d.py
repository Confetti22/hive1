import napari
import numpy as np
from magicgui import magicgui
from helper.simple_viewer import SimpleViewer
from helper.image_seger import SimpleSeger
from helper.napari_view_utilis import MultipleViewerWidget, toggle_layer_visibility

###### init main napari viewer ########
##################################



#main viewer
viewer1 = napari.Viewer(ndisplay=2)
####### init sub viewer #####

dock_widget=MultipleViewerWidget(viewer1)
viewer1.window.add_dock_widget(dock_widget,name="multiViewer")

# adjust size of main_window
origin_geo=viewer1.window.geometry()
viewer1.window.set_geometry(origin_geo[0],origin_geo[1],origin_geo[2],origin_geo[3])

#sub_viewer, can be treated as a viewer
viewer2=viewer1.window._dock_widgets['multiViewer'].widget().viewer_model1




# Create a Z-slider widget using magicgui
# @magicgui(
#     auto_call=True, 
#     z={"widget_type":"Slider","label": "Z-index", "min": 0, "max": dummpy_data.shape[0] - 1, "step": 2},
# )
# def adjust_z_index_wid(z):
#     #refresh the data displayed on the viewer
#     viewer2.layers['dummy2'].data = dummpy_data[z]


# Display the slider as a dock widget at the bottom of the viewer
# viewer1.window.add_dock_widget(adjust_z_index_wid, area='right', name='Z Controller')

import weakref

def on_destroy():
    print("simple_viewer_widget was destroyed (garbage collected)")

# Set up the watcher
simple_viewer_widget = SimpleViewer(viewer1,viewer2)
weakref.finalize(simple_viewer_widget, on_destroy)

simple_seger_widegt = SimpleSeger(viewer1,viewer2,simple_viewer_widget)


@viewer2.bind_key('v')
def toggle_mask_viewer2(_module):
    print(f'press \'v\' at  {_module}')
    toggle_layer_visibility(layers=viewer2.layers,name_patterns=['mask','region','polygon'])


viewer1.window.add_dock_widget(simple_viewer_widget,area='right')
viewer1.window.add_dock_widget(simple_seger_widegt,area='left')


napari.run()
