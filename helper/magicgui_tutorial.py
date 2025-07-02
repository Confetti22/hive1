#using magicgui decorator will wrap the function in a FunctionGui object and turn it into a widget 
import napari
from magicgui import magicgui

viewer = napari.Viewer(ndisplay=2)
@magicgui(call_button='test2',layout='vertical')
def test_widg(viewer):
    global last_label_data
    print("called with viewer",viewer)
    print(f"in test_func,, last_label_data.shape {last_label_data.shape}")

print(type(test_widg))
#the parameter of test_widg is now turned into a sub-widget
test_widg.viewer.value = viewer
#alternative way
print(f"the viewer attribute type {type(test_widg['viewer'])}")
# test_widg['viewer'].value = viewer

viewer.window.add_dock_widget(test_widg, area='right')

napari.run()