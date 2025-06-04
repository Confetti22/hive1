import napari
import numpy as np
from magicgui import magicgui
from helper.simple_viewer import SimpleViewer2
from helper.image_seger import SimpleSeger2
from helper.napari_view_utilis import MultipleViewerWidget, toggle_layer_visibility
from helper.one_dim_statis import OneDimStatis_roi

##### init model dict ###########
#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

os.environ["NAPARI_ASYNC"] = "1"

from lib.arch.ae import build_final_model,load_compose_encoder_dict, build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
import numpy as np
import torchvision.models as models
from torchvision.models import Inception_V3_Weights


device ='cuda'

args = load_cfg('config/t11_3d.yaml')
args.avg_pool_size = (8,8,8) 

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
mlp_ckpt_pth ='/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)

#prepare the pretrained inception_v3 model
weights = Inception_V3_Weights.DEFAULT
incep_model = models.inception_v3(weights = weights, progress =True)
incep_model.eval()
incep_model.to(device)

model_dict={}
model_dict['mlp']=cmpsd_model
model_dict['ae']  = encoder_model
model_dict['inception']=incep_model




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



# Set up the watcher
simple_viewer_widget = SimpleViewer2(viewer1,viewer2)
simple_seger_widegt = SimpleSeger2(viewer1,viewer2,simple_viewer_widget)

statis_widget= OneDimStatis_roi(viewer=viewer1,image_layer=simple_viewer_widget.roi_layer,model_dict=model_dict)
@viewer2.bind_key('v')
def toggle_mask_viewer2(_module):
    print(f'press \'v\' at  {_module}')
    toggle_layer_visibility(layers=viewer2.layers,name_patterns=['mask','region','polygon'])

@viewer1.bind_key('v')
def toggle_mask_viewer2(_module):
    print(f'press \'v\' at  {_module}')
    toggle_layer_visibility(layers=viewer2.layers,name_patterns=['segout'])



viewer1.window.add_dock_widget(simple_viewer_widget,area='right')
viewer1.window.add_dock_widget(simple_seger_widegt,area='left')
viewer1.window.add_dock_widget(statis_widget,area='left')

napari.run()
