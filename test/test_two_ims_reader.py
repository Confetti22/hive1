from image_reader import wrap_image
from image_reader import Ims_Image
import numpy as np

level = 0
stride = 16
channel =2

roi_offset = [7500,5600,4850]
roi_size =[1,1024,1024]

ims_vol1 = Ims_Image(ims_path="/home/confetti/e5_data/t1779/t1779.ims", channel=channel)
ims_vol2 = wrap_image("/home/confetti/e5_data/t1779/t1779.ims")

raw_volume_size1 =ims_vol1.rois[level][3:] #the data shape at r3 for test
raw_volume_size2 =ims_vol2.rois[level][3:] #the data shape at r3 for test
print(f"{raw_volume_size1=}, {raw_volume_size2=}")
whole_volume_size1 = [int(element//2) for element in raw_volume_size1]
whole_volume_offset1 = [int(element//4) for element in raw_volume_size1]
print(f"{whole_volume_offset1=}")


roi1 = ims_vol1.from_roi(coords=[*roi_offset,*roi_size],level=level)
roi2 = ims_vol2.from_roi(coords=[*roi_offset, *roi_size],level= level,channel= channel)
print(f"{np.array_equal(roi1,roi2)=}")

