import model_2D
import cv2
import warp
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
print('ttttttttttttttttttttttttttt')
case_id = '0'
out_dir = 'out'
kwargs = {}
kwargs["verbose"] = False
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = True
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = out_dir + str(case_id)
kwargs["mask"] = False

srcimg = cv2.imread("n01983481_171.JPEG")

dstimg = warp.B_spline_form(srcimg)
dstimg = np.array(dstimg)
print(dstimg.shape)
srcimg = srcimg[:, :, 1]
dstimg = dstimg[:, :, 1]
# srcimg = np.tile(srcimg,[224,1,1])
# dstimg = np.tile(dstimg,[224,1,1])
print(srcimg.shape)
ImpReg = model_2D.ImplicitRegistrator(dstimg, srcimg, **kwargs)
print('fit')
ImpReg.fit()
coordinate_tensor = torch.FloatTensor(dstimg)
output = ImpReg(output_shape=(224,224))
output = np.array(output)
output = (output-np.min(output))/(np.max(output)-np.min(output))
writer = SummaryWriter("runs/2023_8_8")
srcimg =srcimg[np.newaxis,:]
writer.add_image("groundtruth",srcimg,1)
dstimg =dstimg[np.newaxis,:]
writer.add_image("wrap",dstimg,1)
output =output[np.newaxis,:]
writer.add_image("output",output,1)
cv2.imshow('final',output)
cv2.waitKey(0)
np.save('tran_matrix.npy',output)

