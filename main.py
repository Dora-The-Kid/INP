import model_2D
import cv2
import warp
import torch
import numpy as np
import util
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
torch.save(ImpReg,'model')
coordinate_tensor = torch.FloatTensor(dstimg)
output = ImpReg(output_shape=(254,254))
output = np.array(output)
output = (output-np.min(output))/(np.max(output)-np.min(output))
print(output.shape)
cv2.imshow('image',output)
cv2.waitKey(0)
np.save('tran_matrix.npy',output)
