import model
import cv2
import warp
import torch
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
print('2222')
dstimg = warp.B_spline_form(srcimg)
print(dstimg.shape)
srcimg = dstimg[:, :, 1]
dstimg = srcimg[:, :, 1]

ImpReg = model.ImplicitRegistrator(dstimg, srcimg, **kwargs)
print('fit')
ImpReg.fit()
coordinate_tensor = torch.FloatTensor(dstimg)
output = ImpReg(coordinate_tensor)
