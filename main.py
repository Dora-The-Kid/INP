import model_2D
import cv2
import warp
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import util
def create_grid(size, path):
    num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // 10  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))

    plt.figure(figsize=((size[0] + 10) / 100.0, (size[1] + 10) / 100.0))  # 指定图像大小
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')  # 不显示坐标轴
    # 去除白色边框
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path)  # 保存图像


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
np.random.seed(42)
srcimg = cv2.imread("n01983481_171.JPEG")
dstimg = warp.B_spline_form(srcimg)
dstimg = np.array(dstimg)
print(dstimg.shape)
srcimg = srcimg[:, :, 1]
dstimg = dstimg[:, :, 1]
# srcimg = np.tile(srcimg,[224,1,1])
# dstimg = np.tile(dstimg,[224,1,1])
print(srcimg.shape)
ImpReg = model_2D.ImplicitRegistrator(srcimg, dstimg, **kwargs)
print('fit')
ImpReg.fit()
util.grid2contour(ImpReg.deformation_field.detach().numpy())
plt.show()
writer = SummaryWriter("runs/2023_8_24")
output = ImpReg(output_shape=(224,224))
output = np.array(output)
output = (output-np.min(output))/(np.max(output)-np.min(output))
# writer = SummaryWriter("runs/2023_8_8")
srcimg =srcimg[np.newaxis,:]
writer.add_image("groundtruth",srcimg,1)
dstimg =dstimg[np.newaxis,:]
writer.add_image("wrap",dstimg,1)
output =output[np.newaxis,:]
writer.add_image("output",output,1)
writer.close()

cv2.imshow('final',output)
cv2.waitKey(0)
np.save('tran_matrix.npy',output)

