import torch
import numpy as np
import matplotlib.pyplot as plt
def make_coordinate_tensor(dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    print(coordinate_tensor)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    print(coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    print(coordinate_tensor)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    # coordinate_tensor = coordinate_tensor

    return coordinate_tensor

def make_coordinate_tensor_2D(dims=(28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    print(coordinate_tensor)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    print(coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=2)
    print(coordinate_tensor)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 2])

    # coordinate_tensor = coordinate_tensor

    return coordinate_tensor

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0


    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output

def fast_trilinear_interpolation_2D(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5


    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    x = x_indices - x0
    y = y_indices - y0


    output = (
        input_array[x0, y0] * (1 - x) * (1 - y)
        + input_array[x1, y0] * x * (1 - y)
        + input_array[x0, y1] * (1 - x) * y
        + input_array[x1, y1] * x * y

    )
    return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

a = make_coordinate_slice(dims=(5,5),gpu=False)
indices = torch.randperm(
            a.shape[0]
        )
a = a[indices, :]
print(a)
print(a.shape)
plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(a[:,0],a[:,1],a[:,2])
plt.show()