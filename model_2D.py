import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import ncc
import util
import networks
import regularizers
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('network_tensorboard')

class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = util.make_coordinate_tensor_2D(output_shape)
        # print(coordinate_tensor.shape)

        output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output, coordinate_tensor)

        transformed_image = self.transform_no_add(coord_temp)
        print(transformed_image.shape,'++++++++++')
        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
        )

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""
        self.deformation_field = None

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]
        self.loss = 0

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        util.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )

        # Initialization
        self.moving_image = torch.tensor(moving_image)
        self.fixed_image = torch.tensor(fixed_image)
        if self.mask:
            self.possible_coordinate_tensor = util.make_masked_coordinate_tensor(
                self.mask, self.fixed_image.shape
            )
        else:
            self.possible_coordinate_tensor = util.make_coordinate_tensor_2D(self.fixed_image.shape)

        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # Variables specific to this class
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["batch_size"] = 10000
        self.args["layers"] = [2, 256, 256, 256, 2]
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (224, 224)

        self.args["network"] = None

        self.args["epochs"] = 100
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "MLP"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0]
        )[: self.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp

        transformed_image = self.transform_no_add(coord_temp)
        fixed_image = util.fast_trilinear_interpolation_2D(
            self.fixed_image,
            coordinate_tensor[:, 0],
            coordinate_tensor[:, 1]
        )

        # Compute the loss
        loss += self.criterion(transformed_image, fixed_image)

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)
        print(coordinate_tensor.shape)

        # Regularization
        if self.jacobian_regularization:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss_2D(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            loss += self.alpha_bending * regularizers.compute_bending_energy_2D(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )

        # Perform the backpropagation and update the parameters accordingly
        self.loss = loss
        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform(
            self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return util.fast_trilinear_interpolation_2D(
            moving_image,
            transformation[:, 0],
            transformation[:, 1]
        )

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        return util.fast_trilinear_interpolation_2D(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
        )

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]
        dummy_input = torch.rand(1000,2)  # 网络中输入的数据维度
        with SummaryWriter("runs/2023_8_24") as w:
            w.add_graph(self.network, (dummy_input,))  # net是你的网络名

        # Perform training iterations
        writer = SummaryWriter("runs/2023_8_24")
        for i in tqdm.tqdm(range(epochs)):
            self.training_iteration(i)
            writer.add_scalar(tag="loss/train", scalar_value=self.loss,
                              global_step=i)  # 画loss，横坐标为epoch
        writer.close()
        coordinate_tensor = util.make_coordinate_tensor_2D(self.moving_image.shape)
        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        self.deformation_field = torch.reshape(coord_temp,((self.moving_image.shape[0],self.moving_image.shape[1],2)))


