import torch.nn as nn
import torchvision.models as models
from flexflow.torch.model import PyTorchModel

# model = models.alexnet()

# model = models.vgg16()

# model = models.squeezenet1_0()

model = models.densenet161()

# model = models.inception_v3()

# there seems to be a condition in the last step of forward?
"""
https://github.com/pytorch/vision/blob/a33ce08b64eebc3210dd8544e3d84e1dc495dc6b/torchvision/models/googlenet.py#LL94C14-L94C14
"""
#model = models.googlenet()

# model = models.shufflenet_v2_x1_0()

# model = models.mobilenet_v2()
ff_torch_model = PyTorchModel(model)
ff_torch_model.torch_to_file("densenet161.ff")