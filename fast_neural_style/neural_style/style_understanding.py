import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/Users/chenqy/PycharmProjects/examples', '/Users/chenqy/PycharmProjects/examples/fast_neural_style', '/Users/chenqy/PycharmProjects/examples/fast_neural_style/neural_style'])
# Python 3.7.4 (default, Aug 13 2019, 15:17:50)

import utils
filename = "/Users/chenqy/PycharmProjects/examples/fast_neural_style/images/style-images/mosaic.jpg"
size = 391
utils.load_image(filename, size)
# Out[6]: <PIL.Image.Image image mode=RGB size=391x391 at 0x102DF0650>
style = utils.load_image(filename, size)

import torch
import torchvision
from torchvision import transforms
style_transform = transforms.Compose([
        transforms.ToTensor(),                  # PIL image会转成Tensor，从（C*H*W）到（H*W*C），且从[0,255]到[0.0,1.0]。
                                                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
        transforms.Lambda(lambda x: x.mul(255)) # x中每个数乘以255，为什么？相当于只是转成tensor，但是每个值没有做归一化，范围还是在[0,255]
    ])
style = style_transform(style)
style.size()
# Out[14]: torch.Size([3, 391, 391])
batchsize = 4

device = torch.device("cuda" if 0 else "cpu")
style = style.repeat(batchsize,1,1,1).to(device)
style.size()
# Out[20]: torch.Size([4, 3, 391, 391])

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

import vgg
from vgg import Vgg16
vgg = Vgg16(requires_grad=False).to(device)
# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /Users/chenqy/.cache/torch/checkpoints/vgg16-397923af.pth
# 100.0%


batch = style
mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)   # mean的torch.Size([3, 1, 1])
std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)    # std的torch.Size([3, 1, 1])
batch = batch.div_(255.0)                                       # 归一化到[0,1]

normalizeresult = (batch - mean) / std
# tensor([[[[-1.3644, -0.9705,  0.3823,  ..., -0.7308, -0.7650, -0.7308],
#           [ 0.0741, -1.1760, -0.8335,  ...,  0.6392,  0.3309,  0.2111],
#           [ 0.5707,  0.4851, -0.9877,  ...,  0.8276,  0.4679, -1.3644],
#           ...,
#           [ 1.1358,  1.2557,  1.1529,  ..., -0.6109,  1.3070,  1.1015],
#           [ 1.1358,  1.0502,  1.2214,  ..., -1.6898, -0.2171,  1.1872],
#           [ 0.8961,  1.1015,  0.5364,  ..., -0.8507, -1.8782,  0.1597]],
#
#          [[-1.5805, -1.1604,  0.2577,  ..., -0.7752, -0.8102, -0.7752],
#           [-0.0574, -1.3354, -0.9853,  ...,  0.6254,  0.3102,  0.1877],
#           [ 0.4503,  0.3627, -1.1253,  ...,  0.8179,  0.4503, -1.4230],
#           ...,
#           [ 1.6933,  1.6933,  1.4132,  ..., -0.5476,  1.4657,  1.2731],
#           [ 1.7108,  1.5007,  1.4657,  ..., -1.6856, -0.1275,  1.3431],
#           [ 1.4657,  1.5532,  0.7479,  ..., -0.8627, -1.8782,  0.2052]],
#
#          [[-1.6999, -1.2641,  0.1476,  ..., -0.5321, -0.5670, -0.5321],
#           [-0.2010, -1.4384, -...

features_style = vgg(normalizeresult)
# VggOutputs(relu1_2=tensor([[[[ 0.0000,  1.2763,  3.2583,  ...,  0.0000,  0.0000,  1.1532],
#           [ 0.0000,  0.0000,  4.8556,  ...,  0.0000,  0.0000,  0.0000],
#           [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.6969],
#           ...,
#           [ 1.4070,  1.7247,  0.0000,  ...,  8.8745,  6.7917,  1.4484],
#           [ 1.4567,  0.3573,  0.0000,  ...,  0.0000,  9.0376,  7.9567],
#           [ 0.9066,  0.0000,  0.0000,  ...,  0.0000,  2.7535, 10.8945]],
#
#          [[ 0.0000,  0.0000,  0.7368,  ...,  0.0000,  0.0000,  0.0000],
#           [ 0.0000,  0.0000,  0.0000,  ...,  1.3038,  0.0000,  0.0000],
#           [ 1.5022,  0.0000,  0.0000,  ...,  1.3023,  0.0000,  0.0000],
#           ...,
#           [ 0.9290,  0.5096,  2.0468,  ...,  0.6405,  5.2257,  0.0000],
#           [ 3.0888,  3.1716,  0.9424,  ...,  0.0000,  2.6377,  0.0000],
#           [ 1.9257,  2.3722,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
#
#          [[ 0.0000,  0.0000,  1.4539,  ...,  0.0000,  0.0000,  0.0000],
#           [...

gram_style = [utils.gram_matrix(y) for y in features_style]
# [tensor([[[0.0839, 0.0138, 0.0385,  ..., 0.0045, 0.0395, 0.0097],
#          [0.0138, 0.0268, 0.0177,  ..., 0.0204, 0.0284, 0.0009],
#          [0.0385, 0.0177, 0.0624,  ..., 0.0141, 0.0613, 0.0221],
#          ...,
#          [0.0045, 0.0204, 0.0141,  ..., 0.0495, 0.0178, 0.0021],
#          [0.0395, 0.0284, 0.0613,  ..., 0.0178, 0.1260, 0.0235],
#          [0.0097, 0.0009, 0.0221,  ..., 0.0021, 0.0235, 0.2307]],
#
#         [[0.0839, 0.0138, 0.0385,  ..., 0.0045, 0.0395, 0.0097],
#          [0.0138, 0.0268, 0.0177,  ..., 0.0204, 0.0284, 0.0009],
#          [0.0385, 0.0177, 0.0624,  ..., 0.0141, 0.0613, 0.0221],
#          ...,
#          [0.0045, 0.0204, 0.0141,  ..., 0.0495, 0.0178, 0.0021],
#          [0.0395, 0.0284, 0.0613,  ..., 0.0178, 0.1260, 0.0235],
#          [0.0097, 0.0009, 0.0221,  ..., 0.0021, 0.0235, 0.2307]],
#
#         [[0.0839, 0.0138, 0.0385,  ..., 0.0045, 0.0395, 0.0097],
#          [0.0138, 0.0268, 0.0177,  ..., 0.0204, 0.0284, 0.0009],
#          [0.0385, 0.0177, 0.0624,  ..., 0.0141, 0.0613, 0...