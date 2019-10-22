import torch
from PIL import Image


def load_image(filename, size=None, scale=None):        # 这里的size是一个int型的变量
    img = Image.open(filename)
    if size is not None:                                # 裁剪到宽和高一样
        img = img.resize((size, size), Image.ANTIALIAS) # ANTIALIAS的作用：PIL的resize中可能导致图片受损，ANTIALIAS控制图片平滑无锯齿
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)   # mean的torch.Size([3, 1, 1])
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)    # std的torch.Size([3, 1, 1])
    batch = batch.div_(255.0)                                       # 归一化到[0,1]
    return (batch - mean) / std                                     # 返回的结果如下：
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
