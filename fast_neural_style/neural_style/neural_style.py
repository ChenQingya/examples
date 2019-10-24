import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([            # 针对数据集的transform，数据集中有很多张图片，使用dataloader进行枚举
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)   # 定义的TransformerNet，实例化
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([      # 针对style image的transform
        transforms.ToTensor(),                  # PIL image会转成Tensor，从（C*H*W）到（H*W*C），且从[0,255]到[0.0,1.0]。
                                                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
        transforms.Lambda(lambda x: x.mul(255)) # x中每个数乘以255，为什么？相当于只是转成tensor，但是每个值没有做归一化，范围还是在[0,255]
    ])
    style = utils.load_image(args.style_image, size=args.style_size)    # args.style_image:default="images/style-images/mosaic.jpg"。args.style_size:default is the original size of style image
    style = style_transform(style)                                      # 将PIL image进行transform，转成tensor
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)           # tensor的repeat.
    # styleimage（examples/fast_neural_style/images/style-images/mosaic.jpg）
    # 经过vgg16的预训练模型（下载vgg模型并加载了.pth），得到features_style
    features_style = vgg(utils.normalize_batch(style))                  # features_style如下：
                                                                        # VggOutputs(relu1_2=tensor([[[[0.0000, 1.2763, 3.2583, ..., 0.0000, 0.0000, 1.1532],
                                                                        #                              [0.0000, 0.0000, 4.8556, ..., 0.0000, 0.0000, 0.0000],
                                                                        #                              [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.6969],
                                                                        #                              ...,
                                                                        #                              [1.4070, 1.7247, 0.0000, ..., 8.8745, 6.7917, 1.4484],
                                                                        #                              [1.4567, 0.3573, 0.0000, ..., 0.0000, 9.0376, 7.9567],
                                                                        #                              [0.9066, 0.0000, 0.0000, ..., 0.0000, 2.7535, 10.8945]],
                                                                        #
                                                                        #                             [[0.0000, 0.0000, 0.7368, ..., 0.0000, 0.0000, 0.0000],
                                                                        #                              [0.0000, 0.0000, 0.0000, ..., 1.3038, 0.0000, 0.0000],
                                                                        #                              [1.5022, 0.0000, 0.0000, ..., 1.3023, 0.0000, 0.0000],
                                                                        #                              ...,
                                                                        #                              [0.9290, 0.5096, 2.0468, ..., 0.6405, 5.2257, 0.0000],
                                                                        #                              [3.0888, 3.1716, 0.9424, ..., 0.0000, 2.6377, 0.0000],
                                                                        #                              [1.9257, 2.3722, 0.0000, ..., 0.0000, 0.0000, 0.0000]],
                                                                        #
                                                                        #                             [[0.0000, 0.0000, 1.4539, ..., 0.0000, 0.0000, 0.0000],
                                                                        #                              [...
    gram_style = [utils.gram_matrix(y) for y in features_style]         # 有四个y在features_style里，relu1_2,relu2_2,relu3_3,relu4_4。gram_style如下：
                                                                        # [tensor([[[0.0839, 0.0138, 0.0385, ..., 0.0045, 0.0395, 0.0097],
                                                                        #           [0.0138, 0.0268, 0.0177, ..., 0.0204, 0.0284, 0.0009],
                                                                        #           [0.0385, 0.0177, 0.0624, ..., 0.0141, 0.0613, 0.0221],
                                                                        #           ...,
                                                                        #           [0.0045, 0.0204, 0.0141, ..., 0.0495, 0.0178, 0.0021],
                                                                        #           [0.0395, 0.0284, 0.0613, ..., 0.0178, 0.1260, 0.0235],
                                                                        #           [0.0097, 0.0009, 0.0221, ..., 0.0021, 0.0235, 0.2307]],
                                                                        #
                                                                        #          [[0.0839, 0.0138, 0.0385, ..., 0.0045, 0.0395, 0.0097],
                                                                        #           [0.0138, 0.0268, 0.0177, ..., 0.0204, 0.0284, 0.0009],
                                                                        #           [0.0385, 0.0177, 0.0624, ..., 0.0141, 0.0613, 0.0221],
                                                                        #           ...,
                                                                        #           [0.0045, 0.0204, 0.0141, ..., 0.0495, 0.0178, 0.0021],
                                                                        #           [0.0395, 0.0284, 0.0613, ..., 0.0178, 0.1260, 0.0235],
                                                                        #           [0.0097, 0.0009, 0.0221, ..., 0.0021, 0.0235, 0.2307]],
                                                                        #
                                                                        #          [[0.0839, 0.0138, 0.0385, ..., 0.0045, 0.0395, 0.0097],
                                                                        #           [0.0138, 0.0268, 0.0177, ..., 0.0204, 0.0284, 0.0009],
                                                                        #           [0.0385, 0.0177, 0.0624, ..., 0.0141, 0.0613, 0...

    for e in range(args.epochs):
        transformer.train()     # 训练本文提出的transformer模型
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):    # (x, _)表示data，其中x表示？
            n_batch = len(x)                                # n_batch到底表示什么？猜测刚好等于batch_size
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)                # x是数据集中的原始图片（已经转化为Tensor了）
            y = transformer(x)              # y是x经过transform网络后的结果

            y = utils.normalize_batch(y)    # y归一化
            x = utils.normalize_batch(x)    # x归一化

            features_y = vgg(y)
            features_x = vgg(x)

            # 计算contentloss，x和y分别经过预训练模型vgg后的feature之间的loss
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # 计算styleloss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])  # gm_s[:n_batch, :, :]中的索引":n_batch"，表示取从以一个元素到索引为n_batch的元素（不包括n_batch）
            style_loss *= args.style_weight

            # 计算totalloss
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",  # mosaic.jpg大小为470*391
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
