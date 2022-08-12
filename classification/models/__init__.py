from .vgg import *
from .resnet import *
from .resnet112 import resnet18x112
from .resnet50_scratch_dims_2048 import resnet50_pretrained_vgg
from .centerloss_resnet import resnet18_centerloss
from .resatt import *
from .alexnet import *
from .densenet import *
from .googlenet import *
from .inception import *
from .inception_resnet_v1 import *
from .residual_attention_network import *
from .fer2013_models import *
from .res_dense_gle import *
from .masking import masking
# from .ViT import *
# from .cait import Cait
# from .models.beit import Beit
from .resmasking import (
    resmasking,
    resmasking_dropout1,
    resmasking_dropout2,
    resmasking50_dropout1,
    resmasking50
)
from .resmasking_naive import resmasking_naive_dropout1
from .brain_humor import *
from .runet import *
from pytorchcv.model_provider import get_model as ptcv_get_model


# from .ConvNeXt import *
# from .swintransformer import SwinTransformer
# from .coat import CoaT

# def swint(**kwargs):
#     model = SwinTransformer(**kwargs)
#     # patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16,
#     return model


# def cait(**kwargs):
#     model = Cait(**kwargs)
#     # patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16,
#     return model
#
# def beit(**kwargs):
#     model = Beit(**kwargs)
#     # patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16,
#     return model


# def coat(**kwargs):
#     model = CoaT(**kwargs)
#     # patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16,
#     return model


def resnext101(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("resnext101_64x4d", pretrained=False)
    model.output = nn.Linear(2048, num_classes)
    return model


def pyramid(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("pyramidnet236_a220_bn_cifar10", pretrained=False)
    model.output = nn.Linear(2048, num_classes)
    return model


def cbam_resnet152(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("cbam_resnet152", pretrained=False)
    model.output = nn.Linear(2048, num_classes)
    return model


def cbam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("cbam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def fishnet150(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("fishnet150", pretrained=False)
    model.num_classes = num_classes
    model.output = nn.Sequential()
    model.output.add_module("final_conv", conv1x1(in_channels=1056, out_channels=num_classes))
    # model.output = nn.Linear(1056, 4, bias=True)
    return model


def dla102x2(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("dla102x2", pretrained=True)
    model.in_channels = 1024
    model.num_classes = num_classes
    model.output = conv1x1(in_channels=model.in_channels, out_channels=num_classes)
    return model


def seresnext50_32x4d(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("seresnext50_32x4d", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def regnety032(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("regnety032", pretrained=True)
    model.output = nn.Linear(1512, num_classes)
    return model


def scneta50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("scneta50", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def resnesta200(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("resnesta200", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def bam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("bam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def hrnetv2_w32(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("hrnetv2_w32", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def efficientnet_b3(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b3", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1536, num_classes))
    return model


def efficientnet_b3b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b3b", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1536, num_classes))
    return model


def efficientnet_b2c(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b2c", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1408, num_classes))
    return model


def efficientnet_b3c(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b3c", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1536, num_classes))
    return model


def efficientnet_b4c(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b4c", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1792, num_classes))
    return model


def efficientnet_b6c(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b6c", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(2304, num_classes))
    return model


def efficientnet_edge_medium_b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_edge_medium_b", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1280, num_classes))
    return model


def efficientnet_edge_large_b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_edge_large_b", pretrained=False)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(1536, num_classes))
    return model
