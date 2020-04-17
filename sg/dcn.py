import torch
import numpy as np
import pyRbRuntime as rt
import pyRbRuntime.nn as nn

from dcn_v2 import dcn_v2_conv


class CustomDCNv2(nn.Module):

  def __init__(self, def_):
    super().__init__(def_)

  def run(self, inputs, node_def):
    x = torch.from_numpy(inputs[0].numpy()).cuda().float()
    offset = torch.from_numpy(inputs[1].numpy()).cuda().float()
    mask = torch.from_numpy(inputs[2].numpy()).cuda().float()
    weight = torch.from_numpy(inputs[3].numpy()).cuda().float()
    bias = torch.from_numpy(inputs[4].numpy()).cuda().float()
    stride = tuple(node_def.i_list('arg_0'))
    padding = tuple(node_def.i_list('arg_1'))
    dilation = tuple(node_def.i_list('arg_2'))
    deformable_groups = int(node_def.i('arg_3'))
    output = dcn_v2_conv(x, offset, mask,
                         weight, bias,
                         stride,
                         padding,
                         dilation,
                         deformable_groups)
    output = output.cpu().detach().numpy()

    return rt.Tensor(output)



custom_dcnv2 = nn.ModuleDefBuilder("Custom_DCNv2") \
                 .Device(rt.CPU) \
                 .TypeConstraint(rt.T_FLOAT) \
                 .DeviceArg("x", rt.T_FLOAT) \
                 .DeviceArg("offset", rt.T_FLOAT) \
                 .DeviceArg("mask", rt.T_FLOAT) \
                 .DeviceArg("weight", rt.T_FLOAT) \
                 .DeviceArg("bias", rt.T_FLOAT) \
                 .Build()


nn.RegisterOp(CustomDCNv2(custom_dcnv2))