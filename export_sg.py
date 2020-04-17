import sys
import cv2
import rbcompiler.api_v2 as rb
import pyRbRuntime as rt
import numpy as np
# register pyruntime op
import sg.dcn 

import torch
from yolact import Yolact
from data import set_cfg


def load_model(model_file):
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  set_cfg('yolact_plus_resnet50_config')
  net = Yolact()
  net.load_weights(model_file)
  net.eval()
  return net

def export_sg(net):
  # generate sg
  sg = rb.gen_sg_from_pytorch(net, input_shape=[1, 3, 550, 550])
  rb.save_sg(sg, 'yolact_plus.sg')


if __name__ == '__main__':
  net = load_model(sys.argv[1])
  export_sg(net)