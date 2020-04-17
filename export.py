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
from utils.augmentations import FastBaseTransform


from torch.onnx.utils import _model_to_graph
from torch.onnx import OperatorExportTypes

from pytorch2sg.graph import PytorchGraph
from pytorch2sg.engine import PytorchEngine



def load_model(model_file):
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  set_cfg('yolact_plus_resnet50_config')
  net = Yolact()
  net.load_weights(model_file)
  net.eval()
  return net

def run_torch(net):
  

def run_with_runtime(sg_file, img):
  # run runtime
  rb_net = rt.Network(sg_file, dp=[rt.GPU])
  outputs = rb_net.Run(img.cpu().detach().numpy().astype(np.float32))
  return outputs

  print(outputs.keys())

  print(np.max(outputs['1995'] - preds['loc'].cpu().detach().numpy()))
  print(np.max(outputs['2005'] - preds['conf'].cpu().detach().numpy()))
  print(np.max(outputs['2001'] - preds['mask'].cpu().detach().numpy()))

  print(outputs['1995'].reshape(-1)[20:40])
  print( preds['loc'].cpu().detach().numpy().reshape(-1)[20:40])


def parser_with_view(net):
  graph, _, _ = _model_to_graph(net, (torch.randn(1, 3, 550, 550).cuda(),), operator_export_type=OperatorExportTypes.RAW)
  # print(graph)
  engine = PytorchEngine()
  g = PytorchGraph()
  g.load(net, input_shape=[1, 3, 550, 550])
  g = engine.run(g=g, view=True, port=5000, view_rules=[])

def export_sg(net):
  # generate sg
  sg = rb.gen_sg_from_pytorch(net, input_shape=[1, 3, 550, 550])
  rb.save_sg(sg, 'yolact_plus.sg')


if __name__ == '__main__':
  net = load_model(sys.argv[1])

  # # run torch  
  # img_path = './data/example.jpg'
  # img_1 = torch.from_numpy(cv2.imread(img_path)).cuda().float()
  # img = FastBaseTransform()(img_1.unsqueeze(0))
  # preds = net(img)
  # # output order is %1995, %2005, %2001, %2002, %1632
  # #print(preds.keys())
  # #print(preds['loc'].shape)
  # #print(preds['conf'].shape)
  # #print(preds['mask'].shape)

  # parser_with_view(net)
  # export_sg(net)

  
  

  