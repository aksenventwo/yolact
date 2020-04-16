import sys
import cv2
import rbcompiler.api_v2 as rb
import pyRbRuntime as rt
import numpy as np
import sg.dcn 

import torch
import torch.backends.cudnn as cudnn
from yolact import Yolact
from data import cfg, set_cfg, set_dataset
from utils.augmentations import BaseTransform, FastBaseTransform, Resize

from torch.onnx.utils import _model_to_graph
from torch.onnx import OperatorExportTypes


from pytorch2sg.graph import PytorchGraph
from pytorch2sg.engine import PytorchEngine




if __name__ == '__main__':
  #cudnn.fastest = True
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  set_cfg('yolact_plus_resnet50_config')
  net = Yolact()
  net.load_weights(sys.argv[1])
  net.eval()

  # # run torch  
  # img_path = './data/example.jpg'
  # img = torch.from_numpy(cv2.imread(img_path)).cuda().float()
  # img = FastBaseTransform()(img.unsqueeze(0))
  # preds = net(img)
  # print(preds.keys())

  # # run runtime
  # rb_net = rt.Network('./sg/yolact_plus.sg', dp=[rt.CPU])
  # outputs = rb_net.Run(img.cpu().detach().numpy().astype(np.float32))
  # print(outputs.keys())

  # parser
  graph, params_dict, torch_out = _model_to_graph(net, (torch.randn(1, 3, 550, 550).cuda(),), operator_export_type=OperatorExportTypes.RAW)
  print(graph)
  # engine = PytorchEngine()
  # g = PytorchGraph()
  # g.load(net, input_shape=[1, 3, 550, 550])
  # g = engine.run(g=g, view=True, port=5000, view_rules=[])

  # # generate sg
  # sg = rb.gen_sg_from_pytorch(net, input_shape=[1, 3, 550, 550])
  # rb.save_sg(sg, 'yolact_plus.sg')
