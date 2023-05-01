import sys
import torch
from flexflow.core import FFConfig, FFModel, DataType
from flexflow.torch.model import PyTorchModel
from torch.fx.graph_module import GraphModule
# sys.path.append(Path(".").absolute().as_posix())
sys.path.append("/storage")
from examples.python.pytorch.bert_fs import *
from examples.python.pytorch.customed_tracer import *
model = BERT()

if 1==0:
    name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    tracer = CustomedTracer()
    graph = tracer.trace(model)
    traced = GraphModule(tracer.root, graph, name)

    sym_graph = torch.fx.symbolic_trace(model)



# fx.torch_to_flexflow(model, "mymodel.ff")
ffconfig = FFConfig()
ffmodel = FFModel(ffconfig)

# batch = make_batch()
traced = PyTorchModel(model, is_hf_model=False)
# ,input_names=input_names, batch_size=batch_size)
output_tensors = traced.torch_to_ff(ffmodel, input_tensors, verbose=True)

