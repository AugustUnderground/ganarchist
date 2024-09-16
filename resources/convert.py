import torch as pt
import onnx

m = pt.jit.load('../models/20240914-183529/trace.pt')
x = pt.rand([10,1])
pt.onnx.export(m, x, "../models/20240914-183529/graph.onnx", verbose=True)

n = onnx.load("../models/20240914-183529/graph.onnx")
onnx.checker.check_model(n)
onnx.helper.printable_graph(n.graph)
