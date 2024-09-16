import torch as pt
import onnx
import onnxruntime as ort

m = pt.jit.load('../models/20240914-183529/trace.pt')
x = pt.rand([10,1])
pt.onnx.export(m, x, "../models/20240914-183529/graph.onnx", verbose=True)

n = onnx.load("../models/20240914-183529/graph.onnx")
onnx.checker.check_model(n)
onnx.helper.printable_graph(n.graph)
r = ort.InferenceSession("../models/20240914-183529/graph.onnx")

assert pt.all(m(x) == pt.from_numpy(r.run(None, {r.get_inputs()[0].name : x.numpy()})[0]))

err = pt.abs(m(x) - pt.from_numpy(r.run(None, {r.get_inputs()[0].name : x.numpy()})[0])) / x
