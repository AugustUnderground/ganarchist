import torch as pt
import onnx
import onnxruntime as ort

t = '20240918-155630'
b = '../models'
p = f'{b}/{t}'
m = pt.jit.load(f'{p}/trace.pt')

v = pt.linspace(50,350,10)
i = pt.linspace(10,100,10)
x = pt.cartesian_prod(v,i)

m(x)

pt.onnx.export(m, x, f'{p}/graph.onnx', verbose=True)

n = onnx.load(f'{p}/graph.onnx')
onnx.checker.check_model(n)
onnx.helper.printable_graph(n.graph)
r = ort.InferenceSession(f'{p}/graph.onnx')

assert pt.all(m(x) == pt.from_numpy(r.run(None, {r.get_inputs()[0].name : x.numpy()})[0]))

err = pt.abs(m(x) - pt.from_numpy(r.run(None, {r.get_inputs()[0].name : x.numpy()})[0]))
