"""Verifica que los ONNX exportados cargan y reporta formas."""
import glob

import onnxruntime as ort

for p in sorted(glob.glob('lama_*.onnx')):
    s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    print(p, 'IN', s.get_inputs()[0].shape, 'OUT', s.get_outputs()[0].shape)
