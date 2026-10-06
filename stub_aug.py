"""Reemplaza transforms IAA de entrenamiento por stubs (no se usan en exportacion ONNX)."""
import sys

target = sys.argv[1] if len(sys.argv) > 1 else 'carve-lama/saicinpainting/training/data/aug.py'
stub = '''"""Stub: transforms de entrenamiento IAA no se usan en la exportacion ONNX."""
from albumentations.core.transforms_interface import DualTransform


class IAAAffine2(DualTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(p=1.0)


class IAAPerspective2(DualTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(p=1.0)
'''
open(target, 'w', encoding='utf-8').write(stub)
print('stub escrito en', target)
