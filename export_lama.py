"""Exporta big-lama a ONNX a la resolucion indicada.
Uso: python export_lama.py <resolucion> <salida.onnx>
Adaptado del notebook oficial export_LaMa_to_onnx.ipynb de Carve-Photos/lama.

Rutas: busca el repo saicinpainting en ./carve-lama (CI) o ./lama (local)."""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, 'carve-lama') if os.path.isdir(os.path.join(BASE, 'carve-lama')) else os.path.join(BASE, 'lama')
sys.path.insert(0, REPO)

import torch
from omegaconf import OmegaConf
from yaml import safe_load

from saicinpainting.training.trainers.default import (
    DefaultInpaintingTrainingModule,
)


class ExportLama(torch.nn.Module):
    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        masked_img = image * (1 - mask)
        if self.model.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)
        predicted_image = self.model.generator(masked_img)
        inpainted = mask * predicted_image + (1 - mask) * image
        return torch.clamp(inpainted * 255, min=0, max=255)


size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, f'lama_{size}.onnx')

config = OmegaConf.create(safe_load(open(os.path.join(REPO, 'big-lama', 'config.yaml'))))
kwargs = dict(config.training_model)
kwargs.pop('kind')
kwargs['use_ddp'] = True

config.generator.resnet_conv_kwargs.use_jit = True
config.losses.resnet_pl.weight = 0
config.visualizer.outdir = os.path.join(BASE, 'viz_out')
os.makedirs(config.visualizer.outdir, exist_ok=True)

print('cargando checkpoint...')
state = torch.load(os.path.join(REPO, 'big-lama', 'models', 'best.ckpt'), map_location='cpu', weights_only=False)
lama = DefaultInpaintingTrainingModule(config, **kwargs)
lama.load_state_dict(state['state_dict'], strict=False)
lama.on_load_checkpoint(state)
lama.freeze()
lama.eval()

exported = ExportLama()
exported.register_module('model', lama)
exported.eval()
exported.to('cpu')

print(f'exportando a {size}x{size} -> {out}')
torch.onnx.export(
    exported,
    (
        torch.rand(1, 3, size, size).type(torch.float32).to('cpu'),
        torch.rand(1, 1, size, size).type(torch.float32).to('cpu'),
    ),
    out,
    input_names=['image', 'mask'],
    output_names=['output'],
    dynamic_axes={
        'image': {0: 'batch'},
        'mask': {0: 'batch'},
        'output': {0: 'batch'},
    },
    export_params=True,
    do_constant_folding=True,
    opset_version=17,
    verbose=False,
)
print('OK:', out, f'({os.path.getsize(out)/1024/1024:.1f} MB)')
