import os
from typing import Literal
import torch as th

from .generators import Generator
from .img_procs import ImgProcBase


class Autoencoder(th.nn.Module):

    enc: Generator
    dec: Generator
    img_proc: ImgProcBase

    def __init__(self, enc: Generator, dec: Generator,
                 img_proc: ImgProcBase) -> None:
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.img_proc = img_proc

    def forward(self, inputs: th.Tensor, mode: Literal["encode",
                                                       "decode"]) -> th.Tensor:
        if mode == "encode":
            return self.encode(inputs)
        elif mode == "decode":
            return self.decode(inputs)
        raise ValueError(f"invalid mode: {mode}")

    def encode(self, inputs: th.Tensor) -> th.Tensor:
        nnet_ins: th.Tensor = self.img_proc.imgs_to_nnet_inputs(inputs)
        codes: th.Tensor = self.enc(nnet_ins)
        return codes

    def decode(self, codes: th.Tensor) -> th.Tensor:
        nnet_outs: th.Tensor = self.dec(codes)
        imgs: th.Tensor = self.img_proc.nnet_outputs_to_imgs(nnet_outs)
        return imgs

    def to_onnx(self,
                export_dir: str,
                inputs: th.Tensor,
                encoder_fname: str = "encoder.onnx",
                decoder_fname: str = "decoder.onnx") -> None:
        self.eval()
        codes: th.Tensor = self.encode(inputs)
        os.makedirs(export_dir, exist_ok=True)
        th.onnx.export(self, (inputs, "encode"),
                       os.path.join(export_dir, encoder_fname),
                       export_params=True,
                       do_constant_folding=True,
                       input_names=["inputs"],
                       output_names=["outputs"],
                       dynamic_axes={
                           "inputs": {
                               0: "batch_size"
                           },
                           "outputs": {
                               0: "batch_size"
                           }
                       })
        th.onnx.export(self, (codes, "decode"),
                       os.path.join(export_dir, decoder_fname),
                       export_params=True,
                       do_constant_folding=True,
                       input_names=["inputs"],
                       output_names=["outputs"],
                       dynamic_axes={
                           "inputs": {
                               0: "batch_size"
                           },
                           "outputs": {
                               0: "batch_size"
                           }
                       })
