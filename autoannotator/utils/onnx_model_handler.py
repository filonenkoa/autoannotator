import gc
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import onnx
import onnxruntime as ort

from autoannotator.types.base import Device

ort.set_default_logger_severity(3)

@dataclass
class _BatchMeta:
    dynamic: bool
    fixed_size: Optional[int]


class OnnxModelHandler:
    def __init__(self, model_path: str, device: str | Device = "cpu"):
        self.model_path = model_path
        runtime_device = self._normalize_device(device)
        self.device = runtime_device
        self.start_session(runtime_device)
        
    def __call__(self, tensor: np.ndarray) -> Any:
        return self.forward(tensor)

    def start_session(self, device: Device):
        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)
        self.input_name = self.onnx_model.graph.input[0].name
        self.input_ndim = len(
            self.onnx_model.graph.input[0].type.tensor_type.shape.dim
        )
        self._batch_meta = self._detect_batch_support()
        match device:
            case Device.CUDA:
                self.ort_sess = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
            case Device.CPU:
                self.ort_sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            case Device.RT:
                self.ort_sess = ort.InferenceSession(self.model_path, providers=['TensorrtExecutionProvider'])
            case _:
                raise Exception(f"Unknown device {device}")

    def stop_session(self):
        del self.ort_sess
        del self.onnx_model
        gc.collect()

    def get_input_shape(self) -> List[int]:
        shape = []
        for val in self.onnx_model.graph.input[0].type.tensor_type.shape.dim[1:]:
            shape.append(val.dim_value)
        return shape

    def get_embedding_size(self) -> int:
        return self.onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value

    @property
    def supports_batch(self) -> bool:
        return self._batch_meta.dynamic or (self._batch_meta.fixed_size is not None and self._batch_meta.fixed_size > 1)

    def forward(self, x: np.ndarray):
        if x.ndim == self.input_ndim - 1:
            x = np.expand_dims(x, axis=0)

        batch_size = x.shape[0] if x.ndim == self.input_ndim else 1

        if batch_size > 1:
            if not self.supports_batch:
                outputs = []
                for i in range(batch_size):
                    outputs.append(
                        self.ort_sess.run(None, {self.input_name: x[i:i + 1]})[0]
                    )

                sample = outputs[0]
                if sample.ndim >= 1 and sample.shape[0] == 1:
                    return np.concatenate(outputs, axis=0)

                return np.stack(outputs, axis=0)

            if self._batch_meta.fixed_size is not None and batch_size != self._batch_meta.fixed_size:
                raise ValueError(
                    f"ONNX model expects batch size {self._batch_meta.fixed_size}, received {batch_size}."
                )

        return self.ort_sess.run(None, {self.input_name: x})[0]

    def _detect_batch_support(self) -> _BatchMeta:
        input_shape = self.onnx_model.graph.input[0].type.tensor_type.shape.dim
        if not input_shape:
            return _BatchMeta(dynamic=False, fixed_size=None)

        batch_dim = input_shape[0]
        if batch_dim.dim_param:
            return _BatchMeta(dynamic=True, fixed_size=None)

        batch_value = batch_dim.dim_value
        if batch_value in [0, None]:
            return _BatchMeta(dynamic=True, fixed_size=None)

        return _BatchMeta(dynamic=False, fixed_size=batch_value)

    @staticmethod
    def _normalize_device(device: str | Device) -> Device:
        if isinstance(device, Device):
            return device

        device_name = device.lower()
        match device_name:
            case "cpu":
                return Device.CPU
            case "cuda" | "gpu":
                return Device.CUDA
            case "rt":
                return Device.RT
            case _:
                raise ValueError(f"Unknown device type {device}")
