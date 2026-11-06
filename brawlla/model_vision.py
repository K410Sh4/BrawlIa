"""Detecção ONNX opcional, apenas para avaliação de vídeo offline.

Implementação independente: não inclui pesos nem código de outros projetos.
Nenhum resultado não observado é repetido como se fosse uma nova detecção.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelSchedule:
    """Executa o modelo a cada N quadros; os intermediários ficam não observados."""

    every_n_frames: int = 3

    def __post_init__(self):
        if self.every_n_frames < 1:
            raise ValueError("every_n_frames deve ser >= 1")

    def should_infer(self, frame_index: int) -> bool:
        if frame_index < 0:
            raise ValueError("frame_index deve ser >= 0")
        return frame_index % self.every_n_frames == 0


def letterbox(frame_bgr, width, height):
    """Preserva proporções e retorna tensor NCHW, escala e margens (x, y)."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        raise ValueError("Frame BGR inválido")
    if frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("Frame BGR deve ser uint8 com três canais")
    if width < 1 or height < 1:
        raise ValueError("Dimensões inválidas")
    original_h, original_w = frame_bgr.shape[:2]
    if original_w < 1 or original_h < 1:
        raise ValueError("Frame vazio")
    scale = min(width / original_w, height / original_h)
    scaled_w = min(width, max(1, round(original_w * scale)))
    scaled_h = min(height, max(1, round(original_h * scale)))
    resized = cv2.resize(frame_bgr, (scaled_w, scaled_h))
    pad_x = (width - scaled_w) // 2
    pad_y = (height - scaled_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        height - scaled_h - pad_y,
        pad_x,
        width - scaled_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = (
        cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255.0
    )
    return np.ascontiguousarray(tensor), scale, (pad_x, pad_y)


def _nms(boxes, scores, threshold):
    """NMS determinístico em coordenadas XYXY, devolvendo índices."""
    if len(scores) == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    kept = []
    while order.size:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break
        others = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[others, 0])
        y1 = np.maximum(boxes[i, 1], boxes[others, 1])
        x2 = np.minimum(boxes[i, 2], boxes[others, 2])
        y2 = np.minimum(boxes[i, 3], boxes[others, 3])
        overlap = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        a = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0, boxes[:, 3] - boxes[:, 1]
        )
        union = a[i] + a[others] - overlap
        iou = overlap / np.maximum(union, 1e-6)
        order = others[iou <= threshold]
    return kept


def decode_yolo(
    output,
    original_shape,
    scale,
    padding,
    classes=(),
    confidence=0.35,
    iou=0.5,
    max_detections=100,
    export_format="yolov8",
):
    """Decodifica exportações comuns YOLOv8 (XYWH) ou end-to-end (XYXY).

    Não atribui nomes arbitrários a classes desconhecidas. As classes são
    fornecidas pelo usuário conforme os metadados do seu próprio modelo.
    """
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("Limiar fora de [0, 1]")
    if export_format not in {"yolov8", "xyxy"}:
        raise ValueError("Formato ONNX não suportado")
    if max_detections < 1:
        raise ValueError("max_detections deve ser positivo")
    data = np.asarray(output)
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("Apenas inferência batch=1 é suportada")
        data = data[0]
    if data.ndim != 2:
        raise ValueError("Tensor YOLO deve ter duas dimensões após batch")
    # Prioriza a contagem conhecida de classes para resolver N×C versus C×N.
    # Nomes ausentes significam que a orientação precisa ser deduzida.
    if export_format == "xyxy":
        if data.shape[1] != 6 and data.shape[0] == 6:
            data = data.T
    elif classes:
        attributes = len(classes) + 4
        if data.shape[1] != attributes and data.shape[0] == attributes:
            data = data.T
    elif data.shape[0] < data.shape[1] and data.shape[0] <= 256:
        data = data.T
    if data.shape[0] == 0:
        return []
    if export_format == "yolov8":
        if data.shape[1] < 5:
            raise ValueError("YOLOv8 exige XYWH e pelo menos uma classe")
        scores = data[:, 4:]
        labels = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), labels]
        centers = data[:, :2]
        sizes = data[:, 2:4]
        boxes = np.column_stack(
            (centers - sizes / 2, centers + sizes / 2)
        )
    else:
        if data.shape[1] != 6:
            raise ValueError("XYXY exige [x1,y1,x2,y2,score,classe]")
        boxes = data[:, :4]
        confs = data[:, 4]
        labels = data[:, 5].astype(np.int64)
    valid = (confs >= confidence) & np.all(np.isfinite(boxes), axis=1)
    valid &= np.isfinite(confs) & (labels >= 0)
    boxes, confs, labels = boxes[valid], confs[valid], labels[valid]
    if not len(boxes):
        return []
    boxes = boxes.astype(np.float64)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / scale
    height, width = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes, confs, labels = boxes[areas > 0], confs[areas > 0], labels[areas > 0]
    selected = []
    for class_id in np.unique(labels):
        subset = np.where(labels == class_id)[0]
        subset = subset[np.argsort(-confs[subset], kind="stable")[:500]]
        selected.extend(subset[k] for k in _nms(boxes[subset], confs[subset], iou))
    selected.sort(key=lambda i: -float(confs[i]))
    result = []
    for i in selected[:max_detections]:
        class_id = int(labels[i])
        label = str(classes[class_id]) if class_id < len(classes) else f"class_{class_id}"
        result.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": round(float(confs[i]), 5),
                "xyxy": [round(float(v), 2) for v in boxes[i]],
            }
        )
    return result


class OnnxVision:
    """Adapter de inferência opcional para modelos ONNX externos, somente leitura."""

    def __init__(
        self,
        model_path,
        classes=(),
        provider="auto",
        input_size=640,
        export_format="yolov8",
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if provider not in {"auto", "cpu", "directml", "cuda"}:
            raise ValueError("Provider deve ser auto, cpu, directml ou cuda")
        if export_format not in {"yolov8", "xyxy"}:
            raise ValueError("Formato ONNX não suportado")
        if input_size < 32:
            raise ValueError("input_size deve ser >= 32")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Instale onnxruntime-directml (Windows AMD) ou onnxruntime (CPU)"
            ) from exc
        available = set(ort.get_available_providers())
        requested = {
            "auto": ("CUDAExecutionProvider", "DmlExecutionProvider"),
            "cpu": (),
            "directml": ("DmlExecutionProvider",),
            "cuda": ("CUDAExecutionProvider",),
        }[provider]
        order = [p for p in requested if p in available]
        order.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=order
        )
        self.provider = self.session.get_providers()[0]
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or shape[1] not in (3, "3", None):
            raise ValueError("Modelo deve receber tensor NCHW RGB")
        self.width = shape[3] if isinstance(shape[3], int) else input_size
        self.height = shape[2] if isinstance(shape[2], int) else input_size
        self.input_name = model_input.name
        self.classes = tuple(classes)
        self.export_format = export_format

    def detect(self, frame_bgr, confidence=0.35, iou=0.5):
        start = time.perf_counter()
        tensor, scale, padding = letterbox(frame_bgr, self.width, self.height)
        outputs = self.session.run(None, {self.input_name: tensor})
        boxes = decode_yolo(
            outputs[0],
            frame_bgr.shape,
            scale,
            padding,
            classes=self.classes,
            confidence=confidence,
            iou=iou,
            export_format=self.export_format,
        )
        return {
            "provider": self.provider,
            "inference_ms": round((time.perf_counter() - start) * 1000, 3),
            "detections": boxes,
        }
