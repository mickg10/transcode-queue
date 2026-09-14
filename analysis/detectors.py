"""Anonymous person/face bounds for editorial framing, without face recognition.

YOLOX tensor decoding follows the model's documented stride/grid layout. Models
are obtained separately from OpenCV Zoo and are not stored in this repository.
"""
from pathlib import Path

import cv2
import numpy as np


class Performers:
    def __init__(self, models: Path, confidence=.3, threads=4, gpu=False):
        cv2.setNumThreads(threads)
        self.session = None
        model = str(models / "object_detection_yolox_2022nov.onnx")
        if gpu:
            import torch  # Load the installed CUDA/cuDNN libraries before ONNX Runtime.
            import onnxruntime as ort
            if not torch.cuda.is_available():
                raise RuntimeError("GPU inference was requested but CUDA is unavailable")
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            self.session = ort.InferenceSession(model, options, providers=[
                ("CUDAExecutionProvider", {"gpu_mem_limit": 1024**3,
                                          "cudnn_conv_algo_search": "HEURISTIC",
                                          "cudnn_conv_use_max_workspace": "0"})])
            if "CUDAExecutionProvider" not in self.session.get_providers():
                raise RuntimeError("ONNX Runtime could not initialize the CUDA provider")
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.net = cv2.dnn.readNet(model)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.face = cv2.FaceDetectorYN.create(
            str(models / "face_detection_yunet_2023mar.onnx"), "", (1152, 768), .65, .3, 5000)
        self.confidence = confidence
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[:640 // stride, :640 // stride]
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full((grids[-1].shape[0], 1), stride))
        self.grids, self.strides = np.concatenate(grids), np.concatenate(strides)

    def detect(self, frame):
        h, w = frame.shape[:2]
        ratio = min(640 / h, 640 / w)
        resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             (int(w * ratio), int(h * ratio)))
        canvas = np.full((640, 640, 3), 114, np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
        if self.session is not None:
            raw = self.session.run(None, {self.input_name: tensor})[0][0].copy()
        else:
            self.net.setInput(tensor)
            raw = self.net.forward()[0].copy()
        raw[:, :2] = (raw[:, :2] + self.grids) * self.strides
        raw[:, 2:4] = np.exp(raw[:, 2:4]) * self.strides
        boxes = raw[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:4] / 2
        person_scores = raw[:, 4] * raw[:, 5]  # COCO class 0: person
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), person_scores.tolist(), self.confidence, .5)
        people = []
        for i in np.asarray(keep).reshape(-1):
            x, y, bw, bh = boxes[i] / ratio
            people.append({"box": [float(x / w), float(y / h), float((x + bw) / w),
                                    float((y + bh) / h)], "confidence": float(person_scores[i])})
        self.face.setInputSize((w, h))
        _, found = self.face.detect(frame)
        faces = [] if found is None else [
            {"box": [float(x[0] / w), float(x[1] / h), float((x[0] + x[2]) / w),
                     float((x[1] + x[3]) / h)], "confidence": float(x[-1])}
            for x in found]
        return {"people": people, "faces": faces}


def stage_candidates(detections, *, maximum_head_y=.60, minimum_height=.06):
    """Retain plausible stage performers; return rejected bounds for review too.

    These scene-specific gates need calibration against the footage. They do not
    establish that every performer was detected; missing/occluded people require
    a wider shot and manual review.
    """
    accepted, rejected = [], []
    for person in detections["people"]:
        x1, y1, x2, y2 = person["box"]
        (accepted if y1 < maximum_head_y and y2 - y1 >= minimum_height else rejected).append(person)
    return accepted, rejected
