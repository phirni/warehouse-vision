"""A fake YOLO model so pipeline/API tests run without torch or weights."""
import numpy as np


class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return _T(self.a.astype(int))
    def tolist(self): return self.a.tolist()


class _Boxes:
    def __init__(self, xyxy, ids, cls, conf):
        self.xyxy, self.id, self.cls, self.conf = _T(xyxy), _T(ids), _T(cls), _T(conf)


class _Res:
    def __init__(self, boxes): self.boxes, self.speed = boxes, {"inference": 1.0}


class FakeModel:
    """One 'person' walking left->right across the frame at constant y."""
    names = {0: "person", 1: "truck"}

    def __init__(self): self.predictor = None; self.n = 0

    def track(self, frame, **kw):
        h, w = frame.shape[:2]
        x = int(self.n * 8 + 20); self.n += 1
        box = [[x - 15, h * 0.6 - 60, x + 15, h * 0.6]]
        return [_Res(_Boxes(box, [1], [0], [0.9]))]
