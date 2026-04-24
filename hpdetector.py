import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional


_C = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "player":  "\033[38;2;80;220;80m",    # зелёный
    "enemy":   "\033[38;2;220;70;70m",    # красный
    "team":    "\033[38;2;80;160;255m",   # синий
    "bar_fill":"\033[38;2;255;210;50m",   # жёлтый (заполнение)
    "bar_empty":"\033[38;2;60;60;60m",    # тёмный (пустота)
    "gray":    "\033[38;2;150;150;150m",
    "white":   "\033[38;2;240;240;240m",
    "orange":  "\033[38;2;255;140;30m",
    "header":  "\033[38;2;200;200;255m",
}

CLASS_NAMES  = ["player", "enemy", "team"]
CLASS_COLORS = {"player": _C["player"], "enemy": _C["enemy"], "team": _C["team"]}
CLASS_ICONS  = {"player": "★", "enemy": "✖", "team": "●"}

BAR_WIDTH    = 20   # символов для ASCII hp-bar


@dataclass
class HPBar:
    class_id:   int
    class_name: str
    confidence: float
    hp_pct:     Optional[float]       # 0.0 – 100.0, None если не удалось
    box_img:    List[int]             # [x1, y1, x2, y2] в пикселях оригинала
    crop:       Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self):
        hp = f"{self.hp_pct:.1f}%" if self.hp_pct is not None else "N/A"
        return f"HPBar({self.class_name}, hp={hp}, conf={self.confidence:.2f})"

class HPdetectBYjousj:
    INPUT_SIZE     = 640
    CONF_THRESHOLD = 0.35
    NMS_IOU        = 0.45

    def __init__(self, model_path: str, preferred_device: str = "auto"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[HPdetectBYjousj] Модель не найдена: {model_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3   # 3=error only, 4=fatal only

        available = ort.get_available_providers()
        candidates = []
        if preferred_device in ("gpu", "auto"):
            if "CUDAExecutionProvider" in available:
                candidates.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available:
                candidates.append("DmlExecutionProvider")
        candidates.append("CPUExecutionProvider")

        self._session = None
        for provider in candidates:
            try:
                sess = ort.InferenceSession(model_path, sess_options=so, providers=[provider])
                dummy = np.zeros((1, 3, self.INPUT_SIZE, self.INPUT_SIZE), dtype=np.float32)
                sess.run(None, {sess.get_inputs()[0].name: dummy})
                self._session = sess
                break
            except Exception:
                continue

        if self._session is None:
            raise RuntimeError("[HPdetectBYjousj] Не удалось инициализировать модель ни на одном провайдере")

        self._input_name   = self._session.get_inputs()[0].name
        self._last_results: List[HPBar] = []
        self._last_ts: float = 0.0

    def detect(self, image_source) -> List[HPBar]:
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Не удалось прочитать: {image_source}")
        else:
            img = image_source

        orig_h, orig_w = img.shape[:2]
        tensor, sx, sy  = self._preprocess(img)
        raw              = self._session.run(None, {self._input_name: tensor})[0]
        detections       = self._postprocess(raw, orig_w, orig_h, sx, sy)

        results: List[HPBar] = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            crop   = img[y1:y2, x1:x2]
            hp_pct = self._calculate_hp_percent(crop)
            results.append(HPBar(
                class_id   = int(cls_id),
                class_name = CLASS_NAMES[int(cls_id)],
                confidence = float(conf),
                hp_pct     = hp_pct,
                box_img    = [x1, y1, x2, y2],
                crop       = crop,
            ))

        results.sort(key=lambda r: r.confidence, reverse=True)
        self._last_results = results
        self._last_ts      = time.time()
        return results

    def print_console(self, results: Optional[List[HPBar]] = None,
                      show_header: bool = True) -> None:
        if results is None:
            results = self._last_results

        if show_header:
            _print_hp_header()

        if not results:
            print(f"  {_C['gray']}нет детекций{_C['reset']}")
            return

        order = {"player": 0, "team": 1, "enemy": 2}
        sorted_res = sorted(results, key=lambda r: (order.get(r.class_name, 9), -r.confidence))

        for bar in sorted_res:
            _print_hp_bar_line(bar)

    def get_player_hp(self, results: Optional[List[HPBar]] = None) -> Optional[float]:
        if results is None:
            results = self._last_results
        players = [r for r in results if r.class_name == "player"]
        if not players:
            return None
        return max(players, key=lambda r: r.confidence).hp_pct

    def get_enemies_hp(self, results: Optional[List[HPBar]] = None) -> List[Optional[float]]:
        if results is None:
            results = self._last_results
        return [r.hp_pct for r in results if r.class_name == "enemy"]

    def _preprocess(self, img: np.ndarray):
        oh, ow  = img.shape[:2]
        scale   = min(self.INPUT_SIZE / oh, self.INPUT_SIZE / ow)
        new_w   = int(ow * scale)
        new_h   = int(oh * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        buf = np.full((1, 3, self.INPUT_SIZE, self.INPUT_SIZE), 128.0 / 255.0, dtype=np.float32)
        buf[0, :, :new_h, :new_w] = np.transpose(
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            (2, 0, 1)
        )

        sx = ow / new_w
        sy = oh / new_h
        return buf, sx, sy

    def _postprocess(self, output: np.ndarray,
                     orig_w: int, orig_h: int,
                     sx: float, sy: float):
        preds        = output[0].T          # (8400, 7)
        boxes_cxcy   = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids    = np.argmax(class_scores, axis=1)
        confidences  = np.max(class_scores, axis=1)
        mask         = confidences >= self.CONF_THRESHOLD
        boxes_cxcy   = boxes_cxcy[mask]
        confidences  = confidences[mask]
        class_ids    = class_ids[mask]

        if len(boxes_cxcy) == 0:
            return []

        cx = boxes_cxcy[:, 0] * sx
        cy = boxes_cxcy[:, 1] * sy
        bw = boxes_cxcy[:, 2] * sx
        bh = boxes_cxcy[:, 3] * sy

        x1 = np.clip(cx - bw / 2, 0, orig_w - 1).astype(int)
        y1 = np.clip(cy - bh / 2, 0, orig_h - 1).astype(int)
        x2 = np.clip(cx + bw / 2, 0, orig_w - 1).astype(int)
        y2 = np.clip(cy + bh / 2, 0, orig_h - 1).astype(int)

        boxes_list = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])]
                      for i in range(len(x1))]
        keep = cv2.dnn.NMSBoxes(boxes_list, confidences.tolist(),
                                 self.CONF_THRESHOLD, self.NMS_IOU)

        results = []
        if len(keep) > 0:
            for i in keep.flatten():
                results.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
                                 float(confidences[i]), int(class_ids[i])])
        return results

    @staticmethod
    def _is_empty_bg(h: int, s: int, v: int) -> bool:
        return (90 <= h <= 140) and (s > 40) and (40 <= v <= 145)

    @staticmethod
    def _is_border(v: int) -> bool:
        return v < 40

    def _calculate_hp_percent(self, crop: np.ndarray) -> Optional[float]:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if w < 10 or h < 3:
            return None

        trim  = max(2, w // 15)
        inner = crop[:, trim: w - trim]
        iw    = inner.shape[1]
        if iw < 4:
            return None

        y0 = h // 4
        y1 = max(y0 + 1, 3 * h // 4)
        hsv = cv2.cvtColor(inner[y0:y1, :], cv2.COLOR_BGR2HSV)

        filled = empty = 0
        for col in range(iw):
            col_hsv = hsv[:, col, :]
            hv = int(np.mean(col_hsv[:, 0]))
            sv = int(np.mean(col_hsv[:, 1]))
            vv = int(np.mean(col_hsv[:, 2]))

            if self._is_border(vv):
                continue
            elif self._is_empty_bg(hv, sv, vv):
                empty += 1
            else:
                filled += 1

        total = filled + empty
        if total == 0:
            return None
        return float(np.clip(filled / total * 100.0, 0.0, 100.0))

def _ascii_bar(pct: Optional[float], width: int = BAR_WIDTH) -> str:
    if pct is None:
        return _C["gray"] + "─" * width + _C["reset"]
    filled = round(pct / 100 * width)
    empty  = width - filled

    if pct > 60:
        fill_c = "\033[38;2;80;220;80m"    # зелёный
    elif pct > 30:
        fill_c = "\033[38;2;255;180;30m"   # оранжевый
    else:
        fill_c = "\033[38;2;220;50;50m"    # красный

    bar  = fill_c + "█" * filled
    bar += _C["bar_empty"] + "░" * empty
    bar += _C["reset"]
    return bar


def _print_hp_header() -> None:
    line = f"{_C['header']}{_C['bold']}{'─'*52}{_C['reset']}"
    print(line)
    print(f"  {_C['header']}{_C['bold']}HP MONITOR  [HPdetectBYjousj]{_C['reset']}")
    print(line)


def _print_hp_bar_line(bar: HPBar) -> None:
    col   = CLASS_COLORS.get(bar.class_name, _C["white"])
    icon  = CLASS_ICONS.get(bar.class_name, "?")
    label = f"{col}{_C['bold']}{icon} {bar.class_name:<7}{_C['reset']}"

    hp_str = f"{bar.hp_pct:5.1f}%" if bar.hp_pct is not None else "  N/A "
    if bar.hp_pct is not None:
        if bar.hp_pct > 60:
            hp_c = "\033[38;2;80;220;80m"
        elif bar.hp_pct > 30:
            hp_c = "\033[38;2;255;180;30m"
        else:
            hp_c = "\033[38;2;220;50;50m"
    else:
        hp_c = _C["gray"]

    ascii_b = _ascii_bar(bar.hp_pct)
    conf_s  = f"{_C['dim']}{bar.confidence:.2f}{_C['reset']}"

    print(f"  {label}  {ascii_b}  {hp_c}{hp_str}{_C['reset']}  {conf_s}")