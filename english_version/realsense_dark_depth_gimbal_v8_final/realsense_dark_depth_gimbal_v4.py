"""
RealSense D435 + YOLO Detection + Depth Distance + optional Face ID + optional DeepFusion low-light enhancement + NEMA17 gimbal

Arduino serial protocol expected:
    V <pan_speed> <tilt_speed>\n
Runtime keys:
    Q -> quit
    S -> stop motors
    D -> toggle dark enhancement ON/OFF without restarting
    A -> auto dark enhancement mode
    R -> raw/no enhancement mode
    E -> cycle RAW -> AUTO -> ALWAYS
    1 -> reverse PAN direction
    2 -> reverse TILT direction
    3 -> reverse both directions

Main additions in this version:
- Every YOLO detection gets a RealSense median-depth distance.
- The tracked target can be selected by class/name/distance.
- Low-light enhancement can use your model.pth + deepfusion_model.py if available.
- If the trained model cannot be loaded, it falls back to a fast GPU gamma enhancer.
"""

import argparse
import importlib.util
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

try:
    import serial
except Exception:
    serial = None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def direction_value(name: str, fallback_sign: float) -> float:
    if name == "normal":
        return 1.0
    if name == "reverse":
        return -1.0
    return 1.0 if fallback_sign >= 0 else -1.0


def direction_text(sign: float) -> str:
    return "NORMAL" if sign >= 0 else "REVERSE"


def signed_text(sign: float) -> str:
    return "+" if sign >= 0 else "-"


def bbox_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-9)


def crop_with_padding(img: np.ndarray, box: np.ndarray, pad_ratio: float = 0.30) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def sample_median_depth(
    depth_m: np.ndarray,
    box: np.ndarray,
    shrink: float = 0.35,
    min_depth: float = 0.15,
    max_depth: float = 8.0,
    min_valid_pixels: int = 20,
) -> Optional[float]:
    """
    Robust distance from a YOLO box.
    Instead of using only the center pixel, it uses the median of the center region.
    This is much more stable with D435 depth holes/noise.
    """
    h_img, w_img = depth_m.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1
    cx1 = int(x1 + w * shrink)
    cx2 = int(x2 - w * shrink)
    cy1 = int(y1 + h * shrink)
    cy2 = int(y2 - h * shrink)

    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    roi = depth_m[cy1:cy2, cx1:cx2]
    valid = roi[(roi > min_depth) & (roi < max_depth)]

    if valid.size < min_valid_pixels:
        return None
    return float(np.median(valid))


def safe_torch_load(path: str, device: str, trust_model_code: bool) -> Any:
    """
    Loads a checkpoint. For state_dict checkpoints, weights_only=True is enough.
    Full torch modules require trust_model_code=True.
    """
    kwargs = {"map_location": device}
    try:
        if trust_model_code:
            return torch.load(path, **kwargs, weights_only=False)
        return torch.load(path, **kwargs, weights_only=True)
    except TypeError:
        return torch.load(path, **kwargs)


def import_python_module(module_path: str):
    spec = importlib.util.spec_from_file_location("dark_arch_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Modül yüklenemedi: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "net", "model", "generator", "enhancer"]:
            val = checkpoint.get(key)
            if isinstance(val, dict):
                return val
        # It may directly be a state_dict.
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_like = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if len(tensor_like) > 0:
                return checkpoint
    return None


def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ["module.", "model.", "net.", "generator.", "enhancer."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


class LowLightEnhancer:
    """
    Low-light enhancer with two backends:
    1) deepfusion: tries to load model.pth + deepfusion_model.py
    2) gamma: fast GPU/CPU adaptive gamma fallback
    """

    def __init__(
        self,
        device: str,
        mode: str = "auto",
        backend: str = "auto",
        dark_threshold: float = 70.0,
        model_path: str = "model.pth",
        module_path: str = "deepfusion_model.py",
        class_name: str = "",
        input_size: int = 0,
        fp16: bool = False,
        trust_model_code: bool = False,
    ):
        self.device = device
        self.mode = mode
        self.backend = "gamma"
        self.dark_threshold = dark_threshold
        self.model_path = model_path
        self.module_path = module_path
        self.class_name = class_name
        self.input_size = input_size
        self.fp16 = fp16 and device.startswith("cuda")
        self.trust_model_code = trust_model_code
        self.model = None
        self.status_text = "gamma"

        wants_deepfusion = backend in ["auto", "deepfusion"]
        if wants_deepfusion and model_path and os.path.exists(model_path):
            ok = self._try_load_deepfusion()
            if ok:
                self.backend = "deepfusion"
                self.status_text = "deepfusion"
            elif backend == "deepfusion":
                print("[WARN] DeepFusion model yüklenemedi. Gamma fallback ile devam ediliyor.")

    def _try_load_deepfusion(self) -> bool:
        try:
            # TorchScript model ise architecture dosyası gerekmez.
            try:
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval().to(self.device)
                if self.fp16:
                    self.model.half()
                print(f"[OK] Low-light TorchScript model yüklendi: {self.model_path}")
                return True
            except Exception:
                pass

            checkpoint = safe_torch_load(self.model_path, self.device, self.trust_model_code)

            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint.eval().to(self.device)
                if self.fp16:
                    self.model.half()
                print(f"[OK] Low-light full PyTorch model yüklendi: {self.model_path}")
                return True

            sd = extract_state_dict(checkpoint)
            if sd is None:
                print("[WARN] model.pth içinde state_dict bulunamadı.")
                return False
            sd = clean_state_dict(sd)

            if not self.module_path or not os.path.exists(self.module_path):
                print(f"[WARN] Architecture dosyası bulunamadı: {self.module_path}")
                return False

            module = import_python_module(self.module_path)
            candidates = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                try:
                    if issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                        candidates.append((name, obj))
                except Exception:
                    pass

            preferred = []
            if self.class_name:
                preferred = [(n, c) for n, c in candidates if n == self.class_name]
            else:
                preferred_names = ["DeepFusionNet", "DeepFusionModel", "DeepFusion", "LowLightNet", "EnhanceNet", "Generator", "UNet"]
                for pn in preferred_names:
                    preferred += [(n, c) for n, c in candidates if n == pn]
                preferred += [(n, c) for n, c in candidates if (n, c) not in preferred]

            best_model = None
            best_name = ""
            best_score = 10**9
            best_report = None

            for name, cls in preferred:
                try:
                    model = cls()
                    report = model.load_state_dict(sd, strict=False)
                    score = len(report.missing_keys) + len(report.unexpected_keys)
                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_name = name
                        best_report = report
                except Exception:
                    continue

            if best_model is None:
                print("[WARN] deepfusion_model.py içinden no-arg nn.Module instantiate edilemedi.")
                return False

            self.model = best_model.eval().to(self.device)
            if self.fp16:
                self.model.half()

            miss = len(best_report.missing_keys) if best_report else 0
            unexp = len(best_report.unexpected_keys) if best_report else 0
            print(f"[OK] Low-light model yüklendi: {best_name} | missing:{miss} unexpected:{unexp}")
            return True
        except Exception as e:
            print(f"[WARN] Low-light model yükleme hatası: {e}")
            return False

    def is_dark(self, bgr: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        p40 = float(np.percentile(gray, 40))
        score = 0.65 * mean_luma + 0.35 * p40
        return score < self.dark_threshold, score

    @staticmethod
    def _pick_tensor(out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for item in out:
                if torch.is_tensor(item):
                    return item
        if isinstance(out, dict):
            for key in ["enhanced", "output", "out", "image", "result", "pred"]:
                val = out.get(key)
                if torch.is_tensor(val):
                    return val
            for val in out.values():
                if torch.is_tensor(val):
                    return val
        raise RuntimeError("Model çıktısından tensor alınamadı.")

    def _to_tensor(self, bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h0, w0 = bgr.shape[:2]
        img = bgr
        if self.input_size and self.input_size > 0:
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
        x = x.float() / 255.0
        if self.fp16:
            x = x.half()
        return x, (h0, w0)

    @torch.inference_mode()
    def _enhance_deepfusion(self, bgr: np.ndarray) -> np.ndarray:
        x, orig_hw = self._to_tensor(bgr)
        out = self.model(x)
        y = self._pick_tensor(out)
        if y.ndim == 4:
            y = y[0]
        if y.ndim != 3:
            raise RuntimeError(f"Beklenmeyen model çıktı şekli: {tuple(y.shape)}")

        # Some models return BCHW RGB in [0,1], some can slightly overflow.
        y = y.float().detach()
        if y.shape[0] == 1:
            y = y.repeat(3, 1, 1)
        y = torch.clamp(y, 0.0, 1.0)
        arr = (y.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        bgr_out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        h0, w0 = orig_hw
        if bgr_out.shape[0] != h0 or bgr_out.shape[1] != w0:
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out

    @torch.inference_mode()
    def _enhance_gamma(self, bgr: np.ndarray, luma_score: float) -> np.ndarray:
        x = torch.from_numpy(bgr).to(self.device, non_blocking=True).float() / 255.0
        norm_luma = clamp(luma_score / 255.0, 0.02, 0.85)
        gamma = float(clamp(np.log(0.55) / np.log(norm_luma + 1e-6), 0.35, 0.90))
        y = torch.pow(torch.clamp(x, 0.0, 1.0), gamma)
        lo = torch.quantile(y, 0.01)
        hi = torch.quantile(y, 0.99)
        y = torch.clamp((y - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        return (y * 255.0).byte().cpu().numpy()

    def set_mode(self, mode: str):
        if mode not in ["auto", "always", "never"]:
            return
        self.mode = mode
        print(f"[ENHANCE] Mod değişti -> {self.mode.upper()}")

    def toggle_on_off(self):
        # D tuşu: kapalıysa direkt karanlık aydınlatmaya geçer, açıksa raw görüntüye döner.
        if self.mode == "never":
            self.set_mode("always")
        else:
            self.set_mode("never")

    def cycle_mode(self):
        order = ["never", "auto", "always"]
        idx = order.index(self.mode) if self.mode in order else 0
        self.set_mode(order[(idx + 1) % len(order)])

    def enhance(self, bgr: np.ndarray) -> Tuple[np.ndarray, bool, float, str]:
        dark, luma_score = self.is_dark(bgr)

        if self.mode == "never":
            return bgr, False, luma_score, "raw"

        if self.mode == "auto" and not dark:
            return bgr, False, luma_score, "raw"

        if self.backend == "deepfusion" and self.model is not None:
            try:
                return self._enhance_deepfusion(bgr), True, luma_score, self.status_text
            except Exception as e:
                print(f"[WARN] DeepFusion inference hatası, gamma fallback: {e}")

        return self._enhance_gamma(bgr, luma_score), True, luma_score, "gamma"


class FaceRecognizer:
    def __init__(self, db_path: str, threshold: float = 0.40):
        self.enabled = False
        self.threshold = threshold
        self.names: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None
        self.app = None
        self.provider_text = "disabled"

        if not db_path or not os.path.exists(db_path):
            print(f"[WARN] Face DB bulunamadı: {db_path}. Kimlik tanıma kapalı.")
            return

        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except Exception as e:
            print(f"[WARN] insightface/onnxruntime yüklenemedi: {e}. Kimlik tanıma kapalı.")
            return

        data = np.load(db_path, allow_pickle=True)
        self.names = data["names"]
        self.embeddings = data["embeddings"].astype(np.float32)
        self.embeddings = np.array([l2_normalize(e) for e in self.embeddings], dtype=np.float32)

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
            self.provider_text = "CUDAExecutionProvider"
        else:
            use_providers = ["CPUExecutionProvider"]
            ctx_id = -1
            self.provider_text = "CPUExecutionProvider"

        self.app = FaceAnalysis(name="buffalo_l", providers=use_providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        self.enabled = True
        print(f"[OK] Face ID aktif. Kişi sayısı: {len(self.names)} | Provider: {self.provider_text}")

    def recognize_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if not self.enabled or crop_bgr.size == 0:
            return "FACE", 0.0
        faces = self.app.get(crop_bgr)
        if len(faces) == 0:
            return "UNKNOWN", 0.0
        face = max(faces, key=lambda f: bbox_area(f.bbox))
        emb = l2_normalize(face.embedding.astype(np.float32))
        sims = np.dot(self.embeddings, emb)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < self.threshold:
            return "UNKNOWN", best_score
        return str(self.names[best_idx]), best_score


class SerialGimbal:
    def __init__(self, port: str, baud: int = 115200, send_hz: float = 50.0):
        self.port = port
        self.baud = baud
        self.send_period = 1.0 / max(send_hz, 1.0)
        self.last_send = 0.0
        self.ser = None
        self.enabled = False
        self.last_cmd = (None, None)

        if not port:
            print("[WARN] Serial port verilmedi. Motor kontrol kapalı.")
            return
        if serial is None:
            print("[WARN] pyserial yüklü değil. Motor kontrol kapalı. Kurulum: pip install pyserial")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
            time.sleep(2.0)
            self.enabled = True
            print(f"[OK] Arduino bağlandı: {port} @ {baud}")
        except Exception as e:
            print(f"[WARN] Arduino bağlanamadı: {e}. Motor kontrol kapalı.")

    def send_velocity(self, pan_speed: int, tilt_speed: int, force: bool = False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_send < self.send_period:
            return
        if not force and self.last_cmd == (pan_speed, tilt_speed):
            return
        msg = f"V {int(pan_speed)} {int(tilt_speed)}\n"
        try:
            self.ser.write(msg.encode("ascii"))
            self.last_send = now
            self.last_cmd = (pan_speed, tilt_speed)
        except Exception as e:
            print(f"[WARN] Serial yazma hatası: {e}")
            self.enabled = False

    def stop(self, force: bool = False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_send < self.send_period:
            return
        try:
            self.ser.write(b"STOP\n")
            self.last_send = now
            self.last_cmd = (0, 0)
        except Exception as e:
            print(f"[WARN] STOP gönderilemedi: {e}")
            self.enabled = False

    def close(self):
        if self.enabled:
            self.stop(force=True)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


@dataclass
class GimbalController:
    pan_kp: float = 28.0
    pan_kd: float = 1.60
    tilt_kp: float = 23.0
    tilt_kd: float = 1.35
    max_pan: float = 6000.0
    max_tilt: float = 5000.0
    deadband_x: float = 10.0
    deadband_y: float = 9.0
    pan_sign: float = -1.0
    tilt_sign: float = -1.0
    smooth: float = 0.62
    last_err_x: float = 0.0
    last_err_y: float = 0.0
    last_pan_cmd: float = 0.0
    last_tilt_cmd: float = 0.0

    def update(self, err_x: float, err_y: float) -> Tuple[int, int]:
        if abs(err_x) < self.deadband_x:
            err_x = 0.0
        if abs(err_y) < self.deadband_y:
            err_y = 0.0
        d_x = err_x - self.last_err_x
        d_y = err_y - self.last_err_y
        self.last_err_x = err_x
        self.last_err_y = err_y
        raw_pan = self.pan_sign * (self.pan_kp * err_x + self.pan_kd * d_x)
        raw_tilt = self.tilt_sign * (self.tilt_kp * err_y + self.tilt_kd * d_y)
        raw_pan = clamp(raw_pan, -self.max_pan, self.max_pan)
        raw_tilt = clamp(raw_tilt, -self.max_tilt, self.max_tilt)
        pan_cmd = self.smooth * self.last_pan_cmd + (1.0 - self.smooth) * raw_pan
        tilt_cmd = self.smooth * self.last_tilt_cmd + (1.0 - self.smooth) * raw_tilt
        self.last_pan_cmd = pan_cmd
        self.last_tilt_cmd = tilt_cmd
        return int(pan_cmd), int(tilt_cmd)

    def reset(self):
        self.last_err_x = 0.0
        self.last_err_y = 0.0
        self.last_pan_cmd = 0.0
        self.last_tilt_cmd = 0.0

    def toggle_pan_direction(self):
        self.pan_sign *= -1.0
        self.reset()

    def toggle_tilt_direction(self):
        self.tilt_sign *= -1.0
        self.reset()

    def direction_summary(self) -> str:
        return f"PAN:{direction_text(self.pan_sign)}({signed_text(self.pan_sign)}) | TILT:{direction_text(self.tilt_sign)}({signed_text(self.tilt_sign)})"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--yolo", type=str, default="models/yolov11n-face.pt", help="YOLO model path. Face, person, object fark etmez.")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--class-name", type=str, default="", help="Sadece bu sınıfı işle. Örn: face/person/car. Boşsa tüm sınıflar.")
    p.add_argument("--track-class", type=str, default="", help="Motor sadece bu sınıfı takip etsin. Boşsa tüm sınıflar aday.")

    p.add_argument("--db", type=str, default="face_db.npz", help="Face ID DB path")
    p.add_argument("--no-id", action="store_true", help="Kimlik tanımayı kapat")
    p.add_argument("--threshold", type=float, default=0.40, help="Face ID threshold")
    p.add_argument("--track-name", type=str, default="", help="Sadece bu kişiyi takip et")
    p.add_argument("--known-only", action="store_true", help="UNKNOWN hedefleri takip etme")

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--min-depth", type=float, default=0.15)
    p.add_argument("--max-depth", type=float, default=8.0)
    p.add_argument("--depth-shrink", type=float, default=0.35)
    p.add_argument("--min-depth-pixels", type=int, default=20)

    p.add_argument("--serial", type=str, default="", help="Windows: COM14 | Linux: /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--send-hz", type=float, default=50.0)
    p.add_argument("--lost-stop-time", type=float, default=0.20)

    p.add_argument("--pan-kp", type=float, default=28.0)
    p.add_argument("--pan-kd", type=float, default=1.60)
    p.add_argument("--tilt-kp", type=float, default=23.0)
    p.add_argument("--tilt-kd", type=float, default=1.35)
    p.add_argument("--max-pan", type=float, default=6000.0)
    p.add_argument("--max-tilt", type=float, default=5000.0)
    p.add_argument("--deadband-x", type=float, default=10.0)
    p.add_argument("--deadband-y", type=float, default=9.0)
    p.add_argument("--cmd-smooth", type=float, default=0.62)
    p.add_argument("--pan-dir", choices=["auto", "normal", "reverse"], default="auto")
    p.add_argument("--tilt-dir", choices=["auto", "normal", "reverse"], default="auto")
    p.add_argument("--invert-pan", action="store_true")
    p.add_argument("--invert-tilt", action="store_true")
    p.add_argument("--pan-sign", type=float, default=-1.0)
    p.add_argument("--tilt-sign", type=float, default=-1.0)

    p.add_argument("--enhance", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--enhance-backend", choices=["auto", "deepfusion", "gamma"], default="auto")
    p.add_argument("--dark-th", type=float, default=70.0)
    p.add_argument("--dark-model", type=str, default="model.pth", help="Dark project model.pth path")
    p.add_argument("--dark-module", type=str, default="deepfusion_model.py", help="Dark project deepfusion_model.py path")
    p.add_argument("--dark-class", type=str, default="", help="Architecture class name, if known")
    p.add_argument("--dark-input-size", type=int, default=0, help="0: orijinal çözünürlük. 320/480 daha hızlı olabilir")
    p.add_argument("--dark-fp16", action="store_true", help="CUDA'da low-light model FP16")
    p.add_argument("--trust-model-code", action="store_true", help="Full PyTorch model yüklemek ve deepfusion_model.py import etmek için")
    p.add_argument("--show-enhanced", action="store_true", help="Ekranda YOLO'ya verilen aydınlatılmış görüntüyü göster")

    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def select_target(detections: List[Dict[str, Any]], args) -> Optional[Dict[str, Any]]:
    candidates = detections
    if args.track_class:
        candidates = [d for d in candidates if args.track_class.lower() in d["class_name"].lower()]
    if args.track_name:
        candidates = [d for d in candidates if d["name"].lower() == args.track_name.lower()]
    if args.known_only:
        candidates = [d for d in candidates if d["name"] not in ["UNKNOWN", "FACE"]]
    if not candidates:
        return None
    with_depth = [d for d in candidates if d["distance"] is not None]
    if with_depth:
        return min(with_depth, key=lambda d: d["distance"])
    return max(candidates, key=lambda d: d["area"])


def maybe_set_realsense_options(depth_sensor):
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if depth_sensor.supports(rs.option.laser_power):
            max_laser = depth_sensor.get_option_range(rs.option.laser_power).max
            depth_sensor.set_option(rs.option.laser_power, max_laser)
    except Exception as e:
        print(f"[WARN] Depth sensor option ayarlanamadı: {e}")


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"[INFO] Torch device: {device}")

    yolo = YOLO(args.yolo)
    yolo.to(device)
    print(f"[OK] YOLO model: {args.yolo}")
    print(f"[INFO] YOLO classes: {yolo.names}")

    recognizer = None if args.no_id else FaceRecognizer(args.db, threshold=args.threshold)

    enhancer = LowLightEnhancer(
        device=device,
        mode=args.enhance,
        backend=args.enhance_backend,
        dark_threshold=args.dark_th,
        model_path=args.dark_model,
        module_path=args.dark_module,
        class_name=args.dark_class,
        input_size=args.dark_input_size,
        fp16=args.dark_fp16,
        trust_model_code=args.trust_model_code,
    )

    gimbal = SerialGimbal(args.serial, baud=args.baud, send_hz=args.send_hz)

    pan_sign = direction_value(args.pan_dir, args.pan_sign)
    tilt_sign = direction_value(args.tilt_dir, args.tilt_sign)
    if args.invert_pan:
        pan_sign *= -1.0
    if args.invert_tilt:
        tilt_sign *= -1.0

    controller = GimbalController(
        pan_kp=args.pan_kp,
        pan_kd=args.pan_kd,
        tilt_kp=args.tilt_kp,
        tilt_kd=args.tilt_kd,
        max_pan=args.max_pan,
        max_tilt=args.max_tilt,
        deadband_x=args.deadband_x,
        deadband_y=args.deadband_y,
        pan_sign=pan_sign,
        tilt_sign=tilt_sign,
        smooth=args.cmd_smooth,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale}")
    maybe_set_realsense_options(depth_sensor)

    align = rs.align(rs.stream.color)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filter = rs.hole_filling_filter()
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    prev_time = time.time()
    last_seen_time = 0.0
    stopped = False
    smooth_target_dist: Optional[float] = None
    dist_alpha = 0.35

    print(f"[INFO] Motor yönü: {controller.direction_summary()}")
    print("[INFO] Başladı. Q çıkış | S stop | D dark aç/kapat | A auto | R raw | E mod değiştir | 1/2/3 motor yön")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_frame = spatial_filter.process(depth_frame).as_depth_frame()
            depth_frame = temporal_filter.process(depth_frame).as_depth_frame()
            depth_frame = hole_filter.process(depth_frame).as_depth_frame()

            color_img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            infer_img, enhanced, luma_score, enh_backend = enhancer.enhance(color_img)

            result = yolo.predict(
                source=infer_img,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            detections: List[Dict[str, Any]] = []
            if result.boxes is not None:
                for b in result.boxes:
                    box = b.xyxy[0].detach().cpu().numpy()
                    conf = float(b.conf[0].detach().cpu().item())
                    cls_id = int(b.cls[0].detach().cpu().item())
                    class_name = yolo.names.get(cls_id, str(cls_id))

                    if args.class_name and args.class_name.lower() not in class_name.lower():
                        continue

                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    dist = sample_median_depth(
                        depth_m,
                        box,
                        shrink=args.depth_shrink,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        min_valid_pixels=args.min_depth_pixels,
                    )

                    X = Y = Z = None
                    if dist is not None:
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], dist)

                    name, id_score = "FACE", 0.0
                    if recognizer is not None and recognizer.enabled:
                        crop = crop_with_padding(infer_img, box, pad_ratio=0.30)
                        name, id_score = recognizer.recognize_crop(crop)

                    detections.append({
                        "box": box,
                        "conf": conf,
                        "class_name": class_name,
                        "name": name,
                        "id_score": id_score,
                        "distance": dist,
                        "xyz": (X, Y, Z) if dist is not None else None,
                        "center": (cx, cy),
                        "area": bbox_area(box),
                    })

            target = select_target(detections, args)
            # Enhancement aktifse ekranda da aydınlatılmış görüntüyü gösteriyoruz.
            # Böylece D tuşuna basınca etkisini direkt görürsün.
            view = infer_img.copy() if (args.show_enhanced or enhanced) else color_img.copy()
            h, w = view.shape[:2]
            cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)

            if target is not None:
                last_seen_time = time.time()
                stopped = False
                tx, ty = target["center"]
                err_x = tx - (w / 2.0)
                err_y = ty - (h / 2.0)
                pan_cmd, tilt_cmd = controller.update(err_x, err_y)
                gimbal.send_velocity(pan_cmd, tilt_cmd)
                if target["distance"] is not None:
                    if smooth_target_dist is None:
                        smooth_target_dist = target["distance"]
                    else:
                        smooth_target_dist = dist_alpha * target["distance"] + (1.0 - dist_alpha) * smooth_target_dist
            else:
                if time.time() - last_seen_time > args.lost_stop_time:
                    controller.reset()
                    smooth_target_dist = None
                    if not stopped:
                        gimbal.stop(force=True)
                        stopped = True

            for d in detections:
                x1, y1, x2, y2 = map(int, d["box"])
                is_target = d is target
                if d["name"] == "UNKNOWN":
                    color = (0, 0, 255)
                elif is_target:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                thickness = 3 if is_target else 2
                cv2.rectangle(view, (x1, y1), (x2, y2), color, thickness)

                dist_text = "depth yok" if d["distance"] is None else f"{d['distance']:.2f}m"
                if recognizer is not None and recognizer.enabled:
                    label = f"{d['class_name']} | {d['name']} {d['id_score']:.2f} | {dist_text}"
                else:
                    label = f"{d['class_name']} {d['conf']:.2f} | {dist_text}"
                cv2.putText(view, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

                if d["xyz"] is not None:
                    X, Y, Z = d["xyz"]
                    xyz_text = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m"
                    cv2.putText(view, xyz_text, (x1, min(h - 10, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

                cx, cy = d["center"]
                cv2.circle(view, (cx, cy), 4, color, -1)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            motor_status = "ON" if gimbal.enabled else "OFF"
            dir_status = f"P{signed_text(controller.pan_sign)} T{signed_text(controller.tilt_sign)}"
            enh_status = f"{enhancer.mode.upper()}:{enh_backend.upper() if enhanced else 'RAW'}"
            cv2.putText(
                view,
                f"FPS:{fps:.1f} | {device} | ENH:{enh_status} luma:{luma_score:.1f} | Motor:{motor_status} | DIR:{dir_status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
            )
            cv2.putText(
                view,
                "Keys: D dark on/off | A auto | R raw | E cycle | S stop | 1/2/3 dir | Q quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
            )

            if target is not None:
                tx, ty = target["center"]
                cv2.line(view, (w // 2, h // 2), (tx, ty), (0, 255, 255), 2)
                if smooth_target_dist is not None:
                    cv2.putText(
                        view,
                        f"TARGET: {target['class_name']} {target['name']} | DIST: {smooth_target_dist:.2f} m",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2,
                    )
                else:
                    cv2.putText(
                        view,
                        f"TARGET: {target['class_name']} {target['name']}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2,
                    )

            cv2.imshow("RealSense + Dark Enhance + YOLO Depth + Gimbal", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                controller.reset()
                gimbal.stop(force=True)
                stopped = True
            elif key == ord("d"):
                enhancer.toggle_on_off()
            elif key == ord("a"):
                enhancer.set_mode("auto")
            elif key == ord("r"):
                enhancer.set_mode("never")
            elif key == ord("e"):
                enhancer.cycle_mode()
            elif key == ord("1"):
                controller.toggle_pan_direction()
                gimbal.stop(force=True)
                stopped = True
                print(f"[DIR] PAN yön değişti -> {controller.direction_summary()}")
            elif key == ord("2"):
                controller.toggle_tilt_direction()
                gimbal.stop(force=True)
                stopped = True
                print(f"[DIR] TILT yön değişti -> {controller.direction_summary()}")
            elif key == ord("3"):
                controller.toggle_pan_direction()
                controller.toggle_tilt_direction()
                gimbal.stop(force=True)
                stopped = True
                print(f"[DIR] PAN+TILT yön değişti -> {controller.direction_summary()}")

    finally:
        gimbal.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
