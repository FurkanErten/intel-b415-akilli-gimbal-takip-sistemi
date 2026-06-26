# RealSense + Dark Enhance + YOLO Mesafe + NEMA17 Gimbal

Bu paket çalışan gimbal mantığını bozmadan şu eklemeleri yapar:

- RealSense depth ile **her YOLO tespitinin kameraya uzaklığı** ölçülür.
- Mesafe tek pikselden değil, bbox orta bölgesinden **median depth** ile alınır.
- Karanlık ortamda görüntü YOLO'ya verilmeden önce aydınlatılır.
- `model.pth + deepfusion_model.py` varsa onu kullanmaya çalışır.
- Model yüklenemezse GPU/CPU adaptive gamma fallback ile çalışır.
- Arduino tarafında mevcut protokol korunur: `V panSpeed tiltSpeed`.

## Dosya yerleşimi

Önerilen klasör:

```text
project/
 ├── realsense_dark_depth_gimbal_v4.py
 ├── build_face_db.py
 ├── gimbal_2axis_800_fast.ino
 ├── face_db.npz                  # varsa
 ├── model.pth                    # dark_gimbal_project içinden kopyala
 ├── deepfusion_model.py           # dark_gimbal_project içinden kopyala
 └── models/
      └── yolov11n-face.pt
```

## Kurulum

```bash
pip install -r requirements.txt
```

CUDA'lı sistemde InsightFace için:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

## Sadece yüz + mesafe + motor

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --no-id --serial COM14
```

## Karanlık aydınlatma model.pth ile

`model.pth` ve `deepfusion_model.py` aynı klasördeyse:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --no-id --serial COM14 --enhance auto --enhance-backend deepfusion --dark-model model.pth --dark-module deepfusion_model.py --trust-model-code
```

Daha hızlı test için:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --no-id --serial COM14 --enhance auto --enhance-backend gamma
```

## Kimlik tanıma

Önce klasör oluştur:

```text
known_faces/
 ├── Furkan/
 │    ├── 1.jpg
 │    ├── 2.jpg
 │    └── 3.jpg
 └── Ahmet/
      ├── 1.jpg
      └── 2.jpg
```

DB oluştur:

```bash
python build_face_db.py --known-dir known_faces --out face_db.npz
```

Çalıştır:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --db face_db.npz --serial COM14
```

Sadece Furkan'ı takip et:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --db face_db.npz --track-name Furkan --serial COM14
```

## Her obje için mesafe

Face modeli yerine COCO model kullanırsan tüm nesnelerin mesafesini yazar:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo yolo11n.pt --no-id --serial COM14
```

Sadece insan/person takip etmek için:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo yolo11n.pt --no-id --track-class person --serial COM14
```

Sadece yüz sınıfını işlemek için:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --class-name face --serial COM14
```

## Motor yönleri

Çalışırken:

```text
1 -> PAN yönünü tersle
2 -> TILT yönünü tersle
3 -> PAN + TILT tersle
S -> motor stop
Q -> çıkış
```

Başlatırken sabitlemek için:

```bash
python realsense_dark_depth_gimbal_v4.py --yolo models/yolov11n-face.pt --no-id --serial COM14 --pan-dir reverse --tilt-dir normal
```

## Önemli not

`model.pth` bazen sadece `state_dict` olur, bazen full PyTorch model olur. Bu yüzden kod şu sırayla dener:

1. TorchScript model olarak yükle
2. Full PyTorch model olarak yükle
3. `deepfusion_model.py` içindeki `nn.Module` sınıflarına state_dict yükle
4. Başarısız olursa adaptive gamma fallback

Eğer model sınıfı otomatik bulunamazsa komutta sınıf adını ver:

```bash
--dark-class DeepFusionNet
```


## Runtime karanlık modu

Kod açıkken kapatıp açmadan mod değiştirebilirsin:

```text
D  -> karanlık aydınlatmayı aç/kapat
A  -> otomatik mod: ortam karanlıksa aydınlatır
R  -> raw mod: aydınlatma kapalı
E  -> RAW -> AUTO -> ALWAYS arasında döner
```

D tuşuna basınca enhancement aktifse ekranda da aydınlatılmış görüntü gösterilir. YOLO inference aynı görüntü üzerinden yapılır.
