# RealSense Dark Depth Gimbal - Nihai Entegrasyon

Bu paket, `gpu_realsense_dark_depth_gimbal_v7_fast_dark.py` dosyasının mühendislik düzenlenmiş nihai halini içerir. Aynı kod ayrıca `gpu_realsense_dark_depth_gimbal_v8_modes.py` adıyla da bırakıldı.

## Ana özellikler

- RealSense RGB + depth hizalama korunur.
- Yüz tespiti YOLO ile çalışır.
- Kırmızı ve mavi hedefler HSV tabanlı ayrı modüllerle tespit edilir.
- Yüz / kırmızı / mavi modları çalışma sırasında tuşla açılıp kapanır.
- Gece görüş modu tuşla açılıp kapanır.
- Gece görüşte iki backend vardır:
  - `gamma`: hızlı, gerçek zamanlı takip için önerilir.
  - `deepfusion`: `model.pth + deepfusion_model.py` ile kalite odaklı, FPS daha düşük olabilir.
- Yüz kapalıyken YOLO inference atlanır; sadece renk takibi daha hızlı çalışır.
- Tüm hedefler aynı depth median mesafe, 3B nokta ve NEMA17 hız kontrol mimarisini kullanır.

## Tuşlar

| Tuş | Görev |
|---|---|
| `F` | Yüz tespitini aç/kapat |
| `K` | Kırmızı renk tespitini aç/kapat |
| `M` | Mavi renk tespitini aç/kapat |
| `N` veya `D` | Gece görüşü aç/kapat |
| `A` | Gece görüşü otomatik moda al |
| `R` | Ham görüntü / gece görüş kapalı |
| `E` | RAW -> AUTO -> ALWAYS döngüsü |
| `B` | Gamma / DeepFusion backend değiştir |
| `T` | Hedef seçimini değiştir: priority / nearest / largest |
| `S` | Motorları durdur |
| `1` | PAN yönünü tersle |
| `2` | TILT yönünü tersle |
| `3` | PAN + TILT yönlerini tersle |
| `Q` | Çıkış |

## Önerilen başlangıç komutu

```powershell
python gpu_realsense_dark_depth_gimbal_v7_fast_dark.py --yolo models/yolov11l-face.pt --no-id --serial COM14 --width 640 --height 480 --fps 30 --depth-width 640 --depth-height 480 --depth-fps 30 --imgsz 640 --enhance never --enhance-backend gamma
```

Bu ayarda program yüz tespiti açık, kırmızı/mavi kapalı, gece görüş kapalı başlar. Çalışırken:

- `F`: yüzü kapat/aç
- `K`: kırmızıyı aç/kapat
- `M`: maviyi aç/kapat
- `N`: gece görüşü aç/kapat

## Kırmızı + mavi baştan açık başlatma

```powershell
python gpu_realsense_dark_depth_gimbal_v7_fast_dark.py --yolo models/yolov11l-face.pt --no-id --serial COM14 --start-red --start-blue --width 640 --height 480 --fps 30 --depth-width 640 --depth-height 480 --depth-fps 30 --imgsz 640
```

## Kalite odaklı DeepFusion gece görüş

```powershell
python gpu_realsense_dark_depth_gimbal_v7_fast_dark.py --yolo models/yolov11l-face.pt --no-id --serial COM14 --enhance always --enhance-backend deepfusion --dark-model model.pth --dark-module deepfusion_model.py --dark-input-size 320 --show-enhanced
```

DeepFusion FPS düşürürse şu ayarlarla stabiliteyi artır:

```powershell
python gpu_realsense_dark_depth_gimbal_v7_fast_dark.py --yolo models/yolov11l-face.pt --no-id --serial COM14 --enhance never --enhance-backend gamma --width 640 --height 480 --depth-width 640 --depth-height 480 --imgsz 512
```

## Hedef seçimi

Varsayılan hedef önceliği:

```text
face -> red -> blue
```

Bunu değiştirmek için:

```powershell
--target-priority red,blue,face
```

Seçim politikaları:

- `priority`: önce öncelik sırasına bakar, o grupta depth varsa en yakını seçer.
- `nearest`: mod fark etmeksizin depth ölçülen en yakın hedefi seçer.
- `largest`: ekranda en büyük kutuyu seçer.

Çalışırken `T` ile politika değişir.

## Renk ayarı notları

Dış ortamda ışık değişirse HSV eşikleri şu parametrelerle ayarlanabilir:

```powershell
--red-s-min 70 --red-v-min 45 --blue-s-min 50 --blue-v-min 45 --color-min-area 450
```

Uzak hedefte renk küçük kalıyorsa `--color-min-area 150` gibi düşür. Gürültü çoksa `--color-min-area` ve `--color-min-fill` değerlerini artır.

## Dosyalar

- `gpu_realsense_dark_depth_gimbal_v7_fast_dark.py`: nihai entegre dosya
- `gpu_realsense_dark_depth_gimbal_v8_modes.py`: aynı nihai kodun versiyon isimli kopyası
- `gpu_realsense_dark_depth_gimbal_v7_fast_dark_ORIGINAL_BACKUP.py`: dokunulmamış orijinal v7 yedeği
- `model.pth`: DeepFusion gece görüş modeli
- `deepfusion_model.py`: DeepFusion model mimarisi
- `models/yolov11l-face.pt`: yüz YOLO modeli
- `gimbal_2axis_800_fast.ino`: Arduino tarafı

