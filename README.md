# Intel B415 Kamera ile Derinlik Tabanlı Akıllı Gimbal Takip Sistemi

**TR:** Bu proje; Intel B415/RealSense RGB-D kamera, YOLO tabanlı yüz algılama, HSV tabanlı kırmızı/mavi hedef algılama, düşük ışık görüntü iyileştirme ve iki eksenli NEMA17 gimbal kontrolünü tek bir gerçek zamanlı sistemde birleştirir.

**EN:** This project integrates an Intel B415/RealSense RGB-D camera, YOLO-based face detection, HSV-based red/blue target detection, low-light image enhancement and two-axis NEMA17 gimbal control into a real-time tracking system.

---

## Proje Ekibi / Project Team

- Furkan Erten
- Proje ortağı / Project partner: **[Ad Soyad eklenecek]**

Bursa Teknik Üniversitesi, Elektrik-Elektronik Mühendisliği  
EEM0401 Tasarım Çalışması

---

## Donanım / Hardware

| Bileşen / Component | Adet / Qty | Görev / Purpose |
|---|---:|---|
| Intel B415/RealSense RGB-D Kamera | 1 | RGB görüntü ve derinlik ölçümü / RGB and depth acquisition |
| NEMA17 step motor | 2 | PAN ve TILT eksen hareketi / PAN and TILT motion |
| TB6600 step motor sürücüsü | 2 | NEMA17 motor sürme / NEMA17 motor driving |
| Arduino Uno | 1 | Seri komutları motor hızına dönüştürme / Serial-to-motor control |
| Bilgisayar / GPU destekli PC | 1 | YOLO, OpenCV, RealSense ve kontrol yazılımı / Vision and control software |

---

## Temel Özellikler / Main Features

- Gerçek zamanlı yüz takibi / Real-time face tracking
- Kırmızı ve mavi hedef takibi / Red and blue target tracking
- RealSense derinlik verisi ile mesafe ölçümü / Distance estimation with RealSense depth
- PAN/TILT eksenlerinde NEMA17 + TB6600 kontrolü / NEMA17 + TB6600 control on PAN/TILT axes
- Gamma ve DeepFusion tabanlı düşük ışık iyileştirme / Gamma and DeepFusion low-light enhancement
- CUDA/FP16 destekli YOLO çalışma modu / CUDA/FP16 supported YOLO inference
- Seri haberleşme ile Arduino motor kontrol protokolü / Arduino motor control over serial protocol

---

## Sistem Mimarisi / System Architecture

```text
Intel B415 Kamera
      |
      v
Python: OpenCV + RealSense SDK + YOLO + HSV + Depth
      |
      |  V <pan_hizi> <tilt_hizi>
      v
Arduino Uno
      |
      v
TB6600 Sürücüler -> NEMA17 PAN/TILT Motorları
```

---

## Klasör Yapısı / Repository Structure

```text
hava_savunma_gimbal_takip_sistemi/
├── ana_takip_sistemi.py
├── dusuk_isik_modeli.py
├── yuz_veritabani_olustur.py
├── requirements.txt
├── arduino/nema17_iki_eksen_gimbal/nema17_iki_eksen_gimbal.ino
├── modeller/
├── komutlar/
├── docs/
├── rapor/
└── arsiv/eski_surumler/
```

---

## Kurulum / Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux:

```bash
source .venv/bin/activate
```

Gereksinimler / Requirements:

```bash
pip install -r requirements.txt
```

Arduino IDE ile şu dosyayı Arduino Uno'ya yükleyin / Upload this firmware to Arduino Uno:

```text
arduino/nema17_iki_eksen_gimbal/nema17_iki_eksen_gimbal.ino
```

---

## Çalıştırma / Running

### Stabil yüz takibi / Stable face tracking

```powershell
python ana_takip_sistemi.py --yolo modeller/yuz_tespit_yolov11l.pt --no-id --serial COM14 --width 640 --height 480 --fps 30 --depth-width 640 --depth-height 480 --depth-fps 30 --imgsz 640 --enhance never --enhance-backend gamma --dark-model modeller/dusuk_isik_modeli.pth --dark-module dusuk_isik_modeli.py
```

### Yüz + kırmızı + mavi hedef takibi / Face + red + blue tracking

```powershell
python ana_takip_sistemi.py --yolo modeller/yuz_tespit_yolov11l.pt --no-id --serial COM14 --start-red --start-blue --width 640 --height 480 --fps 30 --depth-width 640 --depth-height 480 --depth-fps 30 --imgsz 640
```

### DeepFusion gece görüş / DeepFusion night enhancement

```powershell
python ana_takip_sistemi.py --yolo modeller/yuz_tespit_yolov11l.pt --no-id --serial COM14 --enhance always --enhance-backend deepfusion --dark-model modeller/dusuk_isik_modeli.pth --dark-module dusuk_isik_modeli.py --dark-input-size 320 --show-enhanced
```

---

## Klavye Kontrolleri / Keyboard Controls

| Tuş / Key | Görev / Function |
|---|---|
| F | Yüz algılamayı aç/kapat / Toggle face detection |
| K | Kırmızı hedef takibini aç/kapat / Toggle red target tracking |
| M | Mavi hedef takibini aç/kapat / Toggle blue target tracking |
| N veya D | Gece görüşü aç/kapat / Toggle night enhancement |
| A | Otomatik gece görüş / Automatic enhancement |
| R | Ham görüntü / Raw image |
| E | HAM → OTO → SÜREKLİ mod geçişi / Mode cycle |
| B | Gamma / DeepFusion arka uç değişimi / Backend switch |
| T | Hedef seçim politikası / Target policy |
| S | Motorları durdur / Stop motors |
| 1 | PAN yönünü tersle / Reverse PAN direction |
| 2 | TILT yönünü tersle / Reverse TILT direction |
| 3 | PAN + TILT yönlerini tersle / Reverse both axes |
| Q | Çıkış / Quit |

---

## Seri Haberleşme Protokolü / Serial Protocol

```text
V <pan_hizi> <tilt_hizi>
STOP
```

---

## GitHub Notu / GitHub Note

Model dosyaları büyük olduğu için Git LFS kullanılması önerilir. YOLO ve DeepFusion ağırlık dosyalarının lisansları ayrıca kontrol edilmelidir.

Model files are large; Git LFS is recommended. YOLO and DeepFusion model weights may have separate license requirements.
