# Intel B415 Akıllı Gimbal Takip Sistemi

## Geliştiriciler

| İsim                | GitHub                                                                   |
| ------------------- | ------------------------------------------------------------------------ |
| Furkan Erten        | [github.com/FurkanErten](https://github.com/FurkanErten)                 |
| Mehmet Berke Bülbül | [github.com/mehmet-berke-bulbul](https://github.com/mehmet-berke-bulbul) |

Bu çalışma, Intel B415 kamera, görüntü işleme, derinlik algılama ve iki eksenli motor kontrolü kullanılarak geliştirilen akıllı gimbal takip sistemidir. Projede kamera görüntüsü üzerinden hedef tespiti yapılmakta, hedefin görüntü merkezine göre konumu hesaplanmakta ve iki adet NEMA 17 step motor ile pan-tilt eksenlerinde otomatik takip gerçekleştirilmektedir.

---

## Proje Özeti

Bu projede amaç, kamera tabanlı hedef algılama yapan ve algılanan hedefi iki eksenli mekanik sistem ile takip edebilen gerçek zamanlı bir gimbal sistemi geliştirmektir. Sistem; bilgisayarlı görü, derinlik ölçümü, seri haberleşme, gömülü sistem kontrolü ve step motor sürme yapılarının birlikte çalıştığı bütünleşik bir mühendislik uygulamasıdır.

Kamera tarafında Intel B415 kamera kullanılmıştır. Görüntü işleme algoritmaları Python üzerinde çalıştırılmış, hedefin görüntü düzlemindeki konumu belirlenmiş ve kontrol komutları Arduino Uno kartına seri haberleşme ile gönderilmiştir. Arduino Uno, iki adet TB6600 step motor sürücüsü üzerinden iki adet NEMA 17 step motoru kontrol etmektedir.

---

## Kullanılan Donanımlar

| Donanım                  | Adet | Açıklama                                       |
| ------------------------ | ---: | ---------------------------------------------- |
| Intel B415 Kamera        |    1 | Görüntü alma ve derinlik destekli hedef takibi |
| NEMA 17 Step Motor       |    2 | Pan ve tilt eksenlerinin hareket ettirilmesi   |
| TB6600 Step Motor Sürücü |    2 | NEMA 17 motorların sürülmesi                   |
| Arduino Uno              |    1 | Motor kontrolü ve seri haberleşme              |
| Bilgisayar               |    1 | Görüntü işleme ve karar mekanizması            |
| Güç Kaynağı              |    1 | Motor ve sistem beslemesi                      |

---

## Kullanılan Yazılımlar ve Teknolojiler

* Python
* OpenCV
* PySerial
* NumPy
* Intel RealSense SDK / kamera kütüphaneleri
* YOLO tabanlı nesne algılama modeli
* Arduino IDE
* C/C++ tabanlı Arduino motor kontrol kodu
* Seri haberleşme protokolü

---

## Sistem Mimarisi

Sistem genel olarak dört ana bölümden oluşmaktadır:

1. **Görüntü Alma Katmanı**
   Intel B415 kamera üzerinden gerçek zamanlı görüntü alınır.

2. **Görüntü İşleme ve Hedef Tespiti Katmanı**
   Python tabanlı yazılım ile hedef tespiti yapılır. Hedefin görüntü merkezine göre yatay ve dikey hata değeri hesaplanır.

3. **Kontrol ve Haberleşme Katmanı**
   Hesaplanan hata değerleri kontrol algoritmasına aktarılır. Pan ve tilt eksenleri için gerekli hız/yön komutları seri haberleşme ile Arduino Uno’ya gönderilir.

4. **Motor Sürme Katmanı**
   Arduino Uno, gelen komutları yorumlayarak TB6600 sürücüler üzerinden iki adet NEMA 17 step motoru kontrol eder.

Genel veri akışı:

```text
Intel B415 Kamera
        ↓
Python Görüntü İşleme
        ↓
Hedef Tespiti ve Hata Hesabı
        ↓
Seri Haberleşme
        ↓
Arduino Uno
        ↓
TB6600 Motor Sürücüleri
        ↓
NEMA 17 Pan-Tilt Gimbal
```

---

## Klasör Yapısı

```text
intel-b415-akilli-gimbal-takip-sistemi/
│
├── ana_takip_sistemi.py
├── README.md
├── AUTHORS.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── arduino/
│   └── nema17_iki_eksen_gimbal/
│       └── nema17_iki_eksen_gimbal.ino
│
├── modeller/
│   └── model dosyaları
│
├── komutlar/
│   └── örnek başlatma komutları
│
├── rapor/
│   └── tasarım çalışması raporu
│
└── arsiv/
    └── eski_surumler/
```

---

## Kurulum

Öncelikle Python ortamı oluşturulması önerilir.

```bash
python -m venv .venv
```

Windows için:

```bash
.venv\Scripts\activate
```

Linux için:

```bash
source .venv/bin/activate
```

Gerekli Python paketlerini yüklemek için:

```bash
pip install -r requirements.txt
```

---

## Arduino Kurulumu

Arduino Uno kartına `arduino/nema17_iki_eksen_gimbal/nema17_iki_eksen_gimbal.ino` dosyası yüklenmelidir.

Motor sürücü bağlantıları sistemde kullanılan pin tanımlarına göre yapılmalıdır. TB6600 sürücülerin mikrostep, akım limiti ve yön bağlantıları motor özelliklerine uygun şekilde ayarlanmalıdır.

Örnek bağlantı yapısı:

| Eksen | Motor   | Sürücü | Görev                |
| ----- | ------- | ------ | -------------------- |
| Pan   | NEMA 17 | TB6600 | Sağ-sol hareket      |
| Tilt  | NEMA 17 | TB6600 | Yukarı-aşağı hareket |

---

## Çalıştırma

Ana Python dosyası aşağıdaki şekilde çalıştırılabilir:

```bash
python ana_takip_sistemi.py
```

Model, kamera çözünürlüğü, seri port ve takip modu gibi parametreler kullanılan sürüme göre komut satırından değiştirilebilir.

Örnek:

```bash
python ana_takip_sistemi.py --serial COM14 --width 640 --height 480 --fps 30
```

Linux üzerinde seri port örneği:

```bash
python ana_takip_sistemi.py --serial /dev/ttyUSB0 --width 640 --height 480 --fps 30
```

---

## Temel Çalışma Mantığı

Sistem çalıştırıldığında kamera görüntüsü alınır ve hedef tespit algoritması devreye girer. Hedef bulunduğunda hedef kutusunun merkezi hesaplanır. Görüntü merkezinden hedef merkezine olan fark, pan ve tilt eksenleri için hata değeri olarak kullanılır.

Bu hata değerleri kontrol algoritmasına girer. Kontrol çıktısı, motorların hangi yönde ve hangi hızda hareket edeceğini belirler. Python tarafında oluşturulan komutlar seri port üzerinden Arduino Uno’ya gönderilir. Arduino Uno ise TB6600 sürücüler aracılığıyla NEMA 17 motorları hareket ettirir.

---

## Özellikler

* Gerçek zamanlı kamera görüntüsü işleme
* YOLO tabanlı hedef tespiti
* Derinlik destekli takip altyapısı
* İki eksenli pan-tilt gimbal kontrolü
* Arduino Uno ile motor sürme
* TB6600 sürücüler ile yüksek torklu step motor kontrolü
* Seri haberleşme tabanlı bilgisayar-kontrolcü mimarisi
* Modüler ve geliştirilebilir yazılım yapısı

---

## Kullanım Alanları

Bu proje aşağıdaki alanlarda örnek bir mühendislik altyapısı olarak kullanılabilir:

* Akıllı kamera takip sistemleri
* Savunma ve güvenlik odaklı görüntü takip uygulamaları
* Otonom hedef izleme sistemleri
* Robotik pan-tilt kamera platformları
* Bilgisayarlı görü tabanlı kontrol sistemleri
* Gömülü sistem ve motor kontrol uygulamaları

---

## Güvenlik Notları

Bu proje akademik ve mühendislik geliştirme amacıyla hazırlanmıştır. Motorlu sistemlerde test yapılırken mekanik bağlantıların sağlam olduğundan, motor sürücü akım ayarlarının doğru yapıldığından ve güç beslemesinin güvenli olduğundan emin olunmalıdır.

Yüksek akım çeken motor sürücüleri kullanılırken kısa devre, aşırı ısınma ve ters bağlantı risklerine karşı dikkatli olunmalıdır.

---

## Lisans

Bu proje, depoda yer alan `LICENSE` dosyasında belirtilen lisans koşulları altında paylaşılmıştır.

---

# Intel B415 Intelligent Gimbal Tracking System

## Developers

| Name                | GitHub                                                                   |
| ------------------- | ------------------------------------------------------------------------ |
| Furkan Erten        | [github.com/FurkanErten](https://github.com/FurkanErten)                 |
| Mehmet Berke Bülbül | [github.com/mehmet-berke-bulbul](https://github.com/mehmet-berke-bulbul) |

This project is an intelligent two-axis gimbal tracking system developed using an Intel B415 camera, computer vision, depth-assisted target tracking and step motor control. The system detects a target from the camera image, calculates its position relative to the image center and automatically controls the pan-tilt mechanism using two NEMA 17 step motors.

---

## Project Overview

The aim of this project is to develop a real-time camera-based target tracking system capable of controlling a two-axis mechanical gimbal. The system combines computer vision, depth sensing, serial communication, embedded control and step motor driving in a single integrated engineering application.

The Intel B415 camera is used for image acquisition. Image processing algorithms run on Python, where the target position is detected and converted into pan-tilt control commands. These commands are transmitted to an Arduino Uno over serial communication. The Arduino Uno controls two NEMA 17 step motors through two TB6600 step motor drivers.

---

## Hardware Components

| Component                | Quantity | Description                                   |
| ------------------------ | -------: | --------------------------------------------- |
| Intel B415 Camera        |        1 | Image acquisition and depth-assisted tracking |
| NEMA 17 Step Motor       |        2 | Pan and tilt axis motion                      |
| TB6600 Step Motor Driver |        2 | Driving NEMA 17 step motors                   |
| Arduino Uno              |        1 | Motor control and serial communication        |
| Computer                 |        1 | Image processing and decision system          |
| Power Supply             |        1 | Powering the motors and control system        |

---

## Software and Technologies

* Python
* OpenCV
* PySerial
* NumPy
* Intel RealSense SDK / camera libraries
* YOLO-based object detection model
* Arduino IDE
* Arduino C/C++ motor control firmware
* Serial communication protocol

---

## System Architecture

The system consists of four main layers:

1. **Image Acquisition Layer**
   Real-time video is captured using the Intel B415 camera.

2. **Image Processing and Target Detection Layer**
   The Python software detects the target and calculates the horizontal and vertical error values relative to the image center.

3. **Control and Communication Layer**
   The calculated error values are processed by the control algorithm. Required velocity and direction commands for pan and tilt axes are sent to the Arduino Uno via serial communication.

4. **Motor Driving Layer**
   The Arduino Uno interprets the incoming commands and drives two NEMA 17 motors using TB6600 motor drivers.

General data flow:

```text
Intel B415 Camera
        ↓
Python Image Processing
        ↓
Target Detection and Error Calculation
        ↓
Serial Communication
        ↓
Arduino Uno
        ↓
TB6600 Motor Drivers
        ↓
NEMA 17 Pan-Tilt Gimbal
```

---

## Repository Structure

```text
intel-b415-akilli-gimbal-takip-sistemi/
│
├── ana_takip_sistemi.py
├── README.md
├── AUTHORS.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── arduino/
│   └── nema17_iki_eksen_gimbal/
│       └── nema17_iki_eksen_gimbal.ino
│
├── modeller/
│   └── model files
│
├── komutlar/
│   └── example run commands
│
├── rapor/
│   └── design project report
│
└── arsiv/
    └── old_versions/
```

---

## Installation

Creating a Python virtual environment is recommended.

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

For Linux:

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Arduino Setup

Upload the file `arduino/nema17_iki_eksen_gimbal/nema17_iki_eksen_gimbal.ino` to the Arduino Uno.

Motor driver connections should be made according to the pin definitions used in the project. The microstep settings, current limits and direction wiring of the TB6600 drivers must be configured according to the selected motor specifications.

Example axis configuration:

| Axis | Motor   | Driver | Function            |
| ---- | ------- | ------ | ------------------- |
| Pan  | NEMA 17 | TB6600 | Horizontal movement |
| Tilt | NEMA 17 | TB6600 | Vertical movement   |

---

## Running the System

The main Python file can be executed as follows:

```bash
python ana_takip_sistemi.py
```

Depending on the software version, model path, camera resolution, serial port and tracking mode can be configured using command-line arguments.

Example:

```bash
python ana_takip_sistemi.py --serial COM14 --width 640 --height 480 --fps 30
```

Example for Linux serial port:

```bash
python ana_takip_sistemi.py --serial /dev/ttyUSB0 --width 640 --height 480 --fps 30
```

---

## Working Principle

When the system starts, the camera stream is acquired and the target detection algorithm is activated. Once a target is detected, the center of the target bounding box is calculated. The difference between the target center and the image center is used as the error value for the pan and tilt axes.

These error values are processed by the control algorithm. The control output determines the direction and speed of the motors. Commands generated on the Python side are sent to the Arduino Uno through the serial port. The Arduino Uno then controls the NEMA 17 motors via TB6600 drivers.

---

## Features

* Real-time camera image processing
* YOLO-based target detection
* Depth-assisted tracking infrastructure
* Two-axis pan-tilt gimbal control
* Motor control with Arduino Uno
* High-torque step motor control using TB6600 drivers
* Serial communication-based computer-controller architecture
* Modular and extendable software structure

---

## Application Areas

This project can be used as an engineering reference for:

* Intelligent camera tracking systems
* Defense and security-oriented visual tracking applications
* Autonomous target tracking systems
* Robotic pan-tilt camera platforms
* Computer vision-based control systems
* Embedded systems and motor control applications

---

## Safety Notes

This project is developed for academic and engineering purposes. During motorized system tests, mechanical connections must be checked carefully, motor driver current limits must be configured correctly and the power supply must be used safely.

When working with high-current motor drivers, precautions should be taken against short circuits, overheating and incorrect wiring.

---

## License

This project is shared under the license terms specified in the `LICENSE` file.
