# Teknik Mimari / Technical Architecture

## TR

Sistem dört ana katmandan oluşur:

1. **Algılama Katmanı:** Intel B415/RealSense kameradan RGB ve derinlik görüntüsü alınır.
2. **Görüntü İşleme Katmanı:** YOLO ile yüz, HSV eşikleriyle kırmızı/mavi hedefler tespit edilir.
3. **Kontrol Katmanı:** Hedef merkezi ile görüntü merkezi arasındaki hata hesaplanır ve PD tabanlı hız komutu üretilir.
4. **Eyleyici Katmanı:** Arduino Uno, seri porttan gelen hız komutlarını TB6600 sürücüler üzerinden NEMA17 motorlara uygular.

## EN

The system consists of four main layers:

1. **Sensing Layer:** RGB and depth frames are acquired from the Intel B415/RealSense camera.
2. **Vision Layer:** Faces are detected with YOLO, while red/blue targets are detected with HSV segmentation.
3. **Control Layer:** The error between the target center and image center is converted into PD-based speed commands.
4. **Actuation Layer:** Arduino Uno receives serial speed commands and drives NEMA17 motors through TB6600 drivers.
