@echo off
cd /d %~dp0\..
python ana_takip_sistemi.py --yolo modeller/yuz_tespit_yolov11l.pt --no-id --serial COM14 --enhance always --enhance-backend deepfusion --dark-model modeller/dusuk_isik_modeli.pth --dark-module dusuk_isik_modeli.py --dark-input-size 320 --show-enhanced
pause
