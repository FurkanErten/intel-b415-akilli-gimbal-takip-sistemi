$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $PSScriptRoot "..")
python ana_takip_sistemi.py --yolo modeller/yuz_tespit_yolov11l.pt --no-id --serial COM14 --width 640 --height 480 --fps 30 --depth-width 640 --depth-height 480 --depth-fps 30 --imgsz 640 --enhance never --enhance-backend gamma --dark-model modeller/dusuk_isik_modeli.pth --dark-module dusuk_isik_modeli.py
