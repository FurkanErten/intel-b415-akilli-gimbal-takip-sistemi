# GitHub'a Yükleme Rehberi

```bash
git init
git add .
git commit -m "İlk resmi proje sürümü"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADI/REPO_ADI.git
git push -u origin main
```

Model dosyaları büyük olduğu için Git LFS önerilir:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
git commit -m "Model dosyaları için Git LFS ayarları"
```
