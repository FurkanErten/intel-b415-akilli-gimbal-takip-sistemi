"""
TR: InsightFace ile yüz kimliği veritabanı oluşturma aracı.
Klasör yapısı:
    bilinen_yuzler/
        Furkan/
            1.jpg
            2.jpg
        Ahmet/
            1.jpg

EN: Utility for creating a face identity database with InsightFace.
Folder layout:
    bilinen_yuzler/
        PersonName/
            1.jpg
            2.jpg
"""

import argparse
import os

import cv2
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-9)


def area(box) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def get_providers():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1


def main():
    p = argparse.ArgumentParser(description="Yüz kimliği veritabanı oluşturma aracı / Face identity database builder")
    p.add_argument("--known-dir", type=str, default="bilinen_yuzler")
    p.add_argument("--out", type=str, default="yuz_veritabani.npz")
    p.add_argument("--det-size", type=int, default=640)
    args = p.parse_args()

    from insightface.app import FaceAnalysis

    providers, ctx_id = get_providers()
    print(f"[BILGI] InsightFace sağlayıcıları: {providers}")

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    names = []
    embeddings = []

    if not os.path.isdir(args.known_dir):
        raise FileNotFoundError(f"Klasör yok: {args.known_dir}")

    for person_name in sorted(os.listdir(args.known_dir)):
        person_path = os.path.join(args.known_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        person_embs = []
        for file_name in sorted(os.listdir(person_path)):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            path = os.path.join(person_path, file_name)
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Okunamadı: {path}")
                continue

            faces = app.get(img)
            if not faces:
                print(f"[WARN] Yüz bulunamadı: {path}")
                continue

            face = max(faces, key=lambda f: area(f.bbox))
            emb = l2_normalize(face.embedding.astype(np.float32))
            person_embs.append(emb)
            print(f"[OK] {person_name}: {file_name}")

        if person_embs:
            mean_emb = l2_normalize(np.mean(person_embs, axis=0).astype(np.float32))
            names.append(person_name)
            embeddings.append(mean_emb)
            print(f"[DB] {person_name} eklendi. Fotoğraf: {len(person_embs)}")
        else:
            print(f"[WARN] {person_name} için geçerli yüz yok.")

    if not embeddings:
        raise RuntimeError("Hiç yüz eklenemedi. bilinen_yuzler klasörünü kontrol et.")

    np.savez(args.out, names=np.array(names), embeddings=np.array(embeddings, dtype=np.float32))
    print(f"[OK] DB oluşturuldu: {args.out} | Kişi sayısı: {len(names)}")


if __name__ == "__main__":
    main()
