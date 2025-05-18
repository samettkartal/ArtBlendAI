import os
from datasets import load_dataset
from PIL import Image

def download_and_save_wikiart(output_root="data/wikiart", max_images=None):
    print("WikiArt veri seti yükleniyor...")
    dataset = load_dataset("huggan/wikiart")
    train_data = dataset['train']

    print(f"Veri seti başarıyla yüklendi. Toplam görsel sayısı: {len(train_data)}")

    os.makedirs(output_root, exist_ok=True)

    for i, item in enumerate(train_data):
        style = item['style']
        image = item['image']

        if not style or not isinstance(image, Image.Image):
            continue

        # Stil bir liste olabilir, her durumda bir stringe çevirelim
        style = str(style).replace(" ", "_").lower()

        style_folder = os.path.join(output_root, style)
        os.makedirs(style_folder, exist_ok=True)

        image_path = os.path.join(style_folder, f"{style}_{i}.jpg")
        try:
            image.convert("RGB").save(image_path)
        except Exception as e:
            print(f"Hata: {e} → {image_path}")
            continue

        if (i + 1) % 100 == 0:
            print(f"{i + 1} görsel kaydedildi...")

        if max_images and (i + 1) >= max_images:
            break

    print("İşlem tamamlandı.")

if __name__ == "__main__":
    download_and_save_wikiart(output_root="data/wikiart", max_images=100000)  # İstersen 5000'den fazlasını alabilirsin
