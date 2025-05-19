import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from models.generator import Generator
from models.discriminator import Discriminator

from matplotlib import pyplot as plt
import matplotlib.image as mpimg


class WikiArtDataset(Dataset):
    def __init__(self, image_paths, labels, style_dict, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.style_dict = style_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        style_name = list(self.style_dict.keys())[list(self.style_dict.values()).index(label)]

        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label, style_name
        except:
            return None


def embed_prompt_batch(texts, tokenizer, model, device):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings


def train_gan(
    data_path="data/wikiart",
    img_size=128,
    num_epochs=15,
    batch_size=16,
    save_interval=2,
    output_dir="outputs/"
):
    if not torch.cuda.is_available():
        raise EnvironmentError("❌ Bu kod GPU olmadan çalışmaz.")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    print(f"🚀 GPU kullanılıyor: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)

    with open("frontend/public/styles.json", "r", encoding="utf-8") as f:
        style_dict = json.load(f)

    num_styles = len(style_dict)
    style_dim = 128
    prompt_dim = 384  # all-MiniLM-L6-v2 için geçerli

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    st_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    image_paths, labels = [], []
    for style_name, idx in style_dict.items():
        folder = os.path.join(data_path, str(idx))
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder, fname))
                labels.append(idx)

    dataset = WikiArtDataset(image_paths, labels, style_dict, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    generator = Generator(style_dim=style_dim, prompt_dim=prompt_dim, img_size=img_size).to(device)
    discriminator = Discriminator(style_dim=style_dim, prompt_dim=prompt_dim, img_size=img_size).to(device)
    style_embedding = nn.Embedding(num_styles, style_dim).to(device)

    loss_fn = nn.BCELoss()
    optimizer_G = optim.Adam(list(generator.parameters()) + list(style_embedding.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    best_g_loss = float("inf")

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            if batch is None or len(batch) != 3:
                continue

            imgs, labels_tensor, prompt_names = batch
            imgs = imgs.to(device)
            labels_tensor = labels_tensor.to(device)
            style_vec = style_embedding(labels_tensor)
            prompt_vec = embed_prompt_batch(prompt_names, tokenizer, st_model, device)

            # Generator
            # --- Güçlendirilmiş Generator adımı ---
            optimizer_G.zero_grad()

            # 1) Üretilen görüntüler
            gen_imgs = generator(style_vec, prompt_vec)

            # 2) Adversarial loss (label smoothing: hedef 0.9)
            validity = discriminator(gen_imgs, style_vec.detach(), prompt_vec.detach())
            real_targets = torch.full_like(validity, 0.9)
            adv_loss = loss_fn(validity, real_targets)

            # 3) Total Variation (TV) loss – daha pürüzsüz çıktılar için
            tv_h = torch.mean(torch.abs(gen_imgs[:, :, 1:, :] - gen_imgs[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(gen_imgs[:, :, :, 1:] - gen_imgs[:, :, :, :-1]))
            tv_loss = tv_h + tv_w

            # 4) Toplam Generator loss: adversarial + küçük TV düzenlemesi
            g_loss = adv_loss + 1e-5 * tv_loss

            # 5) Geri yayılım ve adım
            g_loss.backward()
            optimizer_G.step()


            # Discriminator
            if i % 20 == 0:
                optimizer_D.zero_grad()

    # Hafif instance noise ile gerçek görüntüleri yumuşat
                noise_std = 0.03
                imgs_noisy = imgs + noise_std * torch.randn_like(imgs)

                # İleri geçiş
                real_validity = discriminator(imgs_noisy, style_vec.detach(), prompt_vec.detach())
                fake_validity = discriminator(gen_imgs.detach(), style_vec.detach(), prompt_vec.detach())

                # Label smoothing: gerçekleri 0.9, sahteleri 0.1
                real_targets = torch.full_like(real_validity, 0.9)
                fake_targets = torch.full_like(fake_validity, 0.1)

                # D kaybını hesapla ve biraz zayıflatmak için ölçeklendir
                d_loss_real = loss_fn(real_validity, real_targets)
                d_loss_fake = loss_fn(fake_validity, fake_targets)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)  # ek bir 0.5 faktörüyle gradyanı yumuşat

                d_loss.backward()

                # İsteğe bağlı: gradyanları kırpmak
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                optimizer_D.step()

            if i % 5 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Kaydetme: En iyi generator ve embedding'i kaydet
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            torch.save(generator.state_dict(), os.path.join(output_dir, "final_model.pth"))
            torch.save(style_embedding.state_dict(), os.path.join(output_dir, "final_embedding.pth"))

        # Görsel çıktıları kaydet
        save_image(gen_imgs[:25], os.path.join(output_dir, f"{epoch:03d}_generated.png"), nrow=5, normalize=True)

    print(f"✅ Eğitim tamamlandı. Model dosyası: final_model.pth ve final_embedding.pth")
    img = mpimg.imread(os.path.join(output_dir, f"{epoch:03d}_generated.png"))
    plt.imshow((img * 0.5 + 0.5))  # Normalize tersine çevir
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_gan()

