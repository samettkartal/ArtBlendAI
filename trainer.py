import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from models.generator import Generator
from models.discriminator import Discriminator
from torch.cuda.amp import autocast, GradScaler


def embed_prompt_batch(texts, tokenizer, model, device):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings


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
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception:
            return None


def train_gan(
    data_path="data/wikiart",
    img_size=128,
    num_epochs=50,
    batch_size=32,
    save_every=5,
    output_dir="outputs/"
):
    if not torch.cuda.is_available():
        raise EnvironmentError("❌ Bu kod GPU olmadan çalışmaz.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    os.makedirs(output_dir, exist_ok=True)

    # Load styles
    with open("frontend/public/styles.json", "r", encoding="utf-8") as f:
        style_dict = json.load(f)
    num_styles = len(style_dict)
    style_dim, prompt_dim = 128, 384

    # Models & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    st_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    generator = Generator(style_dim, prompt_dim, img_size).to(device)
    discriminator = Discriminator(style_dim, prompt_dim, img_size).to(device)
    style_embedding = nn.Embedding(num_styles, style_dim).to(device)

    from torch.cuda.amp import GradScaler
    import torch.optim as optim

    # --- Optimizatörler ---
    optimizer_G = optim.Adam(
        list(generator.parameters()) + list(style_embedding.parameters()),
        lr=3e-4,               # ↑ G için daha yüksek öğrenme hızı
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=1e-4,               # ↓ D için biraz daha düşük öğrenme hızı
        betas=(0.5, 0.999)
    )

    # --- Scheduler: Cosine Annealing ---
    # T_max = toplam epoch sayınız; eta_min = lr'nin düşeceği minimum seviye
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G,
        T_max=num_epochs,
        eta_min=1e-5
    )
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # --- Mixed Precision Scalers ---
    # Daha agresif başlangıç ölçeğiyle FP16 sırasında daha stabil gradient ölçeği
    scaler_G = GradScaler(init_scale=2**16)
    scaler_D = GradScaler(init_scale=2**16)


    # DataLoader
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image_paths, labels = [], []
    for style_name, idx in style_dict.items():
        folder = os.path.join(data_path, str(idx))
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder, fname))
                labels.append(idx)

    dataset = WikiArtDataset(image_paths, labels, style_dict, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    loss_fn = nn.BCEWithLogitsLoss() 
    best_g_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        d_loss, g_loss = 0.0, 0.0
        for i, batch in enumerate(dataloader):
            if batch is None or len(batch) != 2: continue
            imgs, labels_tensor = batch
            imgs, labels_tensor = imgs.to(device), labels_tensor.to(device)
            style_vec = style_embedding(labels_tensor)
            prompt_vec = embed_prompt_batch(
                [list(style_dict.keys())[labels_tensor[j].item()] for j in range(labels_tensor.size(0))],
                tokenizer, st_model, device
            )

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            with autocast():
                gen_imgs = generator(style_vec, prompt_vec)
                validity = discriminator(gen_imgs, style_vec.detach(), prompt_vec.detach())
                real_targets = torch.full_like(validity, 0.9, device=device)
                adv_loss = loss_fn(validity, real_targets)
                # Total Variation
                tv_loss = (torch.mean(torch.abs(gen_imgs[:, :, 1:] - gen_imgs[:, :, :-1])) +
                           torch.mean(torch.abs(gen_imgs[:, :, :, 1:] - gen_imgs[:, :, :, :-1])))
                g_loss_step = adv_loss + 1e-5 * tv_loss
            scaler_G.scale(g_loss_step).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if i % 3 == 0:
                optimizer_D.zero_grad()
                with autocast():
                    noise_std = 0.02
                    imgs_noisy = imgs + noise_std * torch.randn_like(imgs)
                    real_validity = discriminator(imgs_noisy, style_vec.detach(), prompt_vec.detach())
                    fake_validity = discriminator(gen_imgs.detach(), style_vec.detach(), prompt_vec.detach())
                    real_targets = torch.full_like(real_validity, 0.9, device=device)
                    fake_targets = torch.full_like(fake_validity, 0.1, device=device)
                    d_loss_step = 0.5 * (loss_fn(real_validity, real_targets) + loss_fn(fake_validity, fake_targets))
                scaler_D.scale(d_loss_step).backward()
                scaler_D.step(optimizer_D)
                scaler_D.update()

            if i % 5 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D_loss: {d_loss_step.item():.4f} G_loss: {g_loss_step.item():.4f}")

        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()

        # Save best model
        if g_loss_step.item() < best_g_loss:
            best_g_loss = g_loss_step.item()
            torch.save(generator.state_dict(), os.path.join(output_dir, "final_model.pth"))
            torch.save(style_embedding.state_dict(), os.path.join(output_dir, "final_embedding.pth"))

        # Periodic image outputs
        if epoch % save_every == 0 or epoch == num_epochs:
            with torch.no_grad():
                sample = generator(style_vec, prompt_vec)
                save_image(sample[:16], os.path.join(output_dir, f"epoch_{epoch:03d}.png"), nrow=4, normalize=True)

    print(f"✅ Eğitim tamamlandı. En iyi G kaybı: {best_g_loss:.4f}")


if __name__ == "__main__":
    train_gan()
