import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    """
    Güçlendirilmiş Generator: 
    - Gizli boyutlar artırıldı
    - BatchNorm + LeakyReLU + Dropout eklendi
    - Spectral Normalization ile ağırlıklar kısıtlandı
    """
    def __init__(
        self,
        latent_dim: int = 100,
        style_dim: int = 26,
        prompt_dim: int = 384,
        img_size: int = 128
    ):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.prompt_dim = prompt_dim
        self.img_size = img_size

        # Daha yüksek kapasiteli gizli boyutlar
        hidden_dim1 = 512
        hidden_dim2 = 1024
        hidden_dim3 = 2048
        output_dim = img_size * img_size * 3

        # Fully connected blokları
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim + style_dim + prompt_dim, hidden_dim1, bias=False)),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim1, hidden_dim2, bias=False)),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim2, hidden_dim3, bias=False)),
            nn.BatchNorm1d(hidden_dim3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        # Son katman doğrudan çıktı boyutuna
        self.fc4 = spectral_norm(nn.Linear(hidden_dim3, output_dim))

        self.tanh = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        """DCGAN benzeri küçük normal ağırlık başlatma."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, *inputs) -> torch.Tensor:
        # 2 veya 3 input kabul et (z auto-generate)
        if len(inputs) == 2:
            style_vec, prompt_vec = inputs
            batch = style_vec.size(0)
            device = style_vec.device
            z = torch.randn(batch, self.latent_dim, device=device)
        elif len(inputs) == 3:
            z, style_vec, prompt_vec = inputs
        else:
            raise ValueError(f"Expected 2 or 3 inputs, got {len(inputs)}")

        x = torch.cat((z, style_vec, prompt_vec), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        # Yeniden şekillendir ve [-1,1] aralığına sıkıştır
        img = x.view(-1, 3, self.img_size, self.img_size)
        return self.tanh(img)
