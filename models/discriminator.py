import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Daha hafif bir discriminator: katman boyutları küçültüldü, Dropout eklendi.
    """
    def __init__(
        self,
        style_dim: int = 26,
        prompt_dim: int = 384,
        img_size: int = 128,
        img_channels: int = 3,
        dropout_p: float = 0.3
    ):
        super(Discriminator, self).__init__()
        # Görüntüyü düzleştirmek için boyut
        flattened_img_dim = img_channels * img_size * img_size
        input_dim = flattened_img_dim + style_dim + prompt_dim

        # Daha küçük katmanlar ve dropout ile hafifletildi
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Ağırlıkları küçük normal dağılımdan, bias’ları sıfırdan başlat."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        img: torch.Tensor,
        style_vec: torch.Tensor,
        prompt_vec: torch.Tensor
    ) -> torch.Tensor:
        # Görüntüyü tersine çevirmekten (detach) kaçınmak istiyorsanız bu satırı kaldırabilirsiniz.
        img = img.detach()
        batch_size = img.size(0)
        img_flat = img.view(batch_size, -1)

        # Görüntü, stil ve prompt vektörlerini birleştir
        x = torch.cat((img_flat, style_vec, prompt_vec), dim=1)
        validity = self.model(x)
        return validity
