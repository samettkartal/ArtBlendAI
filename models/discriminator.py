import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator that classifies real vs fake images conditioned on style and prompt.
    Automatically detaches the image tensor to avoid backward-through-graph errors.
    """
    def __init__(self, style_dim: int = 26, prompt_dim: int = 384,
                 img_size: int = 128, img_channels: int = 3):
        super(Discriminator, self).__init__()
        # Compute flattened image dimension
        flattened_img_dim = img_channels * img_size * img_size
        input_dim = flattened_img_dim + style_dim + prompt_dim

        # Simple feed-forward classifier
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with normal distribution for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor,
                style_vec: torch.Tensor,
                prompt_vec: torch.Tensor) -> torch.Tensor:
        # Detach image to break generator's gradient graph
        img = img.detach()
        # Flatten image
        batch_size = img.size(0)
        img_flat = img.view(batch_size, -1)

        # Concatenate image, style, and prompt vectors
        x = torch.cat((img_flat, style_vec, prompt_vec), dim=1)
        # Forward through classifier
        validity = self.model(x)
        return validity
