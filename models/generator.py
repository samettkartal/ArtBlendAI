import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator that maps noise, style embedding, and prompt embedding to an image.
    Supports calls with or without explicit noise vector for Trainer compatibility.
    """
    def __init__(self, latent_dim: int = 100, style_dim: int = 26,
                 prompt_dim: int = 384, img_size: int = 128):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.prompt_dim = prompt_dim
        self.img_size = img_size

        hidden_dim1 = 128
        hidden_dim2 = 256
        hidden_dim3 = 512
        output_dim = img_size * img_size * 3

        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim + style_dim + prompt_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

        self.activation = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with normal distribution and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, *inputs) -> torch.Tensor:
        # Handle both trainer calls: (style_vec, prompt_vec) or (z, style_vec, prompt_vec)
        if len(inputs) == 2:
            # No explicit noise provided, generate internally
            style_vec, prompt_vec = inputs
            batch_size = style_vec.size(0)
            device = style_vec.device
            z = torch.randn(batch_size, self.latent_dim, device=device)
        elif len(inputs) == 3:
            z, style_vec, prompt_vec = inputs
        else:
            raise ValueError(f"Generator.forward expects 2 or 3 inputs, got {len(inputs)}")

        # Concatenate noise, style, and prompt embeddings
        x = torch.cat([z, style_vec, prompt_vec], dim=1)

        # Forward through fully connected layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        # Reshape to image and apply tanh activation
        batch_size = x.size(0)
        img = x.view(batch_size, 3, self.img_size, self.img_size)
        return self.tanh(img)
