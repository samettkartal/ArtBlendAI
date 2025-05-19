import os
import json
import torch
import random
import re
from torch import nn
from torchvision.utils import save_image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from models.generator import Generator

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load style dictionary
with open("frontend/public/styles.json", "r", encoding="utf-8") as f:
    style_dict = json.load(f)

# Embedding and model dimensions
style_emb_dim = 128             # Must match training
num_styles = len(style_dict)
latent_dim = 100                # Must match training
prompt_dim = 384                # From sentence-transformers all-MiniLM-L6-v2
img_size = 128                  # Must match model

# Initialize and load style embedding
style_embedding = nn.Embedding(num_styles, style_emb_dim).to(device)
emb_path = "outputs/final_embedding.pth"
if not os.path.isfile(emb_path):
    raise FileNotFoundError(f"Style embedding checkpoint not found: {emb_path}")
style_embedding.load_state_dict(torch.load(emb_path, map_location=device))
style_embedding.eval()

# Set style_dim for Generator
style_dim = style_emb_dim

# Initialize and load prompt encoder (frozen)
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
st_model.eval()
for param in st_model.parameters():
    param.requires_grad = False

# Initialize and load Generator
generator = Generator(latent_dim, style_dim, prompt_dim, img_size).to(device)
gen_ckpt_path = "outputs/final_model.pth"
if not os.path.isfile(gen_ckpt_path):
    raise FileNotFoundError(f"Generator checkpoint not found: {gen_ckpt_path}")
generator.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
generator.eval()

# Utility: clean and translate prompt
def preprocess_prompt(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    stopwords = {'a','an','the','and','or','but','with','of','to','at','in','on','for','like'}
    return " ".join(w for w in text.split() if w not in stopwords)

# FastAPI schema and router
class GenerateRequest(BaseModel):
    prompt: str
    style1: str
    style2: str = None
    blend_mode: str = None  # 'style1', 'style2', or 'mix'
    blend_alpha: float = None  # fallback for direct alpha blending

router = APIRouter()

@router.post("/generate")
async def generate_image(request: GenerateRequest):
    # Validate blend mode
    mode = request.blend_mode.lower()
    if mode not in {"style1","style2","mix"}:
        raise HTTPException(status_code=400, detail="blend_mode must be 'style1', 'style2', or 'mix'.")

    # Embed styles
    def get_vec(name: str):
        idx = style_dict.get(name.strip().lower())
        if idx is None:
            raise ValueError(f"Style not found: {name}")
        return style_embedding(torch.tensor([idx], device=device))

    try:
        if mode == "style1":
            style_vec = get_vec(request.style1)
        elif mode == "style2":
            style_vec = get_vec(request.style2)
        else:
            if not request.style2:
                raise ValueError("mix mode requires both style1 and style2.")
            style_vec = 0.5 * get_vec(request.style1) + 0.5 * get_vec(request.style2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Preprocess and embed prompt
    try:
        text_en = GoogleTranslator(source='auto', target='en').translate(request.prompt)
    except Exception:
        text_en = request.prompt
    cleaned = preprocess_prompt(text_en)
    prompt_emb = st_model.encode(cleaned, convert_to_tensor=True).unsqueeze(0).to(device)

    # Generate image
    z = torch.randn(1, latent_dim, device=device)
    try:
        with torch.no_grad():
            img_tensor = generator(z, style_vec, prompt_emb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {e}")

    # Save and return URL
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"gen_{random.randint(100000,999999)}.png"
    save_image(img_tensor, os.path.join(out_dir, fname), normalize=True)
    return {"image_url": f"http://127.0.0.1:8000/outputs/{fname}"}

@router.get("/generated")
def list_generated_images():
    """
    Lists the web-accessible paths of generated image files
    in the outputs directory, sorted by modification time (newest first).
    """
    # The filesystem path to the directory
    output_dir_fs = "outputs"
    # The URL base path where this directory is served statically by FastAPI.
    # This MUST match your FastAPI static file configuration (e.g., app.mount("/outputs", ...))
    # Using a relative path like "/outputs" is generally better than a full URL like "http://..."
    base_web_path = "http://127.0.0.1:8000/outputs/"

    image_details = [] # To store (modification_time, web_path)

    # Check if the directory exists on the filesystem
    if os.path.exists(output_dir_fs) and os.path.isdir(output_dir_fs):
        try:
            # List all files in the directory
            filenames = os.listdir(output_dir_fs)

            # Filter for image files and get their details
            for filename in filenames:
                file_path_fs = os.path.join(output_dir_fs, filename)

                # Check if it's a file and ends with a common image extension (case-insensitive)
                if os.path.isfile(file_path_fs) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    try:
                        # Get modification time (timestamp)
                        mtime = os.path.getmtime(file_path_fs)
                        # Construct the web-accessible path
                        web_path = f"{base_web_path}/{filename}"
                        image_details.append((mtime, web_path))
                    except Exception as e:
                        # Handle potential errors getting file info (less common but possible)
                        print(f"Error getting info for file {filename}: {e}")
                        # Skip this file or handle as needed

            # Sort images by modification time, newest first
            # item[0] is the modification time
            image_details.sort(key=lambda item: item[0], reverse=True)

            # Extract just the web paths from the sorted list
            sorted_image_web_paths = [item[1] for item in image_details]
            return sorted_image_web_paths

        except Exception as e:
            # Catch errors during directory listing or file processing
            print(f"Error listing generated images: {e}")
            # Depending on desired behavior, return empty or raise error
            # For a gallery, returning an empty list is often more user-friendly
            return [] # Or raise HTTPException(status_code=500, detail="Failed to list images")
    else:
        # If the directory doesn't exist, return an empty list
        # This is common when the app starts for the first time
        # print(f"Outputs directory not found: {output_dir_fs}") # Optional: log this only once or at debug level
        return []
