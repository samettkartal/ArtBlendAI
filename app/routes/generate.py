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
from models.generator import Generator
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stil eşlemesi
with open("frontend/public/styles.json", "r", encoding="utf-8") as f:
    style_dict = json.load(f)

style_dim = 128
prompt_dim = 384
img_size = 128
num_styles = len(style_dict)

# Prompt modeli
st_model = SentenceTransformer("all-MiniLM-L6-v2")
st_model.to(device)

# Generator yüklenir
generator = Generator(style_dim=style_dim, prompt_dim=prompt_dim, img_size=img_size).to(device)
generator.load_state_dict(torch.load("outputs/final_model.pth", map_location=device))
generator.eval()

# Style embedding yüklenir
style_embedding = nn.Embedding(num_styles, style_dim).to(device)
style_embedding.load_state_dict(torch.load("outputs/style_embedding_final.pth", map_location=device))
style_embedding.eval()

# Prompt temizleyici
def clean_prompt(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'with', 'of', 'to', 'at', 'in', 'on', 'for', 'like']
    return " ".join([w for w in text.split() if w not in stopwords])

# Prompt gömme
def embed_prompt(text):
    return st_model.encode([text], convert_to_tensor=True).to(device)

# Stil ismini ID'ye çevir
def get_style_id(style_name):
    idx = style_dict.get(style_name.strip().lower())
    if idx is None:
        raise ValueError(f"Stil bulunamadı: {style_name}")
    return torch.tensor([idx], device=device)

# API modeli
class GenerateRequest(BaseModel):
    prompt: str
    style1: str
    style2: str
    blendMode: str  # Yeni eklendi

router = APIRouter()

@router.post("/generate")
def generate_image(request: GenerateRequest):
    # Stil ID'lerini al
    try:
        style_id_1 = get_style_id(request.style1)
        style_id_2 = get_style_id(request.style2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Stil vektörlerini al ve blend moduna göre birleştir
    with torch.no_grad():
        s1 = style_embedding(style_id_1)
        s2 = style_embedding(style_id_2)

        mode = request.blendMode.lower()
        if mode == "style1":
            style_vec = s1
        elif mode == "style2":
            style_vec = s2
        elif mode == "random":
            alpha = torch.rand(1).item()
            style_vec = alpha * s1 + (1 - alpha) * s2
        else:  # "mean" veya bilinmeyen değerler için ortalama
            style_vec = 0.5 * s1 + 0.5 * s2

    # Prompt temizle, çevir, göm
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(request.prompt)
        cleaned = clean_prompt(translated)
        prompt_vec = embed_prompt(cleaned)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prompt işlenemedi: " + str(e))

    # Görsel üret
    try:
        with torch.no_grad():
            gen_img = generator(style_vec, prompt_vec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Görsel üretimi başarısız: {str(e)}")

    # Kaydet
    output_path = f"outputs/generated_{random.randint(10000, 99999)}.png"
    save_image(gen_img, output_path, normalize=True)

    return {"image_path": output_path}
