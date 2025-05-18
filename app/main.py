from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import generate
from fastapi.staticfiles import StaticFiles
import uvicorn  # Uvicorn'u burada import ediyoruz

app = FastAPI()

# API Root Route
@app.get("/")
def read_root():
    return {"message": "API'ye hoş geldiniz!"}

# Frontend ile haberleşme için CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gerekirse sadece frontend domainiyle sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotaları ekle
app.include_router(generate.router)

# Görsellerin sunulması
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Uygulama çalıştırma
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)  # Portu 8001 olarak değiştirdik
