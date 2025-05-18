import os
import json

# style_dict yolunu ve wikiart veri yolunu belirt
STYLE_DICT_PATH = "frontend/public/styles.json"
WIKIART_DATA_PATH = "data/wikiart"

# style_dict dosyasını yükle
with open(STYLE_DICT_PATH, "r", encoding="utf-8") as f:
    style_dict = json.load(f)

# ID -> stil ismi dönüşümünü hazırla
id_to_style = {v: k for k, v in style_dict.items()}

# Mevcut klasörleri (etiket ID'lerini) oku
present_ids = sorted([int(d) for d in os.listdir(WIKIART_DATA_PATH) if d.isdigit()])

# Veri setinde gerçekten yer alan stil isimlerini yazdır
print("🎨 Veri setinde kullanılan stiller:\n")
for style_id in present_ids:
    style_name = id_to_style.get(style_id, "❌ styles.json'da eksik!")
    print(f"{style_id}: {style_name}")
