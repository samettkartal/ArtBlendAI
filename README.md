

# **ArtBlendAI: Generative Art with Style Transfer and Text Prompts**

## **Proje Açıklaması**

**ArtBlendAI** projesi, kullanıcıların belirli stil ve metin (prompt) verilerine dayalı olarak sanatsal görseller üretmelerini sağlayan bir yapay zeka uygulamasıdır. Proje, **stil transferi** ve **text-to-image generation** (metinden görsel üretimi) kullanarak, farklı sanat akımlarını ve metin açıklamalarını birleştirir ve yaratıcı görseller oluşturur. Kullanıcı, seçtiği sanat stilleri ve verdiği metin ile istediği tarzda sanatsal görseller üretebilir.

**Özellikler:**

* Kullanıcılar iki farklı sanat stilini seçebilir ve bunları karıştırarak yeni bir stil oluşturabilirler.
* Kullanıcıdan alınan metin (prompt), görsel üretimi için kullanılmadan önce dilsel embedding'e dönüştürülür.
* **FastAPI** backend ile hızlı görsel üretimi ve dinamik görsel gösterimi sağlanır.
* Kullanıcı dostu bir React arayüzü ile görsel üretim süreci kolayca başlatılabilir ve sonuçlar anında görüntülenebilir.

## **Kullanılan Teknolojiler**

### **1. Model ve Yapay Zeka:**

* **Generator Modeli**: Stil transferi ve prompt ile görsel üretimi sağlamak için eğitilen **Generator** modeli, stil ve metin embedding'lerini alarak yeni görseller üretir. Generator, hem stil bilgilerini hem de prompt bilgilerini kullanarak yaratıcı bir görsel çıktı üretir.
* **Style Embedding**: Stil vektörlerinin sayısal temsilleridir. Kullanıcı tarafından seçilen stil isimleri, modelde kullanılacak sayısal vektörlere dönüştürülür. Bu stil embedding'leri, stilin görsel özelliklerini modelin anlayabileceği bir biçime dönüştürür. Örneğin, bir **cubism** stilinin geometrik şekillerini ve soyut öğelerini temsil eden bir embedding üretilir.
* **Pretrained Model Ağırlıkları**: **Generator** ve **Style Embedding** modelleri, daha önce eğitilmiş ve kaydedilmiş ağırlıklarla yüklenir. Bu sayede model daha hızlı bir şekilde öğrenmeye başlar ve kullanıcıdan alınan stil ve prompt ile doğru görselleri üretir.

### **2. Embedding İşlemi ve Katkısı:**

**Embedding işlemi**, modelin doğru ve anlamlı görseller üretmesine yardımcı olan en önemli bileşendir. Bu işlem, metin ve stil bilgilerini **sayısal vektörlere** dönüştürerek, modelin bu bilgileri daha etkili bir şekilde işlemesini sağlar. İşte bu işlemin nasıl çalıştığı ve modelin çıktısına olan katkıları:

* **Prompt Embedding**: Kullanıcıdan alınan metin (prompt), **SentenceTransformer** modeli ile embedding'e dönüştürülür. Bu vektör, metnin anlamını ve bağlamını modelin anlayabileceği şekilde temsil eder. **Prompt embedding**'i, görsel üretim sürecinde modelin daha anlamlı ve bağlama uygun görseller üretmesine yardımcı olur. Örneğin, "sunset over the mountains" gibi bir prompt, modelin dağlar ve gün batımı öğelerini anlamasına ve bunları görselde doğru şekilde yerleştirmesine yardımcı olur.

* **Style Embedding**: **Style embedding**, stil bilgisini sayısal bir temsile dönüştürür. Kullanıcıdan alınan stil isimleri (örneğin, "cubism", "impressionism") önce stil ID'lerine, ardından embedding vektörlerine dönüştürülür. Bu stil embedding'leri, modelin, seçilen stilin özelliklerini öğrenmesine ve bu özellikleri görsel üretiminde kullanmasına olanak tanır. Her stilin kendine özgü görsel özellikleri (örneğin, soyut formlar, geometrik şekiller, renk paletleri) bu embedding'ler aracılığıyla modele aktarılır.

* **Blend Mode ve Embedding**: **Blend Mode** özelliği, iki stilin embedding'lerini birleştirerek yeni bir stil oluşturulmasına olanak tanır. Bu sayede, modelin her iki stilin özelliklerini aynı görselde birleştirmesi sağlanır. Örneğin, stil karışım oranına göre daha özgün ve yaratıcı sonuçlar elde edilir.

* **Modelin Çıktısına Katkısı**: Embedding işlemi, modelin **prompt** ve **stil** verilerini doğru şekilde anlamasına yardımcı olur. Bu sayede, model verilen metni ve stili uygun şekilde birleştirir ve kullanıcıya daha anlamlı, yaratıcı görseller sunar. **Prompt embedding** ve **style embedding**'leri, görsel üretim sürecinin temel bileşenleri olup, görselin içerik ve stil açısından uyumlu olmasını sağlar. Model, bu embedding'ler ile verilen metnin anlamını ve stilin görsel özelliklerini öğrenir, ardından bunları birleştirerek görseli üretir.

**Sonuç Olarak**: Embedding işlemleri, **ArtBlendAI** projesinin başarılı görsel üretimi için kritik bir rol oynar. Hem stilin görsel özelliklerini hem de metnin anlamını doğru şekilde işleyebilmek, yüksek kaliteli ve tutarlı görsellerin üretilebilmesini sağlar. Bu nedenle, embedding işlemleri modelin çıktısına önemli katkılarda bulunur.

### **3. Frontend (Kullanıcı Arayüzü):**

* **React**: Dinamik ve kullanıcı etkileşimini sağlayan bir JavaScript kütüphanesidir. Arayüzdeki bileşenler modüler olarak yapılandırılmıştır. Kullanıcı, stil ve prompt seçimlerini yaparak görsel üretim sürecini başlatır.
* **Vite**: Modern JavaScript uygulamaları için hızlı geliştirme deneyimi sunar. Vite ile hızlı modül paketleme ve derleme işlemleri sağlanır.
* **CSS / Tailwind CSS**: Arayüzde stil uygulamak için kullanılır. Tailwind, hızlı prototipleme ve özelleştirilmiş stiller oluşturmanıza yardımcı olur. Tasarım, şık ve uyumlu bileşenlerle yapılmıştır.
* **React Router**: Tek sayfa uygulamaları (SPA) için yönlendirme sağlar. Sayfalar arasında geçiş yapmadan uygulamanın hızlı bir şekilde çalışmasını sağlar.

  ![arayüz](https://github.com/user-attachments/assets/f286e83c-50b8-46c4-9d70-d3037dff327e)

Kullanıcı Arayüzü

Kısa bir metin prompt ve blend mode girilir.

Açılır menülerden Stil 1 ve Stil 2 seçilir.

“Görseli Üret” butonuna tıklanır.

Arka Uç İşlemleri
Frontend, FastAPI’ye aşağıdaki gibi bir POST /generate isteği yollar:

{
  "prompt": "Güneş batarken bir kedi",
  "style1": "cubism",
  "style2": "baroque",
  "blend_mode": "mix"
}
Metin tabanlı prompt, sentence-transformers ile embedding’e (prompt_vec) dönüştürülür.

Rastgele üretilen latent vektör (z) ile style_vec1 ve style_vec2 birleştirilir.

Bu birleşik vektör, özgün GAN generator modeline aktarılır ve yeni görsel oluşturulur.

Oluşan görüntü PNG formatında kaydedilir; bir image_url oluşturulur.

FastAPI, JSON yanıtında { "status": "success", "image_url": "https://…" } döner.

Frontend, dönen image_url’i alıp ekranda kullanıcıya gösterir.

  ![arayüz2](https://github.com/user-attachments/assets/ed977350-44cc-4789-8ee0-9bec1bfc187c)




* ### **4. Backend (Sunucu Tarafı):**

* **FastAPI**: Python tabanlı bir web framework’üdür. **FastAPI**, RESTful API servisi sunarak **React** frontend'i ile hızlı ve güvenli veri iletişimini sağlar. Görsel üretim istekleri burada işlenir ve sonuç frontend'e döndürülür.
* **PyTorch**: Derin öğrenme framework'ü olarak kullanılır. Model eğitimi ve görsel üretimi işlemleri **PyTorch** ile gerçekleştirilir.
* **SentenceTransformer**: Metin embedding'leri için kullanılır. Kullanıcının verdiği metni, anlamlı bir vektöre dönüştürür, bu da görsel üretiminin anlamlı ve uyumlu olmasını sağlar.
* **Deep Translator (GoogleTranslator)**: Kullanıcıdan alınan prompt'u otomatik olarak İngilizce'ye çevirir, böylece çok dilli destek sağlanır.
* **torchvision**: Görsel işleme ve görsel kaydetme işlemleri için kullanılır.

---

## **Kurulum Adımları**

### 1. **Backend Kurulumu (FastAPI)**

#### Gerekli Bağımlılıklar:

```bash
pip install -r requirements.txt
```

#### FastAPI ile Backend Çalıştırma:

```bash
uvicorn main:app --reload
```

**Not**: Backend için `main.py` dosyasını kullanıyoruz. Bu dosya, görsel üretim API'sini ve stil transferi işlemlerini içerir.

### 2. **Frontend Kurulumu (React)**

#### Gerekli Bağımlılıklar:

```bash
npm install
```

#### React Uygulamasını Çalıştırma:

```bash
npm run dev
```

Frontend kısmı **React** ve **Vite** ile geliştirilmiştir. React uygulaması, kullanıcının stil ve prompt bilgilerini alarak, **FastAPI** backend ile iletişime geçer ve görseli kullanıcıya gösterir.

### 3. **Görsel Üretimi ve API İletişimi**

* Kullanıcı arayüzünde prompt metni ve stil seçimleri yapılır.
* Stil bilgileri **FastAPI** backend'ine gönderilir, ardından görsel üretilip **React** arayüzünde dinamik olarak gösterilir.
* Üretilen görsel, backend tarafından kaydedilip frontend'e gösterilmek üzere döndürülür.

---

## **Proje Yapısı**

```plaintext
GAN-STYLE-FUSION-AI/
├── app/      
│   ├── main.py              # FastAPI uygulama giriş noktası
│   ├── routes/
│      └── generate.py       # API endpoint’leri
│   
│
├── data/
│   ├── wikiart/              # WikiArt veri seti (ignored)
│   
│
├── art-blend-ai-front/
│   ├── public/
│   │   ├── favicon.ico
│   │   └── vite.svg
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── GenerateButton.jsx
│   │   │   ├── GenerateButton.css
│   │   │   ├── ImageDisplay.jsx
│   │   │   ├── ImageDisplay.css
│   │   │   ├── PromptInput.jsx
│   │   │   ├── PromptInput.css
│   │   │   ├── ThemeSelector.jsx
│   │   │   └── ThemeSelector.css
│   │   │
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── styles.json
│   │
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── eslint.config.js
│
├── outputs/          # Üretilen görseller ve modeller (gitignored)
│   ├── best_generator.pth
│   ├── style_embedding_final.pth
│   ├── generated_51196.png(örn.)
│                   
│
├── .gitignore
└── README.md

```

---

## **Proje Kullanımı**

1. **Stil Seçimi**: Kullanıcı, mevcut sanat stillerinden iki tanesini seçer.
2. **Prompt Girişi**: Kullanıcı, görseli oluşturacak metni (prompt) girer.
3. **Blend Mode Seçimi**: Kullanıcı, stil karışım modunu seçer (örneğin: style1, style2, random, mean).
4. **Görsel Üretimi**: FastAPI backend, stil ve prompt verilerine dayalı olarak görseli üretir ve frontend'e gönderir.
5. **Görsel Görüntüleme**: Üretilen görsel, arayüzde kullanıcıya gösterilir.

---

### **Katkıda Bulunma**

1. Repo'yu **fork** edin.
2. Geliştirmeler yapın ve **pull request** gönderin.
3. Yeni stil veya özellik eklemeleri önerin!

---

## **Lisansa İlişkin Notlar**

Bu proje **MIT Lisansı** ile lisanslanmıştır.

