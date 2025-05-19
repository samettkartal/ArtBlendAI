import React from 'react';
import axios from 'axios';

const GenerateButton = ({
  prompt,
  style1,
  style2,
  blendMode,
  useTwoStyles,
  setGeneratedImagePath,
  setLoading
}) => {
  const handleGenerate = async () => {
  if (!prompt || !style1 || (!style2 && useTwoStyles)) {
    alert("Lütfen prompt ve gerekli tema(ları) girin.");
    return;
  }

  const payload = {
    prompt: prompt,
    style1: style1,
    style2: useTwoStyles ? style2 : "",
    blendMode: blendMode
  };

  // 📌 1. Giden veriyi konsola yaz
  console.log("📤 Gönderilen payload:", payload);

  try {
    setLoading(true);

    // 📌 2. API isteğini gönder
    const response = await axios.post("http://localhost:8000/generate", payload);

    const imgPath = response.data.image_path; // dikkat: response.data.path değilse image_path olabilir
    setGeneratedImagePath("/outputs/" + imgPath);
  } catch (err) {
    // 📌 3. Gelen hatayı detaylıca yaz
    console.error("❌ Görsel üretilemedi:", err.response?.data?.detail || err.message);
    alert("Bir hata oluştu: " + (err.response?.data?.detail || err.message));
  } finally {
    setLoading(false);
  }
};


  return (
    <div style={{ marginTop: "40px" }}>
      <button onClick={handleGenerate} className="flip-card__btn">
        Görseli Üret
      </button>
    </div>
  );
};

export default GenerateButton;
