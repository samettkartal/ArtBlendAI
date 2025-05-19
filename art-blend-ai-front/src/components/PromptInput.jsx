import React from 'react';

const PromptInput = ({ prompt, setPrompt }) => {
  return (
    <div className="brutalist-container" style={{ marginBottom: '30px' }}>
      <label htmlFor="prompt" className="brutalist-label">Proje Adı</label>
      <input
        type="text"
        id="prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Örn: Kedim, Bahar Manzarası"
        className="brutalist-input smooth-type"
      />
    </div>
  );
};

export default PromptInput;
