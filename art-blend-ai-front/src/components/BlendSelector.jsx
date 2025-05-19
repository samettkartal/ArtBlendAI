import React from 'react';

const BlendSelector = ({ blendMode, setBlendMode, useTwoStyles }) => {
  return (
    <div style={{ marginTop: '20px' }}>
      <label>Stil uygulama yöntemi:</label><br />
      <select
        value={blendMode}
        onChange={(e) => setBlendMode(e.target.value)}
        disabled={!useTwoStyles && blendMode === "style2"} // 2. stil yoksa kapalı
        style={{ marginTop: '5px' }}
      >
        <option value="style1">Sadece Stil 1</option>
        <option value="style2" disabled={!useTwoStyles}>Sadece Stil 2</option>
        <option value="mix" disabled={!useTwoStyles}>İkisini karıştır</option>
      </select>
    </div>
  );
};

export default BlendSelector;
