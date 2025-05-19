import React, { useEffect, useState } from 'react';

const ThemeSelector = ({ style1, setStyle1, style2, setStyle2, useTwoStyles }) => {
  const [styles, setStyles] = useState([]);
  const [open1, setOpen1] = useState(false);
  const [open2, setOpen2] = useState(false);

  // 🔧 useEffect düzeltildi
  useEffect(() => {
    fetch('/styles.json')
      .then(res => res.json())
      .then(data => {
        console.log("Gelen veri:", data);
        // { name, id } formatına dönüştür
        const formatted = Object.entries(data).map(([name, id]) => ({ name, id }));
        setStyles(formatted);
      })
      .catch(err => console.error("Stil listesi yüklenemedi", err));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
      {/* Stil 1 */}
      <div
        className="custom-dropdown"
        onMouseEnter={() => setOpen1(true)}
        onMouseLeave={() => setOpen1(false)}
      >
        <div className="dropdown-label">Stil 1: {style1 || "Seçim yapılmadı"}</div>
        {open1 && (
          <div className="dropdown-menu">
            {styles.map((style) => (
              <div
                key={style.name}
                className="dropdown-option"
                onClick={() => setStyle1(style.name)} // ✅ SADECE stil ismi gönderiliyor
              >
                {style.name}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stil 2 */}
      {useTwoStyles && (
        <div
          className="custom-dropdown"
          onMouseEnter={() => setOpen2(true)}
          onMouseLeave={() => setOpen2(false)}
        >
          <div className="dropdown-label">Stil 2: {style2 || "Seçim yapılmadı"}</div>
          {open2 && (
            <div className="dropdown-menu">
              {styles.map((style) => (
                <div
                  key={style.name}
                  className="dropdown-option"
                  onClick={() => setStyle2(style.name)} // ✅ Burada da sadece string gönderiliyor
                >
                  {style.name}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ThemeSelector;
