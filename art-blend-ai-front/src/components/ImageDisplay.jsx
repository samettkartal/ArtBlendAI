import React from 'react';
import { Card, CardMedia, CardHeader } from '@mui/material';

export default function ImageDisplay({ imageUrl }) {
  return (
      <Card>
        <CardHeader title="Oluşturulan Görsel" />
        <CardMedia
            component="img"
            image={imageUrl}
            alt="Üretilen görsel"
            sx={{ maxHeight: 400, objectFit: 'contain' }}
        />
      </Card>
  );
}
