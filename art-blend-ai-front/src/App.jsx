import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import PromptForm from './components/PromptForm';
import ImageDisplay from './components/ImageDisplay';
import GeneratedGallery from './components/GeneratedGallery';

export default function App() {
    const [latestImage, setLatestImage] = React.useState(null);

    return (
        <Container maxWidth="sm" sx={{ py: 4 }}>
            <Typography
                variant="h4"
                align="center"
                gutterBottom
                sx={{ fontFamily: 'monospace', fontWeight: 'bold' }}
                mb={10}
            >
                Yapay Zeka Görsel Üretici
            </Typography>

            <PromptForm onGenerated={setLatestImage} />

            {latestImage && (
                <Box sx={{ mt: 4 }}>
                    <ImageDisplay imageUrl={`http://localhost:8000/outputs/${latestImage}`} />
                </Box>
            )}

            <GeneratedGallery sx={{ mt: 6 }} />
        </Container>
    );
}
