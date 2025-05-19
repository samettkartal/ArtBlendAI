import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import App from './App';

const queryClient = new QueryClient();
const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: '#646cff' },
        secondary: { main: '#61dafb' },
        background: { default: '#35414a', paper: '#485761' },
        text: { primary: '#fff' },
    },
    typography: {
        fontFamily: `'PFSquareSansPro-Bold', monospace`,
    },
});

ReactDOM.createRoot(document.getElementById('root')).render(
    <ThemeProvider theme={theme}>
        <CssBaseline />
        <QueryClientProvider client={queryClient}>
            <App />
        </QueryClientProvider>
    </ThemeProvider>
);
