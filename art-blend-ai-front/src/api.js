import axios from 'axios';
import stylesData from './data/styles.json';
const API = axios.create({ baseURL: 'http://localhost:8000' });

export const fetchStyles = () => {
    const formattedStyles = Object.entries(stylesData).map(([name, id]) => ({ name, id }));
    return Promise.resolve(formattedStyles);
}

export const generateImage = payload =>
    API.post('/generate', payload).then(res => res.data.image_path);

export const fetchGallery = () =>
    API.get('/generated').then(res => res.data)
