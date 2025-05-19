import {useMutation, useQueryClient} from '@tanstack/react-query';
import {generateImage, fetchGallery} from '../api';

export const useGenerateImage = () => {
    const qc = useQueryClient();
    return useMutation({
        mutationFn: generateImage,
        onSuccess: imgPath => {
            qc.invalidateQueries(['gallery']);
        }
    });
};