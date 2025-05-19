import {useQuery} from '@tanstack/react-query';
import {fetchStyles} from '../api';

export const useStylesList = () =>
    useQuery({
        queryKey: ['styles'],
        queryFn: fetchStyles,
        staleTime: 1000 * 60 * 5,
        retry: 1
    });
