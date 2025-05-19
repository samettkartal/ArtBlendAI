import React, { useEffect } from 'react';
import { useForm, Controller } from 'react-hook-form';
import {
    Container,
    Grid,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Button,
    CircularProgress,
    Snackbar,
    Alert
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';

import { useStylesList } from '../hooks/useStylesList.js';
import { useGenerateImage } from '../hooks/useGenerateImage';

export default function PromptForm({ onGenerated }) {
    const {
        control,
        watch,
        setValue,
        handleSubmit,
        formState: { isValid }
    } = useForm({
        mode: 'onChange',
        defaultValues: {
            prompt: '',
            blend_mode: 'style1',
            style1: '',
            style2: ''
        }
    });

    const blend_mode = watch('blend_mode');
    const selectedStyle1 = watch('style1');
    const selectedStyle2 = watch('style2');

    const { data: styles = [], isLoading: stylesLoading } = useStylesList();
    const { mutateAsync, isLoading: generating, error } = useGenerateImage();

    // On styles load, set the first style as default for both style1 and style2 (when mixing)
    useEffect(() => {
        if (!stylesLoading && styles.length > 0) {
            if (!selectedStyle1) {
                setValue('style1', styles[0].name);
            }
            if (blend_mode === 'mix' && !selectedStyle2) {
                setValue('style2', styles[0].name);
            }
        }
    }, [stylesLoading, styles, blend_mode, selectedStyle1, selectedStyle2, setValue]);

    const onSubmit = async (vals) => {
        const payload = {
            prompt: vals.prompt,
            blend_mode: vals.blend_mode,
            style1: vals.style1,
            style2: vals.blend_mode === 'mix' ? vals.style2 : ''
        };
        const imgPath = await mutateAsync(payload);
        onGenerated(imgPath);
    };

    const iconStyles = {
        '& .MuiSelect-icon': {
            right: 12,
            left: 'auto',
            top: '50%',
            transform: 'translateY(-50%)'
        }
    };

    return (
        <Container maxWidth="md" sx={{ py: 4 }}>
            <form onSubmit={handleSubmit(onSubmit)} noValidate>
                <Grid container spacing={2} sx={{ width: '100%', maxWidth: 800, mx: 'auto' }}>
                    <Grid item xs={8}>
                        <Controller
                            name="prompt"
                            control={control}
                            rules={{ required: 'Proje adı gerekli' }}
                            render={({ field, fieldState }) => (
                                <TextField
                                    {...field}
                                    label="Proje Adı"
                                    placeholder="Örn: Kedim, Bahar Manzarası"
                                    fullWidth
                                    error={!!fieldState.error}
                                    helperText={fieldState.error?.message}
                                />
                            )}
                        />
                    </Grid>
                    <Grid item xs={4}>
                        <Controller
                            name="blend_mode"
                            control={control}
                            render={({ field }) => (
                                <FormControl fullWidth variant="outlined" sx={{ position: 'relative' }}>
                                    <InputLabel id="blend-label">Stilleri karıştır?</InputLabel>
                                    <Select
                                        {...field}
                                        labelId="blend-label"
                                        label="Stilleri karıştır?"
                                        IconComponent={ArrowDropDownIcon}
                                        sx={iconStyles}
                                    >
                                        <MenuItem value="style1">Stil 1</MenuItem>
                                        <MenuItem value="style2">Stil 2</MenuItem>
                                        <MenuItem value="mix">Karıştır</MenuItem>
                                    </Select>
                                </FormControl>
                            )}
                        />
                    </Grid>
                </Grid>

                <Grid container spacing={2} mt={3} sx={{ width: '100%', maxWidth: 800, mx: 'auto' }}>
                    <Grid item xs={6}>
                        <Controller
                            name="style1"
                            control={control}
                            rules={{ required: 'Stil 1 gerekli' }}
                            render={({ field, fieldState }) => (
                                <FormControl fullWidth variant="outlined" disabled={stylesLoading} sx={{ position: 'relative' }}>
                                    <InputLabel id="style1-label">Stil 1</InputLabel>
                                    <Select
                                        {...field}
                                        labelId="style1-label"
                                        label="Stil 1"
                                        IconComponent={ArrowDropDownIcon}
                                        sx={iconStyles}
                                    >
                                        {styles.map((s) => (
                                            <MenuItem key={s.name} value={s.name}>
                                                {s.name}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                    {stylesLoading && (
                                        <CircularProgress
                                            size={24}
                                            sx={{ position: 'absolute', top: '50%', right: 40, transform: 'translateY(-50%)' }}
                                        />
                                    )}
                                </FormControl>
                            )}
                        />
                    </Grid>

                    {blend_mode === 'mix' && (
                        <Grid item xs={6}>
                            <Controller
                                name="style2"
                                control={control}
                                rules={{ required: 'Stil 2 gerekli' }}
                                render={({ field, fieldState }) => (
                                    <FormControl fullWidth variant="outlined" disabled={stylesLoading} sx={{ position: 'relative' }}>
                                        <InputLabel id="style2-label">Stil 2</InputLabel>
                                        <Select
                                            {...field}
                                            labelId="style2-label"
                                            label="Stil 2"
                                            IconComponent={ArrowDropDownIcon}
                                            sx={iconStyles}
                                        >
                                            {styles.map((s) => (
                                                <MenuItem key={s.name} value={s.name}>
                                                    {s.name}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                        {stylesLoading && (
                                            <CircularProgress
                                                size={24}
                                                sx={{ position: 'absolute', top: '50%', right: 40, transform: 'translateY(-50%)' }}
                                            />
                                        )}
                                    </FormControl>
                                )}
                            />
                        </Grid>
                    )}
                </Grid>

                <Grid container spacing={2} sx={{ width: '100%', maxWidth: 800, mx: 'auto', mt: 2 }}>
                    <Grid item xs={12} sx={{ textAlign: 'center' }}>
                        <Button
                            type="submit"
                            variant="contained"
                            size="large"
                            disabled={!isValid || generating}
                            startIcon={generating && <CircularProgress color="inherit" size={20} />}
                        >
                            {generating ? 'Oluşturuluyor…' : 'Görseli Üret'}
                        </Button>
                    </Grid>
                </Grid>
            </form>

            <Snackbar open={!!error} autoHideDuration={6000}>
                <Alert severity="error">{error?.message || 'Hata oluştu'}</Alert>
            </Snackbar>
        </Container>
    );
}
