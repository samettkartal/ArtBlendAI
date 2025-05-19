import React, {useState} from 'react'; // Import useState
import {useQuery} from '@tanstack/react-query';
import {fetchGallery} from '../api';
import {
    Grid,
    Card,
    CardMedia,
    Typography,
    Box,
    CircularProgress,
    Modal, // Import Modal
    IconButton, // For the close button
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close'; // Import CloseIcon for the button

// Styles for the modal content
const modalStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    bgcolor: 'background.paper',
    border: '2px solid #000', // Optional: adds a border
    boxShadow: 24,
    p: 0, // Padding will be on the image container if needed, not the modal itself
    outline: 'none', // Remove default focus outline
    maxWidth: '90vw', // Max width of the modal
    maxHeight: '90vh', // Max height of the modal
    display: 'flex', // To center the image if it's smaller than maxHeight/maxWidth
    alignItems: 'center',
    justifyContent: 'center',
};

const closeButtonStyle = {
    position: 'absolute',
    top: 8,
    right: 8,
    color: 'white', // Or a color that contrasts with your images
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    '&:hover': {
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
    },
    zIndex: 1, // Ensure it's above the image
};

export default function GeneratedGallery(props) {
    const {data: imgs = [], isLoading} = useQuery({
        queryKey: ['gallery'],
        queryFn: fetchGallery,
        staleTime: 1000 * 60, // Keep data fresh for 1 minute
    });

    const [openModal, setOpenModal] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null);

    const handleOpenModal = (imgUrl) => {
        setSelectedImage(imgUrl);
        setOpenModal(true);
    };

    const handleCloseModal = () => {
        setOpenModal(false);
        setSelectedImage(null); // Clear selected image when closing
    };

    if (isLoading) {
        return (
            <Box textAlign="center" mt={4}>
                <CircularProgress/>
            </Box>
        );
    }
    if (!imgs.length) {
        // Optionally, you could show a message here if desired
        // return <Typography>No images generated yet.</Typography>;
        return null;
    }

    return (
        <Box {...props}>
            <Typography variant="h6" gutterBottom>🎨 Önceki Oluşturulan Görseller</Typography>
            <Grid container spacing={2}>
                {imgs.map((img, i) => (
                    <Grid key={i} item xs={4} sm={3} md={2}> {/* Adjusted grid sizing for better fit */}
                        <Card
                            sx={{
                                cursor: 'pointer',
                                '&:hover': {
                                    transform: 'scale(1.03)', // Slight zoom on hover for thumbnails
                                    transition: 'transform 0.2s ease-in-out',
                                },
                            }}
                            onClick={() => handleOpenModal(img)}
                        >
                            <CardMedia
                                component="img"
                                image={`${img}`} // Ensure img is a valid URL
                                alt={`Generated image ${i + 1}`} // More descriptive alt text
                                sx={{
                                    height: 100, // Increased height for better preview
                                    objectFit: 'cover',
                                }}
                            />
                        </Card>
                    </Grid>
                ))}
            </Grid>

            {/* Modal for Zoomed Image */}
            <Modal
                open={openModal}
                onClose={handleCloseModal}
                aria-labelledby="image-zoom-modal-title"
                aria-describedby="image-zoom-modal-description"
            >
                <Box sx={modalStyle}>
                    <IconButton
                        aria-label="close"
                        onClick={handleCloseModal}
                        sx={closeButtonStyle}
                    >
                        <CloseIcon/>
                    </IconButton>
                    {selectedImage && (
                        <img
                            src={selectedImage}
                            alt="Zoomed generated image"
                            style={{
                                maxWidth: '100%', // Image will scale to fit modal width
                                maxHeight: '100%', // Image will scale to fit modal height
                                objectFit: 'contain', // Show the whole image, don't crop
                                display: 'block', // Remove extra space below image if any
                            }}
                        />
                    )}
                    {/* <Typography id="image-zoom-modal-title" variant="h6" component="h2" sx={{ display: 'none' }}>Zoomed Image</Typography> */}
                    {/* <Typography id="image-zoom-modal-description" sx={{ display: 'none' }}>A larger view of the selected generated image.</Typography> */}
                </Box>
            </Modal>
        </Box>
    );
}