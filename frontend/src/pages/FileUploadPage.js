// src/pages/FileUploadPage.js

import React, { useState } from 'react';
import axios from 'axios';
import { Box, Typography, Button, Paper, List, ListItem, ListItemText, ListItemIcon, IconButton, Alert } from '@mui/material';
import { UploadFile, Description, Delete } from '@mui/icons-material';

const FileUploadPage = ({ uploadedFiles, setUploadedFiles }) => {
    const [error, setError] = useState('');

    const handleFileUpload = async (files) => {
        setError('');
        const formData = new FormData();
        for (const file of files) {
            formData.append('file', file);
        }

        try {
            const response = await axios.post('/api/files/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setUploadedFiles(prev => [...prev, response.data]);
        } catch (err) {
            setError(err.response?.data?.detail || 'File upload failed. Please try again.');
        }
    };

    const onDragOver = (e) => e.preventDefault();
    const onDrop = (e) => {
        e.preventDefault();
        handleFileUpload(e.dataTransfer.files);
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom>File Management</Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Upload the CSV or Excel files you want to analyze. You can upload multiple files for temporal analysis.
            </Typography>

            <Paper 
                onDragOver={onDragOver} 
                onDrop={onDrop}
                sx={{ 
                    border: '2px dashed', 
                    borderColor: 'grey.400', 
                    textAlign: 'center', 
                    p: 4, 
                    mb: 3 
                }}
            >
                <UploadFile sx={{ fontSize: 60, color: 'grey.500' }} />
                <Typography>Drag & Drop files here or</Typography>
                <Button variant="contained" component="label" sx={{ mt: 2 }}>
                    Browse Files
                    <input type="file" hidden multiple onChange={(e) => handleFileUpload(e.target.files)} />
                </Button>
            </Paper>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            <Typography variant="h5" gutterBottom>Uploaded Files</Typography>
            <Paper variant="outlined">
                <List>
                    {uploadedFiles.length > 0 ? (
                        uploadedFiles.map((file, index) => (
                            <ListItem key={index} secondaryAction={<IconButton edge="end" aria-label="delete"><Delete /></IconButton>}>
                                <ListItemIcon><Description /></ListItemIcon>
                                <ListItemText 
                                    primary={file.filename} 
                                    secondary={`Size: ${(file.file_size / 1024).toFixed(2)} KB | Rows: ${file.row_count}`}
                                />
                            </ListItem>
                        ))
                    ) : (
                        <ListItem>
                            <ListItemText primary="No files uploaded yet." />
                        </ListItem>
                    )}
                </List>
            </Paper>
        </Box>
    );
};

export default FileUploadPage;
