// src/pages/DataManagerPage.js

import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { 
    Box, Typography, Button, Paper, Table, TableBody, TableCell, 
    TableContainer, TableHead, TableRow, IconButton, Alert, LinearProgress, Tooltip
} from '@mui/material';
import { UploadFile, Delete, CheckCircle } from '@mui/icons-material';

const DataManagerPage = ({ uploadedFiles, setUploadedFiles }) => {
    const [error, setError] = useState('');
    const [uploading, setUploading] = useState(false);

    const handleFileUpload = async (files) => {
        setError('');
        setUploading(true);
        const newFiles = [];
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await axios.post('/api/files/upload', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });
                newFiles.push(response.data);
            } catch (err) {
                setError(err.response?.data?.detail || `Upload failed for ${file.name}.`);
                // Stop on first error
                setUploading(false);
                return;
            }
        }
        setUploadedFiles(prev => [...prev, ...newFiles]);
        setUploading(false);
    };

    const onDragOver = (e) => e.preventDefault();
    const onDrop = (e) => {
        e.preventDefault();
        handleFileUpload(e.dataTransfer.files);
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom>Data Manager</Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Upload and manage the datasets for your analysis. The system supports multiple CSV and Excel files for temporal comparisons.
            </Typography>

            <Paper 
                onDragOver={onDragOver} 
                onDrop={onDrop}
                sx={{ 
                    border: '2px dashed', 
                    borderColor: 'grey.300', 
                    textAlign: 'center', 
                    p: 4, 
                    mb: 4, 
                    backgroundColor: '#fafafa',
                    cursor: 'pointer'
                }}
            >
                <UploadFile sx={{ fontSize: 50, color: 'grey.500' }} />
                <Typography>Drag & Drop files here or</Typography>
                <Button variant="contained" component="label" sx={{ mt: 2 }} disabled={uploading}>
                    Browse Files
                    <input type="file" hidden multiple onChange={(e) => handleFileUpload(e.target.files)} />
                </Button>
                {uploading && <LinearProgress sx={{ mt: 2 }} />}
            </Paper>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            <Typography variant="h5" gutterBottom>Available Datasets</Typography>
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Status</TableCell>
                            <TableCell>File Name</TableCell>
                            <TableCell>File Size</TableCell>
                            <TableCell>Rows</TableCell>
                            <TableCell>Columns</TableCell>
                            <TableCell>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {uploadedFiles.length > 0 ? (
                            uploadedFiles.map((file) => (
                                <TableRow key={file.file_id}>
                                    <TableCell><Tooltip title="Ready for analysis"><CheckCircle color="success" /></Tooltip></TableCell>
                                    <TableCell>{file.filename}</TableCell>
                                    <TableCell>{(file.file_size / 1024).toFixed(2)} KB</TableCell>
                                    <TableCell>{file.row_count}</TableCell>
                                    <TableCell>{file.columns.length}</TableCell>
                                    <TableCell>
                                        <IconButton edge="end" disabled><Delete /></IconButton>
                                    </TableCell>
                                </TableRow>
                            ))
                        ) : (
                            <TableRow>
                                <TableCell colSpan={6} align="center">No files uploaded yet.</TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
};

export default DataManagerPage;
