// src/App.js

import React, { useState, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { Box, CssBaseline } from '@mui/material';

import Sidebar from './components/Sidebar';
import DataManagerPage from './pages/DataManagerPage';
import AnalysisStudioPage from './pages/AnalysisStudioPage';

// A modern, professional theme for the application
const modernTheme = createTheme({
    palette: {
        primary: { main: '#1976D2' }, // A professional blue
        secondary: { main: '#FFC107' }, // A warm, contrasting amber
        background: { default: '#f4f6f8', paper: '#ffffff' },
    },
    typography: {
        fontFamily: 'Roboto, sans-serif',
        h4: { fontWeight: 600, color: '#1a237e' },
        h5: { fontWeight: 500, color: '#263238' },
    },
    shape: {
        borderRadius: 8,
    },
    components: {
        MuiPaper: {
            styleOverrides: {
                root: {
                    boxShadow: '0px 4px 12px rgba(0,0,0,0.05)',
                },
            },
        },
    },
});

function App() {
    // Central state for the application
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // A unique session ID for the user's visit
    const sessionId = useMemo(() => `session_${Date.now()}` , []);

    return (
        <ThemeProvider theme={modernTheme}>
            <CssBaseline />
            <Router>
                <Box sx={{ display: 'flex', height: '100vh' }}>
                    <Sidebar fileCount={uploadedFiles.length} isAnalyzing={isAnalyzing} />
                    <Box component="main" sx={{ flexGrow: 1, p: 3, height: '100vh', overflow: 'auto' }}>
                        <Routes>
                            <Route path="/" element={<Navigate to="/data" replace />} />
                            <Route 
                                path="/data" 
                                element={<DataManagerPage uploadedFiles={uploadedFiles} setUploadedFiles={setUploadedFiles} />} 
                            />
                            <Route 
                                path="/analysis" 
                                element={<AnalysisStudioPage 
                                    sessionId={sessionId}
                                    uploadedFiles={uploadedFiles}
                                    isAnalyzing={isAnalyzing}
                                    setIsAnalyzing={setIsAnalyzing}
                                    analysisResult={analysisResult}
                                    setAnalysisResult={setAnalysisResult}
                                />} 
                            />
                        </Routes>
                    </Box>
                </Box>
            </Router>
        </ThemeProvider>
    );
}

export default App;