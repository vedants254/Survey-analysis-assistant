// src/pages/AnalysisStudioPage.js

import React from 'react';
import { Box, Grid, Paper, Typography, Alert } from '@mui/material';

import ChatPanel from '../components/ChatPanel';
import ResultsDashboard from '../components/ResultsDashboard';

const AnalysisStudioPage = (props) => {
    const { uploadedFiles, isAnalyzing, analysisResult } = props;

    if (uploadedFiles.length === 0) {
        return (
            <Alert severity="info">
                Please upload at least one data file in the "Data Manager" tab to begin your analysis.
            </Alert>
        );
    }

    return (
        <Box sx={{ height: 'calc(100vh - 100px)' }}>
            <Typography variant="h4" gutterBottom>Analysis Studio</Typography>
            <Grid container spacing={3} sx={{ height: '100%' }}>
                {/* Left Panel: Chat and Controls */}
                <Grid item xs={12} md={5} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <ChatPanel {...props} />
                </Grid>

                {/* Right Panel: Results and Visualizations */}
                <Grid item xs={12} md={7} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Paper sx={{ flexGrow: 1, p: 2, overflow: 'auto' }}>
                        {isAnalyzing && !analysisResult ? (
                            <Box>Loading analysis...</Box> // Replace with a proper progress component
                        ) : analysisResult ? (
                            <ResultsDashboard result={analysisResult} />
                        ) : (
                            <Box sx={{textAlign: 'center', mt: 4}}>
                                <Typography variant="h6" color="text.secondary">Results Dashboard</Typography>
                                <Typography color="text.secondary">Your analysis results and visualizations will appear here.</Typography>
                            </Box>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default AnalysisStudioPage;
