// src/pages/ChatPage.js

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Box, Typography, TextField, Button, Paper, List, CircularProgress, FormControlLabel, Switch, Alert } from '@mui/material';
import { Send } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ChatMessage = ({ message }) => (
    <Box sx={{ mb: 2, display: 'flex', justifyContent: message.isUser ? 'flex-end' : 'flex-start' }}>
        <Paper 
            variant="outlined" 
            sx={{ 
                p: 2, 
                bgcolor: message.isUser ? 'primary.main' : 'background.paper',
                color: message.isUser ? 'white' : 'text.primary',
                maxWidth: '80%',
                borderRadius: message.isUser ? '20px 20px 5px 20px' : '20px 20px 20px 5px',
            }}
        >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
        </Paper>
    </Box>
);

const ResultsDisplay = ({ result }) => {
    if (!result) return null;

    // Example of trying to find data suitable for a line chart
    const chartData = result.analysis_details?.execution_results?.results;

    return (
        <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
            <Typography variant="h5" gutterBottom>Analysis Results</Typography>
            <Typography variant="h6">Executive Summary</Typography>
            <Typography paragraph>{result.executive_summary}</Typography>

            <Typography variant="h6">Key Findings</Typography>
            <ul>
                {result.key_findings.map((item, i) => <li key={i}>{item}</li>)}
            </ul>

            {chartData && typeof chartData === 'object' &&
                <Box sx={{ height: 300, mt: 4 }}>
                    <Typography variant="h6">Visualized Data</Typography>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={Object.keys(chartData).map(key => ({ name: key, value: chartData[key] }))}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="value" stroke="#1E88E5" activeDot={{ r: 8 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </Box>
            }
        </Paper>
    );
};

const ChatPage = ({ sessionId, uploadedFiles, isAnalyzing, setIsAnalyzing, analysisResult, setAnalysisResult }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [analysisMode, setAnalysisMode] = useState('simple');
    const chatEndRef = useRef(null);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = { text: input, isUser: true };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsAnalyzing(true);

        const payload = {
            message: input,
            session_id: sessionId,
            file_context: uploadedFiles.map(f => f.file_id),
            analysis_mode: analysisMode,
        };

        try {
            const endpoint = analysisMode === 'simple' ? '/api/v2/chat/simple' : '/api/v2/analyze/comprehensive';
            const response = await axios.post(endpoint, payload);

            if (analysisMode === 'simple') {
                setMessages(prev => [...prev, { text: response.data.response, isUser: false }]);
            } else {
                const taskId = response.data.task_id;
                pollTaskStatus(taskId);
            }
        } catch (err) {
            const errorMsg = err.response?.data?.detail || "An error occurred.";
            setMessages(prev => [...prev, { text: `Error: ${errorMsg}`, isUser: false }]);
        } finally {
            if (analysisMode === 'simple') {
                setIsAnalyzing(false);
            }
        }
    };

    const pollTaskStatus = (taskId) => {
        const interval = setInterval(async () => {
            try {
                const res = await axios.get(`/api/v2/task/${taskId}/status`);
                if (res.data.status === 'SUCCESS') {
                    clearInterval(interval);
                    const result = res.data.result;
                    setAnalysisResult(result);
                    setMessages(prev => [...prev, { text: `**Analysis Complete!**\n\n${result.executive_summary}`, isUser: false }]);
                    setIsAnalyzing(false);
                }
                 else if (res.data.status === 'FAILURE') {
                    clearInterval(interval);
                    setMessages(prev => [...prev, { text: `Analysis failed: ${res.data.result}`, isUser: false }]);
                    setIsAnalyzing(false);
                }
            } catch (err) {
                clearInterval(interval);
                setMessages(prev => [...prev, { text: "Error fetching analysis status.", isUser: false }]);
                setIsAnalyzing(false);
            }
        }, 3000);
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4">AI Assistant</Typography>
                <FormControlLabel
                    control={<Switch checked={analysisMode === 'comprehensive'} onChange={(e) => setAnalysisMode(e.target.checked ? 'comprehensive' : 'simple')} />}
                    label="Comprehensive Analysis"
                />
            </Box>

            {uploadedFiles.length === 0 && <Alert severity="warning">Please upload files first to enable analysis.</Alert>}

            <Paper variant="outlined" sx={{ flexGrow: 1, p: 2, overflowY: 'auto', mb: 2 }}>
                <List>
                    {messages.map((msg, i) => <ChatMessage key={i} message={msg} />)}
                    {isAnalyzing && analysisMode === 'comprehensive' && <CircularProgress sx={{mt: 2}}/>}
                </List>
                <div ref={chatEndRef} />
            </Paper>

            <Box sx={{ display: 'flex' }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Ask a question about your data..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    disabled={isAnalyzing || uploadedFiles.length === 0}
                />
                <Button variant="contained" onClick={handleSend} disabled={isAnalyzing || uploadedFiles.length === 0} sx={{ ml: 1, px: 4 }}>
                    <Send />
                </Button>
            </Box>
            {analysisResult && <ResultsDisplay result={analysisResult} />}
        </Box>
    );
};

export default ChatPage;
