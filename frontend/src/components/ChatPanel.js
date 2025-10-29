// src/components/ChatPanel.js

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
    Box, Typography, TextField, Button, Paper, List, CircularProgress, 
    FormControlLabel, Switch, Alert, Tooltip 
} from '@mui/material';
import { Send } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatMessage = ({ message }) => (
    <Box sx={{ mb: 2, display: 'flex', justifyContent: message.isUser ? 'flex-end' : 'flex-start' }}>
        <Paper 
            elevation={2}
            sx={{ 
                p: 1.5, 
                bgcolor: message.isUser ? 'primary.main' : 'background.paper',
                color: message.isUser ? 'white' : 'text.primary',
                maxWidth: '90%',
                borderRadius: message.isUser ? '20px 20px 5px 20px' : '20px 20px 20px 5px',
            }}
        >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
        </Paper>
    </Box>
);

const ChatPanel = ({ sessionId, uploadedFiles, setIsAnalyzing, setAnalysisResult }) => {
    const [messages, setMessages] = useState([
        { text: "Hello! Ask a question or switch to **Deep Analysis** mode to begin.", isUser: false }
    ]);
    const [input, setInput] = useState('');
    const [analysisMode, setAnalysisMode] = useState('simple');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const chatEndRef = useRef(null);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = { text: input, isUser: true };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);
        setIsAnalyzing(true);
        setError('');
        setAnalysisResult(null);

        const payload = {
            message: input,
            session_id: sessionId,
            file_context: uploadedFiles.map(f => f.file_id),
            analysis_mode: analysisMode,
        };

        try {
            if (analysisMode === 'simple') {
                const response = await axios.post('/api/v2/chat/simple', payload);
                setMessages(prev => [...prev, { text: response.data.response, isUser: false }]);
                setIsLoading(false);
                setIsAnalyzing(false);
            } else {
                const response = await axios.post('/api/v2/analyze/comprehensive', payload);
                const taskId = response.data.task_id;
                setMessages(prev => [...prev, { text: `Roger that! Starting comprehensive analysis (Task ID: ${taskId}). I will check for results periodically.`, isUser: false }]);
                pollTaskStatus(taskId);
            }
        } catch (err) {
            const errorMsg = err.response?.data?.detail || "An unexpected error occurred.";
            setError(errorMsg);
            setMessages(prev => [...prev, { text: `Error: ${errorMsg}`, isUser: false }]);
            setIsLoading(false);
            setIsAnalyzing(false);
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
                    setMessages(prev => [...prev, { text: `**Analysis Complete!** The results are now displayed on the dashboard.`, isUser: false }]);
                    setIsLoading(false);
                    setIsAnalyzing(false);
                } else if (res.data.status === 'FAILURE') {
                    clearInterval(interval);
                    const errorMsg = res.data.result || "Unknown error during analysis.";
                    setError(errorMsg);
                    setMessages(prev => [...prev, { text: `Analysis failed: ${errorMsg}`, isUser: false }]);
                    setIsLoading(false);
                    setIsAnalyzing(false);
                }
                // If status is PENDING or other, do nothing and wait for the next poll.
            } catch (err) {
                clearInterval(interval);
                setError("Could not retrieve analysis status.");
                setIsLoading(false);
                setIsAnalyzing(false);
            }
        }, 5000); // Poll every 5 seconds
    };

    return (
        <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h5">Conversational AI</Typography>
                <Tooltip title={analysisMode === 'simple' ? "Switch to deep analysis mode" : "Switch to simple Q&A mode"}>
                    <FormControlLabel
                        control={<Switch color="primary" checked={analysisMode === 'comprehensive'} onChange={(e) => setAnalysisMode(e.target.checked ? 'comprehensive' : 'simple')} />}
                        label="Deep Analysis"
                    />
                </Tooltip>
            </Box>

            <Box sx={{ flexGrow: 1, p: 2, overflowY: 'auto' }}>
                <List>
                    {messages.map((msg, i) => <ChatMessage key={i} message={msg} />)}
                    {isLoading && <CircularProgress sx={{ display: 'block', margin: 'auto', mt: 2 }}/>}
                </List>
                <div ref={chatEndRef} />
            </Box>

            {error && <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>}

            <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                        fullWidth
                        variant="outlined"
                        placeholder="Ask a question about your data..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSend()}
                        disabled={isLoading}
                    />
                    <Button variant="contained" onClick={handleSend} disabled={isLoading} sx={{ px: 3 }}>
                        <Send />
                    </Button>
                </Box>
            </Box>
        </Paper>
    );
};

export default ChatPanel;
