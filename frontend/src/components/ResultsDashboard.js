// src/components/ResultsDashboard.js

import React, { useState } from 'react';
import { Box, Typography, Paper, Tabs, Tab, Alert } from '@mui/material';
import { Summarize, DataObject, Insights, Recommend } from '@mui/icons-material';

const TabPanel = (props) => {
    const { children, value, index, ...other } = props;
    return (
        <div role="tabpanel" hidden={value !== index} {...other}>
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
};

const ResultsDashboard = ({ result }) => {
    const [tab, setTab] = useState(0);

    if (!result) {
        return (
            <Paper sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', backgroundColor: '#f5f5f5' }}>
                <Box sx={{textAlign: 'center'}}>
                    <Insights sx={{ fontSize: 60, color: 'grey.400' }} />
                    <Typography variant="h6" color="text.secondary">Results Dashboard</Typography>
                    <Typography color="text.secondary">Your analysis results and visualizations will appear here.</Typography>
                </Box>
            </Paper>
        );
    }

    return (
        <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tab} onChange={(e, newValue) => setTab(newValue)} aria-label="results tabs" variant="fullWidth">
                    <Tab icon={<Summarize />} label="Summary" />
                    <Tab icon={<Recommend />} label="Recommendations" />
                    <Tab icon={<DataObject />} label="Raw Data" />
                </Tabs>
            </Box>

            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                <TabPanel value={tab} index={0}>
                    <Typography variant="h5" gutterBottom>Executive Summary</Typography>
                    <Typography paragraph>{result.executive_summary || "No summary available."}</Typography>
                    <Typography variant="h6" sx={{ mt: 3 }}>Key Findings</Typography>
                    <ul>{result.key_findings?.map((item, i) => <li key={i}><Typography>{item}</Typography></li>) || <li>No findings available.</li>}</ul>
                </TabPanel>
                
                <TabPanel value={tab} index={1}>
                    <Typography variant="h5" gutterBottom>Recommended Actions</Typography>
                    <ul>{result.recommended_actions?.map((item, i) => <li key={i}><Typography>{item}</Typography></li>) || <li>No actions recommended.</li>}</ul>
                </TabPanel>

                <TabPanel value={tab} index={2}>
                    <Typography variant="h5" gutterBottom>Raw JSON Output</Typography>
                    <Paper component="pre" variant="outlined" sx={{ p: 2, maxHeight: '60vh', overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all', background: '#efefef' }}>
                        {JSON.stringify(result, null, 2)}
                    </Paper>
                </TabPanel>
            </Box>
        </Paper>
    );
};

export default ResultsDashboard;