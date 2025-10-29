// src/components/Sidebar.js

import React from 'react';
import { NavLink } from 'react-router-dom';
import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider, Typography, Badge, CircularProgress } from '@mui/material';
import { Folder, Science, Analytics } from '@mui/icons-material';

const navItems = [
    { text: 'Data Manager', icon: <Folder />, path: '/data' },
    { text: 'Analysis Studio', icon: <Science />, path: '/analysis' },
];

const Sidebar = ({ fileCount, isAnalyzing }) => {

    const navLinkStyles = ({ isActive }) => ({
        display: 'flex',
        padding: '12px 20px',
        margin: '4px 10px',
        borderRadius: '8px',
        textDecoration: 'none',
        color: isActive ? 'primary.main' : '#424242',
        backgroundColor: isActive ? '#e3f2fd' : 'transparent',
        transition: 'background-color 0.2s ease-in-out',
        '&.Mui-disabled': {
            color: '#bdbdbd',
            pointerEvents: 'none',
        },
        '&:hover': {
            backgroundColor: isActive ? '#e3f2fd' : '#f5f5f5',
        },
    });

    return (
        <Box
            sx={{
                width: 260,
                flexShrink: 0,
                bgcolor: 'background.paper',
                height: '100vh',
                display: 'flex',
                flexDirection: 'column',
                borderRight: '1px solid #e0e0e0'
            }}
        >
            <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2, justifyContent: 'center' }}>
                <Analytics sx={{ fontSize: 32, color: 'primary.main' }} />
                <Typography variant="h5" noWrap>AI Analyst</Typography>
            </Box>
            <Divider />
            <List sx={{ mt: 2 }}>
                {navItems.map((item) => (
                    <ListItem key={item.text} disablePadding>
                        <ListItemButton 
                            component={NavLink} 
                            to={item.path} 
                            sx={navLinkStyles}
                            disabled={item.path === '/analysis' && fileCount === 0}
                        >
                            <ListItemIcon>
                                <Badge badgeContent={item.path === '/data' ? fileCount : 0} color="secondary">
                                    {item.path === '/analysis' && isAnalyzing ? <CircularProgress size={24} /> : item.icon}
                                </Badge>
                            </ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );
};

export default Sidebar;
