// src/components/Navigation.js

import React from 'react';
import { NavLink } from 'react-router-dom';
import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider, Typography } from '@mui/material';
import { UploadFile, Chat, Analytics } from '@mui/icons-material';

const navItems = [
    { text: 'File Management', icon: <UploadFile />, path: '/upload' },
    { text: 'AI Assistant', icon: <Chat />, path: '/chat' },
];

const Navigation = () => {
    const navLinkStyles = ({ isActive }) => ({
        display: 'flex',
        padding: '10px 15px',
        borderRadius: '8px',
        textDecoration: 'none',
        color: isActive ? 'white' : '#e0e0e0',
        backgroundColor: isActive ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
        marginBottom: '5px',
        '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
        },
    });

    return (
        <Box
            sx={{
                width: 240,
                flexShrink: 0,
                bgcolor: 'primary.main',
                color: 'white',
                height: '100vh',
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Analytics sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h6">Data Analysis</Typography>
            </Box>
            <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.2)' }} />
            <List>
                {navItems.map((item) => (
                    <ListItem key={item.text} disablePadding>
                        <ListItemButton component={NavLink} to={item.path} sx={navLinkStyles}>
                            <ListItemIcon sx={{ color: 'inherit' }}>{item.icon}</ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );
};

export default Navigation;
