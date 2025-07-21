#!/bin/sh

source /venv/bin/activate

# Start the firewall manager in the background
echo "Starting Firewall Manager..."
/venv/bin/python /app/firewall_manager.py

# Wait a moment for the firewall manager to start
sleep 5

# Start nginx in the foreground
echo "Starting Nginx..."
nginx -g "daemon off;"