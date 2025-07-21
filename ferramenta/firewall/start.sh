#!/bin/sh

source /venv/bin/activate

echo "Starting Firewall Manager..."
/venv/bin/python /app/firewall_manager.py

sleep 5

echo "Starting Nginx..."
nginx -g "daemon off;"
