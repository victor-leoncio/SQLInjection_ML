#!/bin/bash

case "$1" in
    "start")
        echo "🚀 Starting SQL Injection Detection System..."
        docker-compose up -d --build
        echo "✅ System started! Access: http://localhost"
        ;;
    "stop")
        echo "🛑 Stopping SQL Injection Detection System..."
        docker-compose down
        echo "✅ System stopped!"
        ;;
    "restart")
        echo "🔄 Restarting SQL Injection Detection System..."
        docker-compose down
        docker-compose up -d --build
        echo "✅ System restarted!"
        ;;
    "logs")
        if [ -z "$2" ]; then
            echo "📋 Showing all logs..."
            docker-compose logs -f
        else
            echo "📋 Showing logs for $2..."
            docker-compose logs -f "$2"
        fi
        ;;
    "status")
        echo "📊 System Status:"
        docker-compose ps
        echo ""
        echo "🔍 Health Checks:"
        echo "Firewall: $(curl -s http://localhost/firewall/health | jq -r '.status' 2>/dev/null || echo 'Not responding')"
        echo "Web App: $(docker-compose exec -T web curl -s http://localhost:8000/ >/dev/null 2>&1 && echo 'OK' || echo 'Not responding')"
        echo "SQL Detector: $(docker-compose exec -T sql_detect curl -s http://localhost:7000/health >/dev/null 2>&1 && echo 'OK' || echo 'Not responding')"
        ;;
    "blocked-ips")
        echo "🚫 Currently Blocked IPs:"
        curl -s http://localhost/firewall/blocked-ips | jq '.' 2>/dev/null || echo "Could not retrieve blocked IPs"
        ;;
    "unblock")
        if [ -z "$2" ]; then
            echo "❌ Please provide an IP address to unblock"
            echo "Usage: $0 unblock <IP_ADDRESS>"
        else
            echo "🔓 Unblocking IP: $2"
            docker exec sql_injection_firewall curl -X POST http://localhost:8080/unblock-ip \
                -H "Content-Type: application/json" \
                -d "{\"ip\": \"$2\"}"
            echo ""
        fi
        ;;
    "test")
        echo "🧪 Testing SQL Injection Detection..."
        echo "This will attempt a SQL injection and should block your IP"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            curl -X POST http://localhost/login \
                -H "Content-Type: application/x-www-form-urlencoded" \
                -d "username=admin' OR '1'='1&password=test"
            echo "Check if your IP was blocked with: $0 blocked-ips"
        fi
        ;;
    "clean")
        echo "🧹 Cleaning up system..."
        docker-compose down -v
        docker system prune -f
        echo "✅ System cleaned!"
        ;;
    *)
        echo "🛡️  SQL Injection Detection System Manager"
        echo ""
        echo "Usage: $0 {command} [options]"
        echo ""
        echo "Commands:"
        echo "  start          - Start the system"
        echo "  stop           - Stop the system"
        echo "  restart        - Restart the system"
        echo "  logs [service] - Show logs (all or specific service)"
        echo "  status         - Show system status"
        echo "  blocked-ips    - Show currently blocked IPs"
        echo "  unblock <ip>   - Unblock a specific IP"
        echo "  test           - Test SQL injection detection"
        echo "  clean          - Clean up system and volumes"
        echo ""
        echo "Services: firewall, web, sql_detect"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs firewall"
        echo "  $0 unblock 192.168.1.100"
        ;;
esac
