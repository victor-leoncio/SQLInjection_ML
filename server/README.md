# SQL Injection Detection System with Firewall

Este projeto implementa um sistema de detecção de SQL Injection usando Machine Learning com uma arquitetura de firewall em camadas.

## Arquitetura

```
Internet → Firewall Container (Nginx + iptables) → Web Container → SQL Detection API
```

### Componentes:

1. **Firewall Container** (`firewall/`)
   - Nginx como proxy reverso
   - Gerenciador de firewall com iptables
   - Rate limiting e headers de segurança
   - API para bloqueio/desbloqueio de IPs

2. **Web Container** (`web/`)
   - Aplicação Flask com interface web
   - Formulários de login e teste de SQL
   - Comunicação com API de detecção

3. **SQL Detection API** (`sql_detect/`)
   - FastAPI com modelo de Machine Learning
   - Detecção de SQL Injection usando Word2Vec + Random Forest
   - Comunicação automática com o firewall

## Como Usar

### 1. Construir e Executar

```bash
# Construir e executar todos os containers
docker-compose up --build

# Executar em background
docker-compose up -d --build
```

### 2. Acessar a Aplicação

- **Interface Web**: http://localhost
- **API de Detecção**: http://localhost/firewall/health (health check)
- **IPs Bloqueados**: http://localhost/firewall/blocked-ips

### 3. Testar SQL Injection

1. Acesse http://localhost
2. Tente fazer login com:
   - Username: `admin' OR '1'='1`
   - Password: `password`
3. O sistema detectará a injeção SQL e bloqueará seu IP automaticamente

### 4. Gerenciar Firewall

#### Verificar IPs Bloqueados
```bash
curl http://localhost/firewall/blocked-ips
```

#### Desbloquear um IP (apenas da rede interna)
```bash
docker exec sql_injection_firewall curl -X POST http://localhost:8080/unblock-ip \
  -H "Content-Type: application/json" \
  -d '{"ip": "SEU_IP_AQUI"}'
```

## Estrutura de Rede

- **Frontend Network** (172.20.0.0/16): Firewall ↔ Internet
- **Backend Network** (172.21.0.0/16): Firewall ↔ Web ↔ SQL Detection API

## Funcionalidades de Segurança

### Firewall Container
- ✅ Bloqueio automático de IPs via iptables
- ✅ Rate limiting (10 req/min para login, 30 req/min geral)
- ✅ Headers de segurança (XSS, CSRF, etc.)
- ✅ Proxy reverso com balanceamento
- ✅ Logs de acesso e erro

### Detecção ML
- ✅ Modelo Random Forest treinado
- ✅ Vetorização Word2Vec
- ✅ Detecção em tempo real
- ✅ Bloqueio automático via API

### Monitoramento
- ✅ Health checks para todos os serviços
- ✅ Logs centralizados
- ✅ Métricas de IPs bloqueados

## Comandos Úteis

### Ver logs
```bash
# Logs do firewall
docker-compose logs firewall

# Logs da aplicação web
docker-compose logs web

# Logs da API de detecção
docker-compose logs sql_detect
```

### Parar e remover
```bash
# Parar containers
docker-compose down

# Parar e remover volumes
docker-compose down -v
```

### Verificar status
```bash
# Status dos containers
docker-compose ps

# Health checks
docker-compose exec firewall curl http://localhost:8080/health
docker-compose exec web curl http://localhost:8000/
docker-compose exec sql_detect curl http://localhost:7000/health
```

## Desenvolvimento

### Estrutura de Arquivos
```
.
├── docker-compose.yml          # Orquestração dos containers
├── firewall/                   # Container do firewall
│   ├── dockerfile
│   ├── firewall_manager.py     # API de gerenciamento do firewall
│   ├── nginx.conf              # Configuração do Nginx
│   ├── requirements.txt
│   └── start.sh               # Script de inicialização
├── sql_detect/                # API de detecção
│   ├── dockerfile
│   ├── main.py
│   ├── model_rf_word2vec.pkl  # Modelo treinado
│   ├── model_word2vec.pkl     # Vetorizador
│   └── requirements.txt
└── web/                       # Aplicação web
    ├── dockerfile
    ├── main.py
    ├── requirements.txt
    └── templates/
        ├── index.html
        └── login.html
```

## Troubleshooting

### Container do firewall não inicia
- Verifique se o Docker tem privilégios para usar iptables
- Execute com `--privileged` se necessário

### IPs não são bloqueados
- Verifique os logs do firewall: `docker-compose logs firewall`
- Teste a comunicação entre containers: `docker-compose exec sql_detect curl http://firewall:8080/health`

### Aplicação não responde
- Verifique se todos os health checks estão passando
- Reinicie os containers: `docker-compose restart`

## Segurança

⚠️ **Importante**: Este sistema é para fins educacionais e de demonstração. Para produção:

1. Configure HTTPS/TLS
2. Use secrets para chaves sensíveis
3. Implemente autenticação robusta
4. Configure logs externos
5. Use redes isoladas
6. Implemente backup dos modelos ML