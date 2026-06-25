# Deployment Guide

## Quick Start (Local Development)

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Build and run with podman
./build.sh                    # Builds image and runs tests
podman compose up -d          # Starts all services

# 3. Test the API
curl http://localhost:8000/health

# 4. Start a conversation
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"message": "I need infrastructure provisioning"}'
```

**What gets started:**
- Redis (port 6379) - For LangGraph checkpoints
- FastAPI (port 8000) - REST API with 4 Gunicorn workers
- MongoDB - External (Atlas cloud — configure `MONGODB_URL` in `.env`)

**Build script (`build.sh`):**
- Installs dependencies with `uv`
- Runs all tests (unit + integration)
- Builds podman image with multi-stage build
- Tags with timestamp version
- **Fails if any test fails** ✅

---

## Production Deployment with podman

### Prerequisites

- podman 20.10+
- podman Compose 2.0+
- MongoDB Atlas account
- OpenAI API key

### Environment Variables Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your production values:**
   ```bash
   # Required - OpenAI API
   OPENAI_API_KEY=sk-your-actual-api-key-here
   
   # Required - MongoDB (use Atlas for production)
   MONGODB_URL=mongodb+srv://user:password@cluster.mongodb.net/
   MONGODB_DB_NAME=ai_chatbot_prod
   
   # Required - API Security
   API_KEY=your-secure-random-api-key-here
   
   # Optional - Redis (auto-configured in podman compose)
   REDIS_URL=redis://redis:6379
   
   # Optional - Feature flags
   DUPLICATE_DETECTION_ENABLED=true
   SEMANTIC_SEARCH_ENABLED=true
   ```

3. **Generate secure API key:**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

### Build and Run

1. **Build the podman image:**
   ```bash
   podman compose build
   ```

2. **Start all services:**
   ```bash
   podman compose up -d
   ```

3. **Check service health:**
   ```bash
   podman compose ps
   podman compose logs -f api
   ```

4. **Test the API:**
   ```bash
   curl -X GET http://localhost:8000/health \
     -H "X-API-Key: your-api-key-here"
   ```

### Production Best Practices

#### 1. Environment Variables

**Never commit `.env` files!** Use one of these approaches:

- **podman Secrets** (podman Swarm):
  ```yaml
  secrets:
    openai_api_key:
      external: true
  ```

- **Kubernetes Secrets**:
  ```bash
  kubectl create secret generic api-secrets \
    --from-literal=openai-api-key=sk-xxx \
    --from-literal=mongodb-url=mongodb+srv://xxx
  ```

- **AWS Secrets Manager / Azure Key Vault**:
  - Store secrets in cloud provider's secret management service
  - Use IAM roles for access control
  - Rotate secrets regularly

#### 2. Security Hardening

- **Use non-root user** ✅ (Already configured in podmanfile)
- **Scan for vulnerabilities:**
  ```bash
  podman scan ai-chatbot-api:latest
  ```

- **Enable TLS/SSL:**
  - Use reverse proxy (Nginx/Traefik) with Let's Encrypt
  - Or use cloud load balancer with SSL termination

- **Rate limiting:**
  - Configure at reverse proxy level
  - Or use FastAPI middleware

#### 3. Monitoring & Logging

- **View logs:**
  ```bash
  podman compose logs -f api
  podman compose logs -f redis
  ```

- **Export logs to external service:**
  ```yaml
  logging:
    driver: "json-file"
    options:
      max-size: "10m"
      max-file: "3"
  ```

- **Add monitoring:**
  - Prometheus + Grafana
  - DataDog / New Relic
  - CloudWatch / Azure Monitor

#### 4. Scaling

- **Horizontal scaling:**
  ```bash
  podman compose up -d --scale api=3
  ```

- **Load balancer:**
  ```yaml
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
  ```

#### 5. Backup & Recovery

- **MongoDB backups:**
  - Use MongoDB Atlas automated backups
  - Or set up mongodump cron job

- **Redis persistence:**
  - Already configured with AOF (append-only file)
  - Volume mounted for data persistence

### Updating the Application

1. **Pull latest changes:**
   ```bash
   git pull origin main
   ```

2. **Rebuild and restart:**
   ```bash
   podman compose build
   podman compose up -d
   ```

3. **Zero-downtime deployment:**
   ```bash
   # Build new image
   podman compose build api
   
   # Rolling update (requires orchestrator)
   podman service update --image ai-chatbot-api:latest api
   ```

### Troubleshooting

#### API not responding
```bash
# Check container status
podman compose ps

# View logs
podman compose logs api

# Restart service
podman compose restart api
```

#### Redis connection issues
```bash
# Check Redis health
podman compose exec redis redis-cli ping

# View Redis logs
podman compose logs redis
```

#### MongoDB connection issues
```bash
# Test Atlas connection from container
podman compose exec api python -c "from pymongo import MongoClient; print(MongoClient('your-atlas-mongodb-url').server_info())"
```

### Cleanup

```bash
# Stop all services
podman compose down

# Remove volumes (WARNING: deletes data)
podman compose down -v

# Remove images
podman rmi ai-chatbot-api:latest
```

### Production Checklist

- [ ] Environment variables configured
- [ ] API key generated and secured
- [ ] MongoDB Atlas configured with IP whitelist
- [ ] SSL/TLS enabled
- [ ] Monitoring and logging set up
- [ ] Backup strategy implemented
- [ ] Rate limiting configured
- [ ] Health checks passing
- [ ] Load testing completed
- [ ] Documentation updated

## Cloud Deployment

### AWS ECS/Fargate

1. Push image to ECR
2. Create ECS task definition
3. Configure ALB with SSL
4. Use AWS Secrets Manager for env vars

### Google Cloud Run

1. Push image to GCR
2. Deploy with `gcloud run deploy`
3. Configure Cloud Load Balancing
4. Use Secret Manager

### Azure Container Instances

1. Push image to ACR
2. Deploy with `az container create`
3. Configure Application Gateway
4. Use Key Vault

---

**Made with Bob**