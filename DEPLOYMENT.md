# 🚀 Deployment Guide — Enterprise Fraud Detection

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Streamlit Cloud](#streamlit-cloud)
4. [Production Deployment](#production-deployment)
5. [Environment Variables](#environment-variables)
6. [Security Checklist](#security-checklist)

---

## Local Development

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/SumedhPatil1507/fraud_detection_project.git
cd fraud_detection_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Save output to ENCRYPTION_KEY env var

# Set environment variables
export ENCRYPTION_KEY="your-generated-key"
export API_KEY_ADMIN="admin-dev-key"
export API_KEY_ANALYST="analyst-dev-key"
export API_KEY_VIEWER="viewer-dev-key"

# Run Streamlit app
streamlit run app.py

# Run FastAPI (separate terminal)
uvicorn api:app --reload --port 8000
```

Access:
- **Streamlit UI:** http://localhost:8501
- **FastAPI Docs:** http://localhost:8000/docs

---

## Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Custom Configuration

Edit `docker-compose.yml` to set environment variables:

```yaml
environment:
  - ENCRYPTION_KEY=your-key
  - API_KEY_ADMIN=your-admin-key
  - SUPABASE_URL=your-supabase-url
  - SUPABASE_KEY=your-supabase-key
  - GROQ_API_KEY=your-groq-key
```

---

## Streamlit Cloud

### Setup

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your fork
4. Add secrets in Settings → Secrets

### Secrets Configuration

Create `.streamlit/secrets.toml`:

```toml
APP_PASSWORD = "your_secure_password"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-anon-key"
GROQ_API_KEY = "gsk_xxxx"
ENCRYPTION_KEY = "your-fernet-key"
```

### Custom Domain

1. Go to App Settings → General
2. Add custom domain
3. Update DNS CNAME record

---

## Production Deployment

### AWS EC2 / DigitalOcean

```bash
# SSH into server
ssh user@your-server-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone repo
git clone https://github.com/SumedhPatil1507/fraud_detection_project.git
cd fraud_detection_project

# Set production env vars
nano .env  # Add all required variables

# Run with docker-compose
docker-compose up -d

# Setup nginx reverse proxy
sudo apt install nginx
sudo nano /etc/nginx/sites-available/fraud-detection
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/fraud-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Kubernetes (Advanced)

```bash
# Create deployment
kubectl apply -f k8s/deployment.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Setup ingress
kubectl apply -f k8s/ingress.yaml
```

---

## Environment Variables

### Required

```bash
# Encryption
ENCRYPTION_KEY="fernet-key-here"

# RBAC
API_KEY_ADMIN="strong-admin-key"
API_KEY_ANALYST="strong-analyst-key"
API_KEY_VIEWER="strong-viewer-key"
```

### Optional

```bash
# Database
SUPABASE_URL="https://xxxx.supabase.co"
SUPABASE_KEY="your-anon-key"

# LLM
GROQ_API_KEY="gsk_xxxx"

# Rate Limiting
RATE_LIMIT_ADMIN="1000/minute"
RATE_LIMIT_ANALYST="200/minute"
RATE_LIMIT_VIEWER="30/minute"

# Webhooks
RETRAIN_WEBHOOK_URL="https://hooks.slack.com/services/xxx"

# Institution
INSTITUTION_NAME="Your Bank Name"
```

---

## Security Checklist

### Before Production

- [ ] Change all default API keys
- [ ] Generate strong encryption key
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable audit logging
- [ ] Configure Supabase RLS policies
- [ ] Review RBAC permissions
- [ ] Set strong APP_PASSWORD
- [ ] Disable debug mode
- [ ] Configure CORS properly
- [ ] Set up monitoring/alerts
- [ ] Backup database regularly
- [ ] Test disaster recovery

### Monitoring

```bash
# Check API health
curl https://your-domain.com/api/

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Monitor resources
docker stats
```

### Backup

```bash
# Backup model artifacts
tar -czf model-backup-$(date +%Y%m%d).tar.gz outputs/

# Backup database (if using local CSV)
tar -czf data-backup-$(date +%Y%m%d).tar.gz outputs/*.csv

# Backup to S3
aws s3 cp model-backup-*.tar.gz s3://your-bucket/backups/
```

---

## Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Train model first
python main.py
```

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**Permission denied:**
```bash
# Fix file permissions
chmod +x scripts/*.sh
```

**Docker build fails:**
```bash
# Clear cache and rebuild
docker-compose build --no-cache
```

---

## Performance Tuning

### Streamlit

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### FastAPI

```python
# Increase workers
uvicorn api:app --workers 4 --host 0.0.0.0 --port 8000
```

### Database

```sql
-- Add indexes to Supabase
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp DESC);
CREATE INDEX idx_predictions_is_fraud ON predictions(is_fraud);
```

---

## Support

For deployment issues:
1. Check logs: `docker-compose logs`
2. Review [ARCHITECTURE.md](ARCHITECTURE.md)
3. Open GitHub issue
4. Contact: [@SumedhPatil1507](https://github.com/SumedhPatil1507)
