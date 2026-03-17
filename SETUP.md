# Setup Guide - Quick Start

## 🚀 Installation Steps

### Prerequisites
- Python 3.9+
- pip / conda
- 4GB+ RAM
- 2GB disk space

### Windows Setup

```bash
# 1. Navigate to project
cd "C:\Users\niharika\OneDrive\Desktop\projects___\Leaf Health Check"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Initialize database
python database/init_db.py

# 7. Create models (or use pre-trained)
python model/create_models.py

# 8. Run application
streamlit run app.py
```

### macOS/Linux Setup

```bash
# 1. Navigate to project
cd ~/Desktop/Leaf\ Health\ Check

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize database
python3 database/init_db.py

# 6. Create models
python3 model/create_models.py

# 7. Run application
streamlit run app.py
```

---

## 🐳 Docker Setup

### Build & Run

```bash
# Build image
docker build -t leaf-health-check:latest .

# Run container
docker run -p 8501:8501 leaf-health-check:latest
```

### Using Docker Compose

```bash
docker-compose up --build
```

Access at: `http://localhost:8501`

---

## ✅ Verify Installation

```bash
# Check Python version (should be 3.9+)
python --version

# Check if packages installed
pip list | grep streamlit
pip list | grep tensorflow

# Test imports
python -c "import streamlit; import tensorflow; import cv2; print('✓ All imports successful!')"
```

---

## 🎯 First Run Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Database initialized (plants.db exists)
- [ ] Models created or available
- [ ] No error messages on `streamlit run app.py`
- [ ] Browser opens to http://localhost:8501

---

## 🔧 Common Issues

**Issue:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip install streamlit==1.28.1
```

---

**Issue:** `CUDA/GPU not available`

**Solution:**
```bash
# Edit requirements.txt
# Change: tensorflow==2.13.1
# To: tensorflow-cpu==2.13.1

pip install --force-reinstall tensorflow-cpu==2.13.1
```

---

**Issue:** Port 8501 already in use

**Solution:**
```bash
streamlit run app.py --server.port 8502
```

---

## 📱 Access Application

After running `streamlit run app.py`:

- **Local:** http://localhost:8501
- **Network:** http://<YOUR-IP>:8501 (from other devices)
- **Docker:** http://localhost:8501

---

## 🔒 Production Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repo and deploy

### AWS

```bash
# Create ECR repository
aws ecr create-repository --repository-name leaf-health-check

# Build and push Docker image
docker build -t leaf-health-check:latest .
docker tag leaf-health-check:latest <ECR-URI>/leaf-health-check:latest
docker push <ECR-URI>/leaf-health-check:latest

# Deploy using App Runner or ECS
```

### Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py" > Procfile

# Deploy
heroku login
heroku create leaf-health-check
git push heroku main
```

---

## 📚 Next Steps

1. Read [README.md](README.md) for full documentation
2. Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. Review [TRAINING.md](TRAINING.md) for ML pipeline
4. Upload a leaf image and test the application
5. Explore analysis history and export features

---

**Happy Diagnosing! 🍃**
