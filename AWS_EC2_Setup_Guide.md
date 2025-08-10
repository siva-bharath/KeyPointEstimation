# AWS EC2 Ubuntu Setup Guide for Keypoint Estimation Project

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Launching AWS EC2 Instance](#launching-aws-ec2-instance)
3. [Connecting to EC2 Instance](#connecting-to-ec2-instance)
4. [System Updates and Dependencies](#system-updates-and-dependencies)
5. [Python and Virtual Environment Setup](#python-and-virtual-environment-setup)
6. [Cloning the Project](#cloning-the-project)
7. [Installing Dependencies](#installing-dependencies)
8. [Running the Project](#running-the-project)
9. [Troubleshooting](#troubleshooting)
10. [Useful Commands](#useful-commands)

---

## Prerequisites

- AWS Account with EC2 access
- SSH key pair (create if you don't have one)
- Basic knowledge of Linux commands
- Git installed on your local machine

---

## Launching AWS EC2 Instance

### Step 1: Access AWS Console
1. Log into [AWS Console](https://aws.amazon.com/)
2. Navigate to EC2 service
3. Click "Launch Instance"

### Step 2: Choose AMI
1. **Name**: `Keypoint-Estimation-Project`
2. **AMI**: Select "Ubuntu Server 22.04 LTS (HVM), SSD Volume Type"
3. **Architecture**: x86 (64-bit)

### Step 3: Instance Type
- **Type**: t2.micro (Free tier eligible)
- **vCPUs**: 1
- **Memory**: 1 GiB
- **Network Performance**: Low to Moderate

### Step 4: Key Pair
1. **Key pair name**: Create new key pair or select existing
2. **Key pair type**: RSA
3. **Private key file format**: .pem
4. **Download the .pem file** and keep it secure

### Step 5: Network Settings
1. **VPC**: Default VPC
2. **Subnet**: Default subnet
3. **Auto-assign public IP**: Enable
4. **Security Group**: Create new security group
   - **Name**: `Keypoint-Project-SG`
   - **Description**: Security group for keypoint estimation project
   - **Inbound rules**:
     - SSH (Port 22): 0.0.0.0/0 (or your IP)
     - HTTP (Port 80): 0.0.0.0/0 (optional)
     - HTTPS (Port 443): 0.0.0.0/0 (optional)

### Step 6: Storage
1. **Volume type**: General Purpose SSD (gp3)
2. **Size**: 20 GiB (minimum)
3. **Delete on termination**: Yes (for cost control)

### Step 7: Launch
1. Review your configuration
2. Click "Launch Instance"
3. Wait for instance to reach "Running" status

---

## Connecting to EC2 Instance

### Step 1: Get Public IP
1. In EC2 console, note the **Public IPv4 address**
2. Example: `54.123.45.67`

### Step 2: Set Key Permissions (Linux/Mac)
```bash
chmod 400 your-key-pair.pem
```

### Step 3: Connect via SSH
```bash
ssh -i your-key-pair.pem ubuntu@YOUR_PUBLIC_IP
```

### Step 4: Windows Users (PowerShell)
```powershell
ssh -i "C:\path\to\your-key-pair.pem" ubuntu@YOUR_PUBLIC_IP
```

---

## System Updates and Dependencies

### Step 1: Update System
```bash
# Update package list
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install essential tools
sudo apt install -y build-essential git curl wget unzip
```

### Step 2: Install Python Dependencies
```bash
# Install Python 3.10+ and pip
sudo apt install -y python3 python3-pip

# Install development tools
sudo apt install -y python3-dev

# Verify installation
python3 --version
pip3 --version

# Try to install python3-venv (may not be available in all Ubuntu versions)
sudo apt install -y python3-venv || echo "python3-venv not available, will use alternative methods"

# Alternative: Install virtualenv if venv is not available
sudo apt install -y python3-virtualenv || pip3 install virtualenv
```

### Step 3: Install Conda (Alternative to venv)
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make installer executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer (accept defaults or customize as needed)
./Miniconda3-latest-Linux-x86_64.sh

# Reload shell configuration
source ~/.bashrc

# Verify conda installation
conda --version

# Update conda
conda update conda -y

# Create conda environment (alternative to venv)
conda create -n keypoint-env python=3.10 -y

# Activate conda environment
conda activate keypoint-env

# Install PyTorch via conda (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Step 4: Install System Libraries
```bash
# Install libraries required for OpenCV and PyTorch
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install additional dependencies
sudo apt install -y libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtcore4 libqtgui4 libqt4-test
```

---

## Python and Virtual Environment Setup

### Environment Management Options

You have two main options for managing Python environments:

1. **Python venv (Recommended for t2.micro)**: Lighter, uses less disk space
2. **Conda**: More features, better package management, but uses more disk space

**Note**: For t2.micro instances (1GB RAM), venv is recommended due to space constraints.

### Option 1: Using Python venv (Recommended)
```bash
# Create project directory
mkdir ~/keypoint-project
cd ~/keypoint-project

# Try to create virtual environment with venv
if python3 -m venv venv 2>/dev/null; then
    echo "Created virtual environment with venv"
else
    echo "venv not available, trying virtualenv instead"
    # Fallback to virtualenv
    python3 -m virtualenv venv
fi

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Verify Virtual Environment
```bash
# Check Python location (should point to venv)
which python
which pip

# Check Python version
python --version

# Upgrade pip
pip install --upgrade pip
```

### Option 2: Using Conda
```bash
# Navigate to project directory
cd ~/keypoint-project

# Create conda environment
conda create -n keypoint-env python=3.10 -y

# Activate conda environment
conda activate keypoint-env

# Verify conda environment
which python
which pip
python --version

# Install PyTorch via conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

---

## Cloning the Project

### Step 1: Clone Repository
```bash
# Make sure you're in the project directory
cd ~/keypoint-project

# Clone your repository
git clone https://github.com/siva-bharath/KeyPointEstimation.git

# Navigate to project directory
cd KeyPointEstimation

# List contents
ls -la
```

**Important**: Make sure your virtual environment is activated before proceeding:
- **For venv**: `source ~/keypoint-project/venv/bin/activate`
- **For conda**: `conda activate keypoint-env`

### Step 2: Verify Project Structure
```bash
# Check project structure
tree -L 2

# Expected structure:
# KeyPointEstimation/
# ├── dataset/
# ├── model/
# ├── train/
# ├── utils/
# ├── setup/
# ├── main.py
# ├── inference.py
# ├── requirements.txt
# └── .gitignore
```

---

## Installing Dependencies

### Step 1: Install PyTorch (CPU Version for t2.micro)
```bash
# Make sure virtual environment is activated
source ~/keypoint-project/venv/bin/activate

# Install PyTorch CPU version (lighter than GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install Other Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# If you encounter memory issues, install one by one:
pip install numpy matplotlib Pillow opencv-python
pip install mlflow tensorboard
pip install onnx onnxruntime
pip install pycocotools tqdm scikit-learn pandas
```

### Step 3: Verify Installation
```bash
# Test imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

---

## Running the Project

### Step 1: Download Dataset
```bash
# Navigate to project directory
cd ~/keypoint-project/KeyPointEstimation

# Run main.py (this will download COCO dataset)
python main.py
```

### Step 2: Monitor Training
```bash
# In a new terminal session, connect to EC2 and monitor:
ssh -i your-key-pair.pem ubuntu@YOUR_PUBLIC_IP

# Navigate to project
cd ~/keypoint-project/KeyPointEstimation

# Check training progress
tail -f nohup.out  # if using nohup
# OR
ps aux | grep python  # check if process is running
```

### Step 3: Access MLflow UI
```bash
# Start MLflow UI (in a separate terminal)
mlflow ui --host 0.0.0.0 --port 5000

# Access from your browser: http://YOUR_PUBLIC_IP:5000
```

---

## Troubleshooting

### Virtual Environment Issues

#### Problem: `python3-venv` not available
```bash
# Solution 1: Update package list and try again
sudo apt update
sudo apt install -y python3-venv

# Solution 2: Use virtualenv instead
sudo apt install -y python3-virtualenv
python3 -m virtualenv venv

# Solution 3: Use conda (most reliable)
conda create -n keypoint-env python=3.10 -y
conda activate keypoint-env
```

#### Problem: Permission denied when creating venv
```bash
# Check current directory permissions
ls -la

# Fix permissions if needed
chmod 755 ~/keypoint-project

# Try creating venv in home directory
cd ~
python3 -m venv keypoint-venv
```

### Common Issues and Solutions

#### Issue 1: Out of Memory
```bash
# Check memory usage
free -h

# Solution: Use smaller batch size in config.py
# Change batch_size from 32 to 8 or 16
```

#### Issue 2: Import Errors
```bash
# Ensure virtual environment is activated
source ~/keypoint-project/venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Issue 5: Virtual Environment Creation Failed
```bash
# Check if venv is available
python3 -m venv --help

# If venv fails, use virtualenv instead
sudo apt install -y python3-virtualenv
python3 -m virtualenv venv

# Alternative: Use conda if both fail
conda create -n keypoint-env python=3.10 -y
conda activate keypoint-env
```

#### Issue 3: Permission Denied
```bash
# Fix file permissions
chmod +x main.py
chmod +x inference.py

# Fix directory permissions
chmod 755 ~/keypoint-project/KeyPointEstimation
```

#### Issue 4: Network Issues
```bash
# Check if instance can access internet
ping google.com

# Check security group rules
# Ensure outbound rules allow all traffic
```

---

## Useful Commands

### System Monitoring
```bash
# Check system resources
htop
df -h
free -h
nvidia-smi  # if using GPU instance

# Check running processes
ps aux | grep python
ps aux | grep mlflow
```

### Virtual Environment Management
```bash
# Activate virtual environment
source ~/keypoint-project/venv/bin/activate

# Deactivate virtual environment
deactivate

# List installed packages
pip list

# Export requirements
pip freeze > requirements_current.txt
```

### Conda Environment Management
```bash
# Activate conda environment
conda activate keypoint-env

# Deactivate conda environment
conda deactivate

# List conda environments
conda env list

# List installed packages
conda list

# Export conda environment
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml

# Update conda
conda update conda -y
```

### Git Operations
```bash
# Check git status
git status

# Pull latest changes
git pull origin main

# Check commit history
git log --oneline
```

### File Management
```bash
# View log files
tail -f training.log

# Check disk usage
du -sh *

# Find large files
find . -size +100M
```

---

## Cost Optimization

### Instance Management
1. **Stop instance** when not in use (saves money)
2. **Use Spot Instances** for cost-effective training
3. **Monitor usage** with AWS Cost Explorer
4. **Set up billing alerts**

### Storage Optimization
1. **Use S3** for large datasets
2. **Clean up checkpoints** regularly
3. **Use lifecycle policies** for automatic cleanup

---

## Security Best Practices

1. **Never commit** API keys or credentials
2. **Use IAM roles** instead of access keys
3. **Restrict security group** to your IP only
4. **Regularly update** system packages
5. **Monitor CloudTrail** for suspicious activity

---

## Next Steps

1. **Train your model** on the full dataset
2. **Experiment with hyperparameters** using MLflow
3. **Deploy model** using ONNX format
4. **Set up CI/CD** pipeline
5. **Monitor performance** in production

---

## Support and Resources

- **AWS Documentation**: [EC2 User Guide](https://docs.aws.amazon.com/ec2/)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)
- **MLflow Documentation**: [mlflow.org](https://mlflow.org/docs/)
- **GitHub Repository**: [Your Project](https://github.com/siva-bharath/KeyPointEstimation)

---

## Quick Reference Commands

```bash
# Connect to EC2
ssh -i key.pem ubuntu@IP_ADDRESS

# Activate environment
source ~/keypoint-project/venv/bin/activate

# Run training
cd ~/keypoint-project/KeyPointEstimation
python main.py

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Check status
git status
ps aux | grep python
```

---

*This guide was created for the Keypoint Estimation Project using PyTorch and MLflow on AWS EC2 Ubuntu AMI.*
