#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3.9 python3.9-venv python3-pip

# Install system dependencies
sudo apt-get install -y ffmpeg
sudo apt-get install -y build-essential
sudo apt-get install -y libsndfile1

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p outputs
mkdir -p processed

# Download model checkpoints if not present
if [ ! -d "checkpoints" ]; then
    echo "Downloading model checkpoints..."
    # Add your model download commands here
    # Example:
    # wget https://your-model-url/checkpoints.zip
    # unzip checkpoints.zip
fi

# Create systemd service file
sudo tee /etc/systemd/system/openvoice.service << EOF
[Unit]
Description=OpenVoice API Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python voice_clone_api.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable openvoice
sudo systemctl start openvoice

# Configure firewall
sudo ufw allow 8000/tcp

echo "Deployment completed! The API should be running on port 8000"
echo "You can check the status with: sudo systemctl status openvoice"
echo "View logs with: sudo journalctl -u openvoice -f" 