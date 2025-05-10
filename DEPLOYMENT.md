# Deploying OpenVoice API on AWS Ubuntu

This guide will help you deploy the OpenVoice API on an AWS Ubuntu instance.

## Prerequisites

1. An AWS account
2. Basic knowledge of AWS EC2
3. SSH access to your instance

## Step 1: Launch an EC2 Instance

1. Go to AWS Console > EC2
2. Click "Launch Instance"
3. Choose Ubuntu Server 22.04 LTS
4. Select an instance type (recommended: t2.large or better for GPU support)
5. Configure instance details:
   - Enable "Auto-assign Public IP"
   - Add storage (at least 20GB)
6. Add tags (optional)
7. Configure security group:
   - Allow SSH (port 22)
   - Allow HTTP (port 80)
   - Allow custom TCP (port 8000)
8. Review and launch
9. Create or select an existing key pair

## Step 2: Connect to Your Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

## Step 3: Clone the Repository

```bash
git clone https://github.com/your-username/OpenVoice.git
cd OpenVoice
```

## Step 4: Run the Deployment Script

```bash
chmod +x deploy.sh
./deploy.sh
```

## Step 5: Verify the Installation

1. Check service status:
```bash
sudo systemctl status openvoice
```

2. View logs:
```bash
sudo journalctl -u openvoice -f
```

3. Test the API:
```bash
curl http://localhost:8000
```

## Step 6: Set Up Domain and SSL (Optional)

1. Point your domain to the EC2 instance IP
2. Install Certbot:
```bash
sudo apt-get install certbot python3-certbot-nginx
```

3. Install and configure Nginx:
```bash
sudo apt-get install nginx
sudo nano /etc/nginx/sites-available/openvoice
```

Add this configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

4. Enable the site and get SSL certificate:
```bash
sudo ln -s /etc/nginx/sites-available/openvoice /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo certbot --nginx -d your-domain.com
```

## Monitoring and Maintenance

1. Monitor logs:
```bash
sudo journalctl -u openvoice -f
```

2. Restart service:
```bash
sudo systemctl restart openvoice
```

3. Update the application:
```bash
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart openvoice
```

## Troubleshooting

1. If the service fails to start:
   - Check logs: `sudo journalctl -u openvoice -f`
   - Verify Python environment: `source venv/bin/activate && python --version`
   - Check model files: `ls checkpoints/`

2. If the API is not accessible:
   - Check firewall: `sudo ufw status`
   - Verify port is open: `netstat -tulpn | grep 8000`
   - Check security group settings in AWS Console

3. If you get memory errors:
   - Increase swap space
   - Use a larger instance type
   - Optimize model loading

## Security Considerations

1. Keep your system updated:
```bash
sudo apt-get update && sudo apt-get upgrade
```

2. Use a firewall:
```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8000/tcp
```

3. Consider adding API authentication
4. Regularly backup your model checkpoints
5. Monitor system resources

## Backup and Recovery

1. Backup model checkpoints:
```bash
tar -czf checkpoints_backup.tar.gz checkpoints/
```

2. Backup configuration:
```bash
cp voice_clone_api.py voice_clone_api.py.backup
```

3. Restore from backup:
```bash
tar -xzf checkpoints_backup.tar.gz
```

## Scaling Considerations

1. Use a load balancer for multiple instances
2. Consider using AWS ECS or Kubernetes for containerization
3. Use AWS CloudWatch for monitoring
4. Implement caching for frequently used voices
5. Consider using AWS S3 for storing audio files

## Support

For issues and support:
1. Check the logs: `sudo journalctl -u openvoice -f`
2. Review AWS CloudWatch metrics
3. Check system resources: `htop`
4. Contact the development team 