# 🤖 Real-Time Sinhala AI Voice Assistant on Edge

<div align="center">

![Raspberry Pi AI Project](https://img.shields.io/badge/Platform-Raspberry_Pi_4-c51a4a?style=for-the-badge&logo=raspberry-pi)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini_2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A low-latency, real-time voice assistant that speaks fluent Sinhala, powered by Google's Gemini Live API and optimized for edge computing on Raspberry Pi.**

[Features](#-features) • [Getting Started](#-getting-started) • [Installation](#-installation) • [Usage](#-usage) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📖 Overview

SinhalaBot is a real-time conversational AI assistant designed specifically for the Sinhala language. Unlike traditional voice assistants that process speech in chunks, SinhalaBot uses streaming audio technology to create natural, radio-announcer style conversations with near-zero latency.

### Why This Project?

- **Real-Time Streaming**: Audio is processed and responded to in real-time, creating natural conversational flow
- **Edge Computing**: Runs entirely on a Raspberry Pi 4, perfect for robotics and embedded systems
- **Sinhala Language Support**: Native support for Sri Lanka's primary language
- **Low Latency**: Optimized for instant responses, ideal for interactive applications
- **Headless Operation**: Can run on boot without manual intervention, perfect for robot deployments

---

## ✨ Features

- 🎤 **Real-time audio streaming** with Google Gemini Live API
- 🗣️ **Native Sinhala language support** for both input and output
- ⚡ **Ultra-low latency** response times (< 500ms)
- 🔊 **High-quality audio output** optimized for Raspberry Pi hardware
- 🚀 **Headless mode support** with systemd service integration
- 🛠️ **Easy configuration** with environment variables
- 📊 **Robust error handling** and audio buffer management

---

## 🎯 Use Cases

- **Educational Robots**: Interactive learning companions for Sri Lankan schools
- **Smart Home Assistants**: Voice-controlled home automation in Sinhala
- **Customer Service Kiosks**: Automated information booths
- **Language Learning Tools**: Practice conversational Sinhala
- **Accessibility Devices**: Voice interfaces for visually impaired users

---

## 🔧 Hardware Requirements

### Essential Components

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Raspberry Pi** | Pi 4 Model B (4GB+ RAM) | 8GB recommended for best performance |
| **Microphone** | USB Microphone | USB avoids Linux driver compatibility issues |
| **Speaker** | 3.5mm AUX or USB | Any amplified speaker will work |
| **Storage** | MicroSD Card (16GB+) | Class 10 or better for faster I/O |
| **Power Supply** | Official Pi 4 PSU (5V/3A) | Stable power prevents audio glitches |
| **Cooling** | Heatsink or small fan | Recommended for sustained operation |

### Optional Components

- **Camera Module**: For future vision integration
- **GPIO Sensors**: Temperature, motion, etc.
- **LED Indicators**: Visual feedback for robot status
- **Battery Pack**: For mobile robot applications

### Wiring Diagram

```
┌─────────────────────────────────┐
│      Raspberry Pi 4             │
│                                 │
│  [USB] ←── USB Microphone       │
│  [3.5mm] ←── Speakers/Headphones│
│  [Ethernet/WiFi] ←── Network    │
│  [USB-C] ←── Power Supply       │
└─────────────────────────────────┘
```


<img src="./setup.jpeg" alt="Reggie Wiring Setup" width="400">

---

##  Getting Started

### Prerequisites

- Raspberry Pi 4 with Raspberry Pi OS (Bookworm or later)
- Active internet connection (WiFi or Ethernet)
- Google AI Studio API key
- Basic familiarity with Linux terminal

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/sinhala-ai-voice-assistant.git
cd sinhala-ai-voice-assistant

# Run the setup script
chmod +x setup.sh
./setup.sh

# Start the assistant
python main.py
```

---

##  Installation

### Step 1: Obtain API Credentials

1. Navigate to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** in the top navigation
4. Create a new API key for your project
5. **Copy and securely store the API key** (you'll need it in Step 4)

> ⚠️ **Security Note**: Never commit your API key to version control. Use environment variables or `.env` files (added to `.gitignore`).

### Step 2: System Setup

Update your Raspberry Pi and install required audio drivers:

```bash
# Update package lists and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Install audio libraries and dependencies
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libatlas-base-dev \
    libasound2-dev \
    git
```

### Step 3: Configure Audio Output

Set the default audio output to ensure sound plays through the correct device:

```bash
# Launch Raspberry Pi configuration tool
sudo raspi-config
```

Navigate through the menu:
1. Select `System Options` → `Audio`
2. Choose your preferred output:
   - **Headphones** (3.5mm jack)
   - **HDMI** (if using monitor speakers)
   - **USB** (for USB speakers)
3. Press `Finish` and reboot if prompted

**Verify Audio Configuration:**

```bash
# List available audio devices
aplay -l

# Test speaker output
speaker-test -t wav -c 2
```

### Step 4: Python Environment Setup

Create an isolated Python environment to avoid dependency conflicts:

```bash
# Create project directory
mkdir -p ~/sinhala_bot
cd ~/sinhala_bot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install required packages
pip install google-genai pyaudio numpy python-dotenv
```

### Step 5: Project Configuration

**Create the main application file:**

```bash
nano main.py
```

Copy the application code into `main.py` (code should be provided separately).

**Create environment configuration:**

```bash
nano .env
```

Add your API key:

```env
GEMINI_API_KEY=your_api_key_here
HARDWARE_IN_RATE=48000
CHUNK_SIZE=1024
```

**Create `.gitignore`:**

```bash
nano .gitignore
```

```gitignore
# Environment variables
.env

# Python
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/

# IDE
.vscode/
.idea/

# Logs
*.log
```

---

##  Usage

### Manual Operation

Activate the virtual environment and run the assistant:

```bash
cd ~/sinhala_bot
source venv/bin/activate
python main.py
```

**Expected Output:**

```
[INFO] Initializing audio streams...
[INFO] Microphone: USB Audio Device
[INFO] Speaker: bcm2835 Headphones
[INFO] Connecting to Gemini Live API...
[SUCCESS] ✓ SinhalaBot is LIVE!
[INFO] Speak into the microphone...
```

**Test Phrases:**

- "කොහොමද?" (How are you?)
- "ඔයාගේ නම මොකද්ද?" (What is your name?)
- "මට උදව්වක් කරන්න පුළුවන්ද?" (Can you help me?)

### Stopping the Assistant

- Press `Ctrl + C` in the terminal
- The assistant will gracefully shut down audio streams

---

## 🔄 Headless Mode (Auto-Start on Boot)

For robot deployments, configure SinhalaBot to start automatically when the Raspberry Pi powers on.

### Create Systemd Service

```bash
sudo nano /etc/systemd/system/sinhalabot.service
```

**Service Configuration:**

```ini
[Unit]
Description=SinhalaBot AI Voice Assistant
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/sinhala_bot
Environment="PATH=/home/pi/sinhala_bot/venv/bin"
ExecStart=/home/pi/sinhala_bot/venv/bin/python /home/pi/sinhala_bot/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Enable and Start Service

```bash
# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable sinhalabot.service

# Start service immediately
sudo systemctl start sinhalabot.service

# Check service status
sudo systemctl status sinhalabot.service
```

### Service Management Commands

```bash
# View live logs
sudo journalctl -u sinhalabot.service -f

# Restart service
sudo systemctl restart sinhalabot.service

# Stop service
sudo systemctl stop sinhalabot.service

# Disable auto-start
sudo systemctl disable sinhalabot.service
```

---

##  Troubleshooting

### Audio Issues

#### No Sound Output

**Symptoms**: Assistant responds but no audio plays

**Solutions**:

```bash
# Check audio mixer settings
alsamixer

# Press F6 to select sound card
# Ensure volume is up (use arrow keys)
# Unmute channels (press M)
# PCM and Master should show "00" not "MM"

# Test audio output
speaker-test -t wav -c 2 -l 1

# Check default audio device
cat /proc/asound/cards
```

#### Microphone Not Detected

**Symptoms**: `Error: No input device found`

**Solutions**:

```bash
# List input devices
arecord -l

# Test microphone
arecord -d 5 -f cd test.wav
aplay test.wav

# Add user to audio group
sudo usermod -aG audio $USER
# Reboot required for changes to take effect
sudo reboot
```

#### Audio Stuttering/Breaking Up

**Symptoms**: Choppy or distorted audio playback

**Solutions**:

1. **Increase buffer size** in `main.py`:
   ```python
   CHUNK = 2048  # Increased from 1024
   ```

2. **Use faster SD card** (Class 10 or UHS-I)

3. **Reduce CPU load**:
   ```bash
   # Disable unnecessary services
   sudo systemctl disable bluetooth
   sudo systemctl disable hciuart
   ```

4. **Check CPU throttling**:
   ```bash
   vcgencmd measure_temp
   vcgencmd get_throttled
   # If output shows throttling, improve cooling
   ```

### Sample Rate Issues

**Symptoms**: `[Errno -9997] Invalid sample rate`

**Cause**: Microphone doesn't support 48kHz

**Solution**: Edit `main.py` or `.env`:

```python
HARDWARE_IN_RATE = 44100  # Changed from 48000
```

### API Connection Issues

#### Authentication Errors

**Symptoms**: `401 Unauthorized` or `Invalid API key`

**Solutions**:

1. Verify API key in `.env` file
2. Check for extra spaces or quotes around the key
3. Regenerate key in Google AI Studio
4. Ensure billing is enabled on Google Cloud account

#### Network Timeouts

**Symptoms**: `Connection timeout` or `Network unreachable`

**Solutions**:

```bash
# Test internet connectivity
ping -c 4 google.com

# Check DNS resolution
nslookup generativelanguage.googleapis.com

# Verify firewall rules
sudo iptables -L

# Test API endpoint
curl -H "Content-Type: application/json" \
     -H "x-goog-api-key: YOUR_API_KEY" \
     https://generativelanguage.googleapis.com/v1beta/models
```

### Performance Issues

#### High CPU Usage

**Solutions**:

1. **Monitor resources**:
   ```bash
   htop
   # Look for CPU % and memory usage
   ```

2. **Optimize Python**:
   ```bash
   # Install NumPy with OpenBLAS
   pip install numpy --force-reinstall --no-binary numpy
   ```

3. **Enable hardware acceleration** (if available)

#### Memory Leaks

**Symptoms**: Performance degrades over time

**Solutions**:

1. Add automatic restart to service:
   ```ini
   [Service]
   RuntimeMaxSec=86400  # Restart after 24 hours
   ```

2. Monitor memory:
   ```bash
   watch -n 5 free -h
   ```

### Permission Issues

**Symptoms**: `Permission denied` errors

**Solutions**:

```bash
# Fix file permissions
chmod +x main.py
chmod 644 .env

# Ensure correct ownership
sudo chown -R pi:pi ~/sinhala_bot

# Add user to necessary groups
sudo usermod -aG audio,video,gpio $USER

# Reboot to apply group changes
sudo reboot
```

---

## 📊 Performance Metrics

Typical performance on Raspberry Pi 4 (4GB):

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency** | 300-500ms | From speech end to response start |
| **CPU Usage** | 40-60% | Single core utilization |
| **Memory** | 800MB-1.2GB | Including Python runtime |
| **Network** | 50-100 kbps | Bidirectional audio streaming |
| **Uptime** | 24+ hours | Stable continuous operation |

---

## 🔒 Security Considerations

- **API Key Protection**: Never commit `.env` files to public repositories
- **Network Security**: Use VPN or private network for sensitive deployments
- **Audio Privacy**: Implement local wake word detection to avoid constant listening
- **Update Regularly**: Keep system packages and Python dependencies up to date

```bash
# Regular security updates
sudo apt update && sudo apt upgrade -y
pip install --upgrade google-genai
```


---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


- [ ] **Wake word detection** (offline "Hey Reggie" activation)
- [ ] **Multi-language support** (Tamil, English code-switching)
- [ ] **Voice activity detection** (VAD) for better silence handling
- [ ] **Custom voice profiles** (personalized responses)
- [ ] **Gesture recognition** integration with camera
- [ ] **Home Assistant** integration
- [ ] **Docker** containerization
- [ ] **Web dashboard** for configuration and monitoring


---


### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/sinhala-ai-voice-assistant.git
cd sinhala-ai-voice-assistant

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Our Team:
**Tenura Pinsara** |
**Thanuka Perera** |
**Sehara Arunodya** |
**Dineth Sanjula**

---


## 📚 Additional Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Raspberry Pi Audio Configuration Guide](https://www.raspberrypi.com/documentation/computers/configuration.html#audio)
- [PortAudio Documentation](http://www.portaudio.com/docs.html)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

---

<div align="center">

⭐ Star this repository if you find it helpful!

</div>
