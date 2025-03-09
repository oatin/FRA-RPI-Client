#!/bin/bash

#  ตั้งค่าตัวแปรหลัก
PROJECT_DIR="$HOME/admin/face_recognition_service"
SERVICE_NAME="face-recognition"
PYTHON_EXEC="$PROJECT_DIR/venv/bin/python3"
MAIN_SCRIPT="$PROJECT_DIR/main.py"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

cp -r . "$PROJECT_DIR/"

#  กำหนดสีสำหรับข้อความ
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

#  ตรวจสอบสิทธิ์
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}โปรดใช้ sudo หรือรันในฐานะ root${NC}"
   exit 1
fi

#  สร้างโครงสร้างโปรเจกต์
echo -e "${YELLOW}กำลังสร้างโครงสร้างโปรเจกต์...${NC}"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/models"
touch "$PROJECT_DIR/logs/app.log"
chmod 755 "$PROJECT_DIR/logs/app.log"

#  อัปเดตแพ็กเกจและติดตั้ง dependencies
echo -e "${YELLOW}กำลังติดตั้งแพ็กเกจที่จำเป็น...${NC}"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip libopencv-dev

#  สร้าง Virtual Environment และติดตั้งไลบรารี
echo -e "${YELLOW}กำลังสร้าง Virtual Environment...${NC}"
python3 -m venv "$PROJECT_DIR/venv"
source "$PROJECT_DIR/venv/bin/activate"

echo -e "${YELLOW}กำลังติดตั้ง Python packages...${NC}"
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo -e "${RED}ไม่พบไฟล์ requirements.txt! กรุณาตรวจสอบ${NC}"
    exit 1
fi

#  สร้างไฟล์ .env (ถ้ายังไม่มี)
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}กำลังสร้างไฟล์ .env...${NC}"
fi

#  สร้าง Systemd Service
echo -e "${YELLOW}กำลังสร้าง Systemd Service...${NC}"
sudo bash -c "cat > $SERVICE_FILE <<EOL
[Unit]
Description=Face Recognition Service
After=network.target

[Service]
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_EXEC $MAIN_SCRIPT --device_id=\$(cat $PROJECT_DIR/.env | grep DEVICE_ID | cut -d '=' -f2)
Restart=always
User=admin
Environment='PYTHONUNBUFFERED=1'

[Install]
WantedBy=multi-user.target
EOL"

#  รีโหลด Systemd และเปิดใช้งาน Service
echo -e "${YELLOW}กำลังเปิดใช้งาน Service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"

#  แจ้งเตือนสำเร็จ
echo -e "${GREEN}Setup สำเร็จ! Face Recognition Service พร้อมใช้งานแล้ว 🚀${NC}"
echo -e "${YELLOW}ใช้คำสั่งเพื่อตรวจสอบสถานะ:${NC}"
echo -e "${GREEN}sudo systemctl status $SERVICE_NAME${NC}"