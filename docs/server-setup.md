# PrettyWords Server Setup

Oracle Cloud A1 Flex 같은 ARM Linux 서버에서 PrettyWords를 실행하는 방법입니다.

## 1. 서버 권장 사양

- 권장: Oracle `VM.Standard.A1.Flex` 4 OCPU / 24 GB RAM
- 가능: 2 OCPU / 12 GB RAM, `qwen3:1.7b` 권장
- 비추천: 1 OCPU / 6 GB RAM에서 4B 모델

4B 모델은 CPU 추론이라 빠르지 않습니다. 실서버에서는 `AI_SCAN_ALL=false`를 유지해서 로컬 필터에 걸린 의심 메시지만 AI로 확인하세요.

## 2. 패키지 설치

Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip curl
```

Oracle Linux:

```bash
sudo dnf install -y git python3 python3-pip curl
```

## 3. Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

4B 기본 모델:

```bash
ollama pull qwen3:4b
```

느리면 1.7B도 준비:

```bash
ollama pull qwen3:1.7b
```

확인:

```bash
curl http://127.0.0.1:11434/api/tags
```

`11434` 포트는 외부에 열지 마세요. 봇과 Ollama는 같은 서버의 `127.0.0.1`로 통신하면 됩니다.

## 4. 봇 코드 배치

예시는 `/opt/prettywords`에 설치합니다.

```bash
sudo mkdir -p /opt/prettywords
sudo chown "$USER:$USER" /opt/prettywords
git clone <your-github-repo-url> /opt/prettywords
cd /opt/prettywords

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. 환경 변수 설정

```bash
cp .env.example .env
nano .env
```

필수/권장 예시:

```env
DISCORD_TOKEN=your_discord_bot_token
BOT_ADMIN_IDS=your_discord_user_id

AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_SECONDS=30
AI_SCAN_ALL=false

DATABASE_PATH=data/prettywords.sqlite3
LOG_LEVEL=INFO
ENABLE_MEMBERS_INTENT=false
```

`BOT_ADMIN_IDS`에는 봇 설정을 바꿀 수 있는 Discord 사용자 ID를 넣습니다. 여러 명이면 쉼표나 공백으로 구분합니다.

## 6. Discord Developer Portal 설정

1. Discord Developer Portal에서 Application 생성
2. Bot 생성 후 token을 `.env`의 `DISCORD_TOKEN`에 입력
3. Bot 탭에서 `Message Content Intent` 켜기
4. OAuth2 URL Generator에서 scope:
   - `bot`
   - `applications.commands`
5. Bot permissions:
   - `View Channels`
   - `Send Messages`
   - `Manage Messages`
   - `Moderate Members`
   - `Use Slash Commands`
6. 봇 역할을 제재 대상 역할보다 위로 이동

## 7. 수동 실행 테스트

```bash
cd /opt/prettywords
source .venv/bin/activate
python main.py
```

Discord 서버에서:

```text
/filter status
/filter log-channel channel:#moderation-log
/filter health
/filter timeout minutes:10
/pw config-admin-add user_id:123456789012345678
```

`/filter log-channel`을 설정하면 10분마다 health summary가 로그 채널에 전송됩니다. 너무 많으면 `/filter health-log enabled:false`로 끌 수 있습니다.

문제 없으면 `Ctrl+C`로 중지하고 systemd 서비스로 등록합니다.

## 8. systemd 서비스 등록

```bash
sudo nano /etc/systemd/system/prettywords.service
```

내용:

```ini
[Unit]
Description=PrettyWords Discord moderation bot
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/prettywords
EnvironmentFile=/opt/prettywords/.env
ExecStart=/opt/prettywords/.venv/bin/python /opt/prettywords/main.py
Restart=always
RestartSec=10
User=opc
Group=opc

[Install]
WantedBy=multi-user.target
```

Ubuntu 기본 사용자가 `ubuntu`라면 `User=ubuntu`, `Group=ubuntu`로 바꾸세요.

서비스 시작:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now prettywords
sudo systemctl status prettywords
```

로그 확인:

```bash
journalctl -u prettywords -f
```

재시작:

```bash
sudo systemctl restart prettywords
```

## 9. 모델 전환

서버가 느리면:

```text
/filter ai provider:ollama model:qwen3:1.7b scan_all:false
```

다시 4B:

```text
/filter ai provider:ollama model:qwen3:4b scan_all:false
```

`.env` 기본값으로 복귀:

```text
/filter ai-reset
```

## 10. 업데이트

```bash
cd /opt/prettywords
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart prettywords
```
