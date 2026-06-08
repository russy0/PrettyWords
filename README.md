# PrettyWords

discord.py 기반 AI 비속어 검열 봇입니다. 메시지를 자동 검사하고, 관리자 설정에 따라 삭제, 타임아웃, 로그 기록, 신고 처리, 학습 반영을 수행합니다.

## 핵심 기능

- AI 자동 필터링: `OPENAI_API_KEY`가 있으면 OpenAI 모델로 문맥 기반 판정
- 로컬 적응형 필터: 키가 없어도 기본/등록 금지어와 우회 표기 감지
- 서버별 설정: 로그 채널, 타임아웃 시간, 확신도 기준, dry-run, DM 경고
- 채널별 비활성화: 특정 채널 필터 제외
- 일시정지/다시시작: 서버 전체 필터 pause/resume
- 금지어/허용어 등록: 서버 운영자 기준 반영
- 신고/검토 루프: 오탐 신고, 관리자 처리, AI 학습 예시 반영
- 자동 학습: 고확신 AI 판정, 관리자 확정, 오탐 처리 기록을 다음 판정 컨텍스트에 사용
- 반복 위반 가중: 최근 위반 횟수에 따라 타임아웃 증가
- 예외 대상: 특정 역할/유저 필터 제외

## 설치

서버 배포는 [Server Setup](docs/server-setup.md)을 참고하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`에서 값을 설정하세요.

```env
DISCORD_TOKEN=...
BOT_ADMIN_IDS=
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_SECONDS=12
AI_SCAN_ALL=false
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-nano
DISCORD_TEST_GUILD_ID=...
```

### Oracle A1 Flex + Ollama

A1 Flex is ARM, so use small local models first. Recommended start:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
ollama serve
```

Then set:

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
AI_SCAN_ALL=false
```

If the instance is too slow, pull a smaller model and switch without restarting:

```bash
ollama pull qwen3:1.7b
```

```text
/filter ai provider:ollama model:qwen3:1.7b scan_all:false
```

Switch back to 4B:

```text
/filter ai provider:ollama model:qwen3:4b scan_all:false
```

Return to `.env` defaults:

```text
/filter ai-reset
```

Keep local keyword filtering enabled; the local LLM should judge ambiguous messages, not carry all moderation alone.

### Config admin IDs

Use `BOT_ADMIN_IDS` for global owners who can always configure the bot:

```env
BOT_ADMIN_IDS=123456789012345678,987654321098765432
```

Per server:

```text
/filter config-admin-add user_id:123456789012345678
/filter config-admin-remove user_id:123456789012345678
/filter config-admin-list
```

If no server config admins are registered, Discord users with admin/manage/moderate permissions can configure the bot. Once at least one server config admin is registered, only the server owner, `BOT_ADMIN_IDS`, and registered config admins can change PrettyWords settings.

Discord Developer Portal에서 봇의 **Message Content Intent**를 켜야 메시지 내용을 검사할 수 있습니다. 멤버 인텐트는 기본으로 꺼져 있으며, 필요하면 `ENABLE_MEMBERS_INTENT=true`로 켭니다.

## 실행

```powershell
python main.py
```

봇 초대 권한:

- `View Channels`
- `Send Messages`
- `Manage Messages`
- `Moderate Members`
- `Use Slash Commands`

봇 역할은 제재 대상 역할보다 위에 있어야 타임아웃이 적용됩니다.

## 주요 명령어

- `/filter log-channel #채널`
- `/filter timeout minutes:10`
- `/filter threshold confidence:0.78`
- `/filter pause`, `/filter resume`
- `/filter disable-channel #채널`, `/filter enable-channel #채널`
- `/filter add-word term severity`, `/filter remove-word term`
- `/filter allow-word term`, `/filter remove-allow term`
- `/filter report reason case_id`
- `/filter resolve-report report_id outcome term`
- `/filter mode dry_run ai_enabled delete_messages dm_users escalate`
- `/filter ai provider model scan_all`, `/filter ai-reset`
- `/filter config-admin-add user_id`, `/filter config-admin-remove user_id`, `/filter config-admin-list`
- `/filter exempt-role-add role`, `/filter exempt-user-add member`
- `/filter status`

## 운영 흐름

1. `/filter log-channel`로 로그 채널을 먼저 지정합니다.
2. `/filter timeout`으로 기본 타임아웃을 정합니다.
3. 특정 채널은 `/filter disable-channel`로 제외합니다.
4. 서버 고유 금지어는 `/filter add-word`로 추가합니다.
5. 오탐 신고가 오면 `/filter resolve-report outcome:false_positive`로 처리합니다.
6. 정상 제재 신고는 `/filter resolve-report outcome:confirmed`로 처리해 학습 예시를 강화합니다.

## 참고한 공식 문서

- discord.py 공식 문서: Gateway Intents와 API Reference
- Discord 공식 Application Commands 문서
- OpenAI Structured Outputs / Chat Completions 문서
