# PrettyWords

discord.py 기반 AI 비속어 검열 봇입니다. 메시지를 자동 검사하고, 관리자 설정에 따라 삭제, 타임아웃, 로그 기록, 신고 처리, 학습 반영을 수행합니다.

## 핵심 기능

- AI 자동 필터링: Groq(무료 호스팅), 로컬 Ollama, OpenAI 중 선택해 문맥 기반 판정 (Groq 사용 시 메시지를 배치로 묶어 전송하고, 키 여러 개 등록 시 자동 순환, 요청 제한 시 로컬 Ollama로 일시 폴백)
- 로컬 적응형 필터: 키가 없어도 기본/등록 금지어와 우회 표기 감지
- 서버별 설정: 로그 채널, 타임아웃 시간, 확신도 기준, 모의 실행, DM 경고
- 채널별 비활성화: 특정 채널 필터 제외
- 일시정지/다시시작: 서버 전체 필터 pause/resume
- 금지어/허용어 등록: 서버 운영자 기준 반영
- 신고/검토 루프: 오탐 신고, 관리자 처리, AI 학습 예시 반영
- 자동 학습: 고확신 AI 판정, 관리자 확정, 오탐 처리 기록을 다음 판정 컨텍스트에 사용
- 반복 위반 가중: 최근 위반 횟수에 따라 타임아웃 증가
- 예외 대상: 특정 역할/유저 필터 제외
- 상태 로그 분리: 제재 로그 채널과 별도 상태 로그 채널 설정 가능
- 카테고리 학습: 욕설, 성적발언, 패드립, 괴롭힘/모욕, 혐오, 위협 등으로 분류
- 메시지 ID 학습: 특정 메시지의 어떤 구간이 어떤 카테고리인지 등록

## 설치

서버 배포는 [서버 세팅 가이드](docs/server-setup.md)를 참고하세요.

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
AI_PROVIDER=groq
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_SECONDS=30
AI_SCAN_ALL=false
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-nano
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BATCH_SIZE=30
GROQ_BATCH_WINDOW_SECONDS=8
GROQ_RATE_LIMIT_COOLDOWN_SECONDS=60
DISCORD_TEST_GUILD_ID=...
```

### Oracle A1 Flex 환경 추천: Groq + Ollama 폴백

A1 Flex는 ARM 아키텍처라 Ollama로 무거운 로컬 모델을 상시 돌리기엔 부담스럽습니다. 그래서 기본값은
무료 호스팅 추론 API인 **Groq**를 1순위로 쓰고, Groq가 요청 제한에 걸렸을 때만 잠깐 로컬
Ollama로 전환하는 방식입니다.

```env
AI_PROVIDER=groq
GROQ_API_KEY=gsk_xxx
GROQ_MODEL=llama-3.3-70b-versatile
AI_SCAN_ALL=false
```

동작 방식:

- AI 판정이 필요한 메시지는 한 건씩 보내지 않고, `GROQ_BATCH_SIZE`개가 모이거나
  `GROQ_BATCH_WINDOW_SECONDS`초가 지나는 것 중 먼저 도달하는 시점에 배치로 묶어 Groq에
  전송합니다 (무료 등급 요청 횟수 제한을 아끼기 위함).
- `GROQ_API_KEY`에 키를 여러 개(콤마/세미콜론/공백 구분) 등록하면, 한 키가 요청 제한에
  걸려도 자동으로 다음 키로 순환하며 재시도합니다.
- 등록된 키 전부가 동시에 요청 제한에 걸렸을 때만 `GROQ_RATE_LIMIT_COOLDOWN_SECONDS` 동안
  로컬 `OLLAMA_MODEL`로 전환했다가 다시 Groq로 복귀합니다.

자세한 옵션 설명은 `.env.example`의 Groq 관련 주석을 참고하세요.

#### 로컬 Ollama만 쓰고 싶을 때

호스트 사양이 충분하다면 Groq 없이 로컬 모델만으로도 운영할 수 있습니다.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
ollama serve
```

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
AI_SCAN_ALL=false
```

인스턴스가 느리면 더 작은 모델을 받아 재시작 없이 바로 전환할 수 있습니다:

```bash
ollama pull qwen3:1.7b
```

```text
/filter ai provider:ollama model:qwen3:1.7b scan_all:false
```

다시 4B로 전환:

```text
/filter ai provider:ollama model:qwen3:4b scan_all:false
```

`.env` 기본값으로 되돌리기:

```text
/filter ai-reset
```

어떤 조합을 쓰든 로컬 키워드 필터는 항상 켜두는 것을 권장합니다. AI(로컬 LLM 또는 Groq)는
애매한 메시지를 판단하는 보조 역할이지, 검열 전체를 혼자 떠맡는 구조가 아닙니다.

### 설정 관리자 ID

항상 봇 설정을 바꿀 수 있는 전역 관리자는 `BOT_ADMIN_IDS`에 등록합니다:

```env
BOT_ADMIN_IDS=123456789012345678,987654321098765432
```

서버별 설정 관리자는 디스코드에서 등록합니다:

```text
/pw config-admin-add user_id:123456789012345678
/pw config-admin-remove user_id:123456789012345678
/pw config-admin-list
```

서버별 설정 관리자가 아직 없으면 관리자/서버 관리/멤버 제재 권한을 가진 Discord 사용자가 봇을 설정할 수 있습니다. 서버별 설정 관리자를 한 명이라도 등록하면 서버 소유자, `BOT_ADMIN_IDS`, 등록된 설정 관리자만 PrettyWords 설정을 바꿀 수 있습니다.

### 상태 로그

상태 로그는 제재/신고 로그와 다른 채널로 분리할 수 있습니다:

```text
/filter log-channel channel:#moderation-log
/filter health-log-channel channel:#bot-health
/filter health
/filter health-log enabled:false
/filter health-log enabled:true
```

상태 로그 채널을 따로 설정하지 않으면 제재/신고 로그 채널로 전송됩니다.

### 카테고리

지원하는 카테고리 값:

- `profanity`: 욕설
- `sexual`: 성적발언
- `family_insult`: 패드립
- `harassment`: 괴롭힘/모욕
- `hate`: 혐오/차별
- `threat`: 위협
- `other`: 기타

카테고리와 함께 금지어 등록:

```text
/filter add-word term:... category:family_insult severity:3
```

메시지 ID로 학습:

```text
/filter learn-message message_id:123456789012345678 term:... category:sexual severity:2 channel:#general
```

이의제기는 자동으로 학습에 반영되지 않습니다. 사용자는 다음처럼 신고합니다:

```text
/filter report case_id:12 reason:오탐입니다
```

이후 봇 설정 관리자가 승인합니다:

```text
/filter resolve-report report_id:3 outcome:false_positive
```

승인 후에만 PrettyWords가 해당 사례를 비속어 아님으로 학습합니다.

Discord Developer Portal에서 봇의 **Message Content Intent**를 켜야 메시지 내용을 검사할 수 있습니다. 멤버 인텐트는 기본으로 꺼져 있으며, 필요하면 `ENABLE_MEMBERS_INTENT=true`로 켭니다.

## 실행

```powershell
python main.py
```

봇 초대 권한:

- 채널 보기 (`View Channels`)
- 메시지 보내기 (`Send Messages`)
- 메시지 관리 (`Manage Messages`)
- 멤버 제재 (`Moderate Members`)
- 슬래시 명령어 사용 (`Use Slash Commands`)

봇 역할은 제재 대상 역할보다 위에 있어야 타임아웃이 적용됩니다.

## 주요 명령어

- `/filter log-channel #채널`
- `/filter health-log-channel #채널`
- `/filter timeout minutes:10`
- `/filter threshold confidence:0.78`
- `/filter pause`, `/filter resume`
- `/filter disable-channel #채널`, `/filter enable-channel #채널`
- `/filter add-word term category severity`, `/filter remove-word term`
- `/filter learn-message message_id term category severity channel`
- `/filter allow-word term`, `/filter remove-allow term`
- `/filter report reason case_id`
- `/filter resolve-report report_id outcome term`
- `/filter mode dry_run ai_enabled delete_messages dm_users escalate`
- `/filter ai provider model scan_all`, `/filter ai-reset`
- `/filter health`, `/filter health-log enabled`
- `/pw config-admin-add user_id`, `/pw config-admin-remove user_id`, `/pw config-admin-list`
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
