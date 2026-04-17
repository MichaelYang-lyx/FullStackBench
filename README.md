# FullStackBench

## 环境配置

```bash
uv venv
uv pip install -r requirements.txt
```

## 启动 Sandbox

```bash
docker run -d --rm -p 8080:8080 volcengine/sandbox-fusion:server-20241204
```

国内镜像：

```bash
docker run -d --rm -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20241204
```

## 运行

### Anthropic

```bash
python src/main_anthropic.py \
  --model claude-sonnet-4-6 \
  --url https://your-api-url \
  --key sk-xxx \
  --parallelism 20 \
  --batch-size 20 \
  --max-tokens 32000
```

### OpenAI

```bash
python src/main_openai.py \
  --model gpt-4o \
  --url https://api.openai.com/v1 \
  --key sk-xxx \
  --parallelism 10 \
  --batch-size 10 \
  --max-tokens 32000
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | - | 模型名称 |
| `--url` | 是 | - | 模型 API base URL |
| `--key` | 是 | - | API Key |
| `--parallelism` | 否 | 20 / 10 | 并发数 |
| `--batch-size` | 否 | 20 / 10 | 批次大小 |
| `--max-tokens` | 否 | 32000 | 最大输出 token 数 |
| `--sandbox-url` | 否 | http://localhost:8080 | Sandbox 地址 |
