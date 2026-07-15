import argparse
import json
import os
import threading
from pathlib import Path
from openai import OpenAI
from sandbox_fusion import (
    SubmitRequest,
    TestConfig,
    set_endpoint,
    submit,
)
from utils import (
    configurable_retry,
    read_jsonl,
    write_jsonl,
)
from tqdm import tqdm
import concurrent.futures
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='模型名称')
parser.add_argument('--url', required=True, help='模型 base_url')
parser.add_argument('--key', required=True, help='API key')
parser.add_argument('--parallelism', type=int, default=10, help='并发数')
parser.add_argument('--batch-size', type=int, default=10, help='批次大小')
parser.add_argument('--max-tokens', type=int, default=32000, help='最大 token 数')
parser.add_argument('--sandbox-url', type=str, default='http://localhost:8080', help='sandbox endpoint')
parser.add_argument('--output-dir', type=str, default='.', help='结果输出目录')
parser.add_argument('--temperature', type=float, default=0.6, help='采样温度')
parser.add_argument('--extra-body', type=str, default='', help='额外请求体 JSON 字符串')
parser.add_argument('--stream', action='store_true', help='使用流式响应')
parser.add_argument('--limit', type=int, default=0, help='仅评测前 N 个样本（0 表示全部，用于小样本验证）')
args = parser.parse_args()

set_endpoint(args.sandbox_url)
samples = read_jsonl('./data/fsb_en_20241204.jsonl')
if args.limit and args.limit > 0:
    samples = samples[:args.limit]

client = OpenAI(
    api_key=args.key,
    base_url=args.url,
)

MODEL_NAME = args.model
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
FILE_NAME = str(output_dir / ('results_' + MODEL_NAME.replace('/', '_') + '.jsonl'))
# token 逐条记录文件：优先用 sb3 注入的环境变量，否则落在 output_dir
TOKEN_RECORDS_FILE = os.getenv('FSB_TOKEN_RECORDS') or str(output_dir / 'token_records.jsonl')

# 解析 extra_body
extra_body = None
if args.extra_body:
    try:
        extra_body = json.loads(args.extra_body)
    except Exception:
        extra_body = None

# 线程安全的 token 逐条记录写入
_token_lock = threading.Lock()

# reuse 场景下已写入过 token record 的 sample_id 集合（方案 A：不去重聚合，靠入口过滤防重）
# main() 启动时预加载现有 token_records.jsonl 中所有带 sample_id 的记录到此 set；
# _append_token_record 写入前判 skip；写入后加入 set。
# 老记录若无 sample_id 不进 set，会照常追加（legacy 段全保留，与聚合器口径一致）。
_written_sample_ids: set = set()


def _preload_written_sample_ids(path: str) -> None:
    """启动时读取现有 token_records.jsonl，把已有 sample_id 载入 _written_sample_ids。"""
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get('sample_id')
                if sid is not None and sid != '':
                    _written_sample_ids.add(str(sid))
        if _written_sample_ids:
            print(f"已加载 {len(_written_sample_ids)} 个 token_records sample_id（跳过重复写入）")
    except Exception as e:
        print(f"预加载 token_records sample_id 时出错: {e}（不影响主流程，继续）")

# tiktoken 兜底编码器（API 不给 reasoning_tokens 时用于分别数 think / prediction）
try:
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')
except Exception:
    _ENC = None


def _tiktoken_len(text):
    if not text or _ENC is None:
        return 0
    try:
        return len(_ENC.encode(text))
    except Exception:
        return 0


def _split_think(text):
    """把响应文本拆成 (think_text, prediction_text)，兼容 <think></think>。"""
    if not text:
        return '', ''
    if '</think>' in text:
        head, _, tail = text.partition('</think>')
        return head.replace('<think>', ''), tail
    return '', text


def _compute_tokens(usage, text=None, reasoning_text=None):
    """计算 (input, prediction, think) tokens，口径与 sensebench 对齐。"""
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    input_tokens = completion_tokens = reasoning_tokens = None
    if usage is not None:
        input_tokens = _get(usage, 'prompt_tokens')
        completion_tokens = _get(usage, 'completion_tokens')
        reasoning_tokens = _get(usage, 'reasoning_tokens')
        if reasoning_tokens is None:
            details = _get(usage, 'completion_tokens_details')
            if details is not None:
                reasoning_tokens = _get(details, 'reasoning_tokens')

    if reasoning_text:
        think_text, pred_text = reasoning_text, (text or '')
    else:
        think_text, pred_text = _split_think(text)

    # reasoning：usage 有则用；否则用 think 文本 tiktoken 兜底
    if reasoning_tokens is None or reasoning_tokens == 0:
        if think_text:
            think_tokens = _tiktoken_len(think_text)
            completion_tokens = None  # usage 的 completion 含 reasoning，弃用
        else:
            think_tokens = 0
    else:
        think_tokens = reasoning_tokens
        if completion_tokens is not None and completion_tokens >= reasoning_tokens:
            completion_tokens = completion_tokens - reasoning_tokens

    # prediction：completion 可用则用；否则用 prediction 文本 tiktoken 兜底
    if completion_tokens is None:
        prediction_tokens = _tiktoken_len(pred_text) if pred_text else 0
    else:
        prediction_tokens = completion_tokens
    return input_tokens, prediction_tokens, think_tokens


def _append_token_record(usage, text, reasoning_text=None, sample_id=None):
    """向 token_records.jsonl 追加一条记录（含 token 用量与 error/empty 标志）。

    sample_id (v2 新增): 样本唯一 id。方案 A 场景下：
      - 有 sample_id 且已在 _written_sample_ids 中 → skip（防止 reuse 补跑时重复写）
      - 有 sample_id 且未写过 → 写入并加入 set
      - 无 sample_id (legacy) → 照常写入（不参与去重）
    """
    try:
        sid_str = str(sample_id) if sample_id is not None else None
        # 加锁一次性检查+写入，保证并发下 set 与文件严格一致
        with _token_lock:
            if sid_str is not None and sid_str in _written_sample_ids:
                return
            input_tokens, prediction_tokens, think_tokens = _compute_tokens(
                usage, text=text, reasoning_text=reasoning_text
            )
            stripped = (text or '').strip()
            is_empty = stripped == ''
            is_error = stripped.upper().startswith('ERROR')
            rec = {
                'input_tokens': input_tokens,
                'prediction_tokens': prediction_tokens,
                'think_tokens': think_tokens,
                'is_error': is_error,
                'is_empty': is_empty,
            }
            if sid_str is not None:
                rec['sample_id'] = sid_str
            line = json.dumps(rec, ensure_ascii=False)
            with open(TOKEN_RECORDS_FILE, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
            if sid_str is not None:
                _written_sample_ids.add(sid_str)
    except Exception:
        pass


@configurable_retry(5)
def single_inference(prompt: str, sample_id=None) -> str:
    for i in range(3):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if extra_body:
                kwargs['extra_body'] = extra_body
            if args.stream:
                # 流式：逐 chunk 累加 content / reasoning_content，并通过 stream_options 拿到最终 usage
                kwargs['stream'] = True
                kwargs['stream_options'] = {"include_usage": True}
                completion = client.chat.completions.create(**kwargs)
                pieces = []
                reasoning_pieces = []
                final_usage = None
                for chunk in completion:
                    # 带 usage 的最终 chunk 通常 choices 为空
                    if getattr(chunk, 'usage', None):
                        final_usage = chunk.usage
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, 'content', None)
                        if piece:
                            pieces.append(piece)
                        r_piece = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                        if r_piece:
                            reasoning_pieces.append(r_piece)
                text = "".join(pieces)
                reasoning_text = "".join(reasoning_pieces) or None
                _append_token_record(final_usage, text, reasoning_text, sample_id=sample_id)
                return text
            completion = client.chat.completions.create(**kwargs)
            msg = completion.choices[0].message
            text = msg.content
            reasoning_text = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
            _append_token_record(completion.usage, text, reasoning_text, sample_id=sample_id)
            return text
        except Exception:
            time.sleep(5)
            continue
    # 三次都失败：记一条 error 记录（仍带 sample_id，reuse 覆盖旧记录时才能对上）
    _append_token_record(None, "ERROR: inference failed", sample_id=sample_id)
    return "error"

def process_sample(sample):
    raw_response = single_inference(sample['content'], sample_id=sample.get('id'))
    if not raw_response:
        raw_response = ''
    if '</think>' in raw_response:
        response = raw_response.split('</think>')[-1].strip()
    else:
        response = raw_response

    eval_result = submit(
        SubmitRequest(
            dataset='FullStackBench',
            id=sample['id'],
            completion=response,
            config=TestConfig(
                compile_timeout=50,
                run_timeout=50,
                dataset_type='AutoEvalDataset',
                provided_data=sample
            )
        )
    )
    return eval_result

def process_batch(batch, parallelism, output_file):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in batch}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batch"):
            result = future.result()
            results.append(result)

    results.sort(key=lambda x: x.id)

    with open(output_file, 'a') as f:
        for result in results:
            f.write(json.dumps(result.dict()) + '\n')

    return results

def main(output_file, batch_size=10, parallelism=10):
    # 方案 A：预加载已有 token_records.jsonl 中的 sample_id，防止 reuse 补跑时重复写入
    _preload_written_sample_ids(TOKEN_RECORDS_FILE)

    processed_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    processed_ids.add(result['id'])
            print(f"已加载 {len(processed_ids)} 个已处理的样本ID")
        except Exception as e:
            print(f"读取已处理结果时出错: {e}")
            print("将从头开始处理")

    total_samples = len(samples)
    remaining_samples = [s for s in samples if s['id'] not in processed_ids]

    print(f"总样本数: {total_samples}, 待处理样本数: {len(remaining_samples)}")

    all_results = []
    with tqdm(total=len(remaining_samples), desc="Overall progress", ncols=100) as overall_progress:
        for i in range(0, len(remaining_samples), batch_size):
            batch = remaining_samples[i:i+batch_size]
            batch_results = process_batch(batch, parallelism, output_file)
            overall_progress.update(len(batch))
            all_results.extend(batch_results)

        print(f"已处理 {len(all_results)} 个样本")

    if all_results:
        pass_rate = sum([r.accepted for r in all_results]) / len(all_results)
        print(f'本次运行通过率: {pass_rate:.4f}')
    else:
        print("没有新样本需要处理")

    total_results = []
    with open(output_file, 'r') as f:
        for line in f:
            total_results.append(json.loads(line))

    total_pass_rate = sum([r['accepted'] for r in total_results]) / len(total_results)
    print(f'总通过率: {total_pass_rate:.4f}')

    print(f"Token 逐条记录已写入: {TOKEN_RECORDS_FILE}")

if __name__ == "__main__":
    main(output_file=FILE_NAME, batch_size=args.batch_size, parallelism=args.parallelism)
