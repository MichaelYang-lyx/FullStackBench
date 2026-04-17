import argparse
import json
import os
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
args = parser.parse_args()

set_endpoint(args.sandbox_url)
samples = read_jsonl('./data/fsb_en_20241204.jsonl')

client = OpenAI(
    api_key=args.key,
    base_url=args.url,
)

MODEL_NAME = args.model
FILE_NAME = 'results_' + MODEL_NAME.replace('/', '_') + '.jsonl'


@configurable_retry(5)
def single_inference(prompt: str) -> str:
    for i in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=args.max_tokens,
            )
            return completion.choices[0].message.content
        except Exception:
            time.sleep(5)
            continue
    return "error"

def process_sample(sample):
    raw_response = single_inference(sample['content'])
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

if __name__ == "__main__":
    main(output_file=FILE_NAME, batch_size=args.batch_size, parallelism=args.parallelism)
