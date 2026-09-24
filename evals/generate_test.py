"""
评测脚本

评测指标:
  - Faithfulness: 答案引用的法条是否都在 context 中
  - Hallucination Rate: 答案中 context 外的法条引用比例
  - Citation Count: 平均每条答案引用几条法条
"""

import json
import os
import re

def evaluate_generate(all_retrieve: list[dict]):

    results = []
    for i, test in enumerate(all_retrieve):
        chunk_list = test['chunk_list']
        answer = test['answer']
        expected = []
        for item in chunk_list:
            expected.extend(
                re.findall('第[零一二三四五六七八九十百千]+条', item)
            )
        expected = set(expected)
        real = set(re.findall(r'第[零一二三四五六七八九十百千]+条', answer))

        # 幻觉（被引用但是没被检索的条数）
        hallucination = real - expected

        # 缺失（检索到但是未被用到的条数）
        miss = expected - real

        # 是否忠诚
        faithfully = len(hallucination) == 0

        # 幻觉率
        hallucination_rate = round(len(hallucination) / len(real), 4) if real else 0.0

        results.append({
            'question': test['question'],
            'expected': list(expected),
            'real': list(real),
            'hallucination': list(hallucination),
            'miss': list(miss),
            'faithful': faithfully,
            'hallucination_rate': hallucination_rate,
        })
        print(f'测评进度：{i+1} / {len(all_retrieve)}')
    # 平均幻觉率
    hallucination_mean = round(sum(result['hallucination_rate'] for result in results) / len(all_retrieve))
    # 忠诚率
    faithful_mean = round(sum(1 for result in results if result['faithful']) / len(all_retrieve))
    total_count = len(all_retrieve)
    summary = {
        'hallucination_mean': hallucination_mean,
        'faithful_mean': faithful_mean,
        'total_count': total_count
    }
    return results, summary

def main():
    with open('evals/results/retrieve_result.json', 'r', encoding = 'utf-8') as f:
        retrieve_result = json.load(f)
    results, summary = evaluate_generate(retrieve_result)
    base = 'evals/results'
    results_path = os.path.join(base, '生成测评结果.json')
    summary_path = os.path.join(base, '生成测试数据汇总.json')

    with open(results_path, 'w', encoding = 'utf-8') as f:
        json.dump(results, f, ensure_ascii = False, indent = 2)
    with open(summary_path, 'w', encoding = 'utf-8') as f:
        json.dump(summary, f, ensure_ascii = False, indent = 2)
    print(f"""
生成测试结果分别保存在：
{results_path},
{summary_path}
    """)

if __name__ == '__main__':
    main()

