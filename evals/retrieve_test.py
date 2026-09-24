"""
检索测评脚本
"""
import math
import os
from collections import defaultdict
import json

def evaluate_retrieve(test_set: list[dict], all_retrieved: list[dict]):
    simply_save = defaultdict(list)
    difficulty_based_save = defaultdict(lambda:defaultdict(list))
    j = 0
    for test, retrieved in zip(test_set, all_retrieved):
        expected = set(test['expected_articles'])
        difficulty = test['difficulty']


        # 召回率
        for k in [5, 10, 15]:
            hits = expected & set(retrieved[:k])
            res = len(hits) / len(expected)
            simply_save[f'recall@{k}'].append(res)
            difficulty_based_save[difficulty][f'recall{k}'].append(res)

        # 正确答案在所有召回中的占比
        for k in [5, 10, 15]:
            hit = expected & set(retrieved[:k])
            res = len(hit) / k
            simply_save[f'precision@{k}'].append(res)
            difficulty_based_save[difficulty][f'precision@{k}'].append(res)

        # MRR （第一个正确答案排名的倒数）
        mrr = 0.0
        for rank, r in enumerate(retrieved, 1):
            if r in expected:
                mrr = 1 / rank
        simply_save['mrr'].append(mrr)
        difficulty_based_save[difficulty]['mrr'].append(mrr)

        # NDCG =（归一化折扣累积增益）
        dcg = 0.0 # 实际
        idcg = 0.0 # 理想状态
        for k in [5, 10, 15]:
            for rank, r in enumerate(retrieved[:k], 1):
                if r in expected:
                    dcg += 1 / math.log2(rank+1)

            for i in range(1, min(k, len(expected))+1):
                idcg += 1 / math.log2(i+1)

        ndcg = dcg / idcg
        simply_save[f'ndcg@{k}'].append(ndcg)
        difficulty_based_save[difficulty][f'ndcg@{k}'].append(ndcg)

        print(f'检索测评进度{j+1}/{len(test_set)}')
        j += 1

    # 汇总
    summary = {}
    difficulty_based_summary = {}

    for k, v in simply_save.items():
        summary[k] = {
            'min': min(v),
            'mean': round(sum(v) / len(v), 2),
            'max': max(v)
        }

    for diff, item in difficulty_based_save.items():
        difficulty_based_summary[diff] = {}
        for k, v in item.items():
            difficulty_based_summary[diff][k] = {
                'min': min(v),
                'mean': round(sum(v) / len(v), 2),
                'max': max(v)
            }
    return summary, difficulty_based_summary

def main():
    test_set_path = 'evals/test_set.json'
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    retrieved_path = 'evals/results/retrieve_result.json'
    with open(retrieved_path, 'r', encoding='utf-8') as f:
        retrieve_result = json.load(f)
    all_retrieved = [r['ids'] for r in retrieve_result]

    summary, difficulty_based_summary = evaluate_retrieve(test_set, all_retrieved)

    base = 'evals/results'
    summary_path = os.path.join(base, 'retrieve_evaluate.json')
    with open(summary_path, 'w', encoding = 'utf-8') as f:
        json.dump(summary, f, ensure_ascii = False, indent = 2)
    difficulty_based_summary_path = os.path.join(base, 'retrieve_evaluate_diff_base.json')
    with open(difficulty_based_summary_path, 'w', encoding = 'utf-8') as f:
        json.dump(difficulty_based_summary, f, ensure_ascii = False, indent = 2)

    print(f'''
已将检索测试结果分别保存在：
{summary_path},
{difficulty_based_summary_path}            
''')

if __name__ == '__main__':
    main()
