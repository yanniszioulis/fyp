#!/usr/bin/env python3
"""Compare transformer vs persistence results."""

import json

# Load results
with open('results/metrics/persistence_results.json', 'r') as f:
    persistence = json.load(f)

with open('results/metrics/transformer_results.json', 'r') as f:
    transformer = json.load(f)

# Create lookup dictionaries
persistence_dict = {}
for r in persistence:
    key = (r['window_id'], r['context_length'], r['horizon'])
    persistence_dict[key] = r

transformer_dict = {}
for r in transformer:
    key = (r['window_id'], r['context_length'], r['horizon'])
    transformer_dict[key] = r

# Compare
print('=' * 80)
print('CONFIGURATIONS WHERE TRANSFORMER BEATS PERSISTENCE')
print('=' * 80)
print()

better_configs = []
worse_configs = []
equal_configs = []

for key in transformer_dict:
    if key in persistence_dict:
        trans_rmse = transformer_dict[key]['metrics']['iv_rmse']
        pers_rmse = persistence_dict[key]['metrics']['iv_rmse']
        improvement = (pers_rmse - trans_rmse) / pers_rmse * 100
        
        config = {
            'window': key[0],
            'context': key[1],
            'horizon': key[2],
            'transformer_rmse': trans_rmse,
            'persistence_rmse': pers_rmse,
            'improvement_pct': improvement
        }
        
        if trans_rmse < pers_rmse:
            better_configs.append(config)
        elif trans_rmse > pers_rmse:
            worse_configs.append(config)
        else:
            equal_configs.append(config)

# Sort by improvement
better_configs.sort(key=lambda x: x['improvement_pct'], reverse=True)
worse_configs.sort(key=lambda x: x['improvement_pct'])

total = len(better_configs) + len(worse_configs) + len(equal_configs)
print(f'Total configurations: {total}')
print(f'Transformer better: {len(better_configs)} ({len(better_configs)/total*100:.1f}%)')
print(f'Persistence better: {len(worse_configs)} ({len(worse_configs)/total*100:.1f}%)')
print(f'Equal: {len(equal_configs)}')
print()

if better_configs:
    print('TOP 20 CONFIGS WHERE TRANSFORMER BEATS PERSISTENCE:')
    print('-' * 80)
    header = f"{'Window':<8} {'Context':<10} {'Horizon':<10} {'Trans RMSE':<12} {'Pers RMSE':<12} {'Improvement':<12}"
    print(header)
    print('-' * 80)
    for cfg in better_configs[:20]:
        print(f"{cfg['window']:<8} {cfg['context']:<10} {cfg['horizon']:<10} {cfg['transformer_rmse']:<12.6f} {cfg['persistence_rmse']:<12.6f} {cfg['improvement_pct']:>+10.2f}%")
    print()

if worse_configs:
    print('WORST 10 CONFIGS (where persistence beats transformer):')
    print('-' * 80)
    header = f"{'Window':<8} {'Context':<10} {'Horizon':<10} {'Trans RMSE':<12} {'Pers RMSE':<12} {'Worse by':<12}"
    print(header)
    print('-' * 80)
    for cfg in worse_configs[:10]:
        worse_by = -cfg['improvement_pct']
        print(f"{cfg['window']:<8} {cfg['context']:<10} {cfg['horizon']:<10} {cfg['transformer_rmse']:<12.6f} {cfg['persistence_rmse']:<12.6f} {worse_by:>+10.2f}%")
    print()

# Group by context/horizon
print('=' * 80)
print('BREAKDOWN BY CONTEXT/HORIZON:')
print('=' * 80)
print()

from collections import defaultdict
by_config = defaultdict(lambda: {'better': 0, 'worse': 0, 'total': 0})

for cfg in better_configs + worse_configs:
    key = (cfg['context'], cfg['horizon'])
    by_config[key]['total'] += 1
    if cfg in better_configs:
        by_config[key]['better'] += 1
    else:
        by_config[key]['worse'] += 1

header = f"{'Context':<10} {'Horizon':<10} {'Better':<10} {'Worse':<10} {'Win Rate':<10}"
print(header)
print('-' * 50)
for (ctx, hz) in sorted(by_config.keys()):
    stats = by_config[(ctx, hz)]
    win_rate = stats['better'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"{ctx:<10} {hz:<10} {stats['better']:<10} {stats['worse']:<10} {win_rate:>8.1f}%")
