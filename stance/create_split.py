
import os
import json
import argparse
from sklearn.model_selection import GroupKFold, KFold
import random
import numpy as np
from collections import defaultdict

from data_utils import read_jsonl


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-s', '--splits', default=5, type=float)
	parser.add_argument('-d', '--seed', default=0, type=float)
	parser.add_argument('-t', '--split_type', default='group')
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	data = read_jsonl(args.input_path)
	split_type = args.split_type.lower()
	if split_type == 'group':
		data_groups = [int(t['misconceptions'][0]['misconception_id']) for t in data]

		group_kfold = GroupKFold(n_splits=args.splits)
		splits = group_kfold.split(data, groups=data_groups)
	elif split_type == 'group_relevant':
		data_groups = []
		num_more_one_group = 0
		num_data = len(data)
		outside_group_id = -1
		num_examples = defaultdict(int)
		for t in data:
			possible_groups = [int(m['misconception_id']) for m in t['misconceptions'] if m['label'] != 'na']
			for group in possible_groups:
				num_examples[group] += 1
				outside_group_id = max(outside_group_id, group)
		outside_group_id += 1
		for t in data:
			possible_groups = [int(m['misconception_id']) for m in t['misconceptions'] if m['label'] != 'na']
			if len(possible_groups) == 0:
				group = outside_group_id
				outside_group_id += 1
			else:
				if len(possible_groups) > 1:
					num_more_one_group += 1
					group = None
					group_count = -1
					# select most common group for split
					for p_g in possible_groups:
						p_g_count = num_examples[p_g]
						if p_g_count > group_count:
							group = p_g
					group = possible_groups[0]
				else:
					group = possible_groups[0]
			data_groups.append(group)
		print(f'{num_more_one_group / num_data:.2f} with more than one group')

		group_kfold = GroupKFold(n_splits=args.splits)
		splits = group_kfold.split(data, groups=data_groups)

	elif split_type == 'normal':
		kfold = KFold(n_splits=args.splits, shuffle=True)
		splits = kfold.split(data)
	else:
		raise ValueError(f'Unknown split type: {split_type}')
	split_id = 1
	for train_index, test_index in splits:
		train_data = []
		for idx in train_index:
			train_data.append(data[idx])
		test_data = []
		for idx in test_index:
			test_data.append(data[idx])

		split = {
			'train': train_data,
			'eval': test_data
		}
		print(f'{split_id}: train={len(train_data)} test={len(test_data)}')
		with open(os.path.join(args.output_path, f'{split_type}_split_{split_id}.json'), 'w') as f:
			json.dump(split, f, indent=2)
		split_id += 1
