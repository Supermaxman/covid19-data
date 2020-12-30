
import os
import json
import argparse
from sklearn.model_selection import GroupKFold
import random
import numpy as np

from data_utils import read_jsonl


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-s', '--splits', default=5, type=float)
	parser.add_argument('-d', '--seed', default=0, type=float)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	data = read_jsonl(args.input_path)
	data_groups = [int(t['misconceptions'][0]['misconception_id']) for t in data]

	group_kfold = GroupKFold(n_splits=args.splits)
	split_id = 1
	for train_index, test_index in group_kfold.split(data, groups=data_groups):
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
		with open(os.path.join(args.output_path, f'split_{split_id}.json'), 'w') as f:
			json.dump(split, f, indent=2)
		split_id += 1
