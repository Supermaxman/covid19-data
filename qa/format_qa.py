
import os
import json
import torch
import argparse
from collections import defaultdict


def load_predictions(input_path):
	pred_list = []
	for file_name in os.listdir(input_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(input_path, file_name))
			pred_list.extend(preds)
	scores = defaultdict(list)

	for prediction in pred_list:
		tweet_id = prediction['id']
		question_id = prediction['question_id']
		scores[tweet_id].append(
			{
				'question_id': question_id,
				'0_score': prediction['0_score'],
				'1_score': prediction['1_score'],
				'2_score': prediction['2_score'],
			}
		)

	return scores


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	predictions = load_predictions(args.input_path)

	with open(args.output_path, 'w') as f:
		json.dump(predictions, f, indent=2)
