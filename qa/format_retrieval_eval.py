
import json
import argparse
from collections import defaultdict
import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()
	scores = {}
	for input_path in args.input_path.split(','):
		if input_path:
			with open(input_path) as f:
				# [twitter_id] -> [m_id, m_scores...]
				scores.update(json.load(f))

	score_func = torch.nn.Softmax(dim=-1)
	predictions = defaultdict(list)
	for tweet_id, m_scores in scores.items():
		for m_score in m_scores:
			irrelevant_score = m_score['0_score']
			agree_score = m_score['1_score']
			disagree_score = m_score['2_score']
			# TODO come up with better ranking
			score = -irrelevant_score
			predictions[tweet_id].append((m_score['question_id'], score))
		predictions[tweet_id] = list(sorted(predictions[tweet_id], key=lambda x: x[1], reverse=True))

	with open(args.output_path, 'w') as f:
		json.dump(predictions, f, indent=2)
