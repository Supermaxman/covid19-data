
import json
import argparse
from collections import defaultdict
import numpy as np
import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()
	scores = defaultdict(list)
	for input_path in args.input_path.split(','):
		if input_path:
			with open(input_path) as f:
				f_scores = json.load(f)
				for tweet_id, t_scores in f_scores.items():
					scores[tweet_id].extend(t_scores)

	predictions = defaultdict(list)
	score_func = torch.nn.Softmax(dim=-1)
	for tweet_id, m_scores in scores.items():
		m_score_list = defaultdict(list)
		for m_score in m_scores:
			irrelevant_score = m_score['0_score']
			agree_score = m_score['1_score']
			disagree_score = m_score['2_score']
			m_id = m_score['question_id']
			logits = torch.tensor([irrelevant_score, agree_score, disagree_score], dtype=torch.float)
			i_score, a_score, d_score = score_func(logits).tolist()
			# TODO come up with better ranking
			score = max(a_score, d_score)
			# score = agree_score
			m_score_list[m_id].append(score)

		# multiple models will make a prediction, take average score
		for m_id, m_s_list in m_score_list.items():
			m_score = np.mean(m_s_list)
			predictions[tweet_id].append((m_id, m_score))

		predictions[tweet_id] = list(sorted(predictions[tweet_id], key=lambda x: x[1], reverse=True))

	with open(args.output_path, 'w') as f:
		json.dump(predictions, f, indent=2)
