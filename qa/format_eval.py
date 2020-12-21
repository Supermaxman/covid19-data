
import os
import json
import torch
import argparse
from collections import defaultdict
import torch

#
# def load_predictions(input_path):
# 	pred_list = []
# 	for file_name in os.listdir(input_path):
# 		if file_name.endswith('.pt'):
# 			preds = torch.load(os.path.join(input_path, file_name))
# 			pred_list.extend(preds)
# 	scores = defaultdict(list)
#
# 	for prediction in pred_list:
# 		tweet_id = prediction['id']
# 		question_id = prediction['question_id']
# 		scores[tweet_id].append(
# 			{
# 				'question_id': question_id,
# 				'0_score': prediction['0_score'],
# 				'1_score': prediction['1_score'],
# 				'2_score': prediction['2_score'],
# 			}
# 		)
#
# 	return scores
#


def get_predictions(logits, threshold, score_func):
	# non-zero class probs
	# [num_labels-1]
	pos_probs = score_func(logits)[1:]
	# filter out non-thresholded classes
	# [num_labels-1]
	pos_probs = pos_probs * ((pos_probs > threshold).float())
	# 1 if any are above threshold, 0 if none are above threshold
	# []
	pos_any_above = ((pos_probs > threshold).int().sum(dim=-1) > 0).int()
	# if none are above threshold then our prediction will be class 0, otherwise it will be
	# between the classes which have probs above the threshold
	# []
	# we add one to the class id to account for the [:, 1:] filtering of only positive probs
	pos_predictions = (pos_probs.max(dim=-1)[1] + 1)
	# []
	predictions = pos_predictions * pos_any_above
	return predictions


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', type=float, required=True)
	args = parser.parse_args()

	with open(args.input_path) as f:
		# [twitter_id] -> [m_id, m_scores...]
		scores = json.load(f)

	score_func = torch.nn.Softmax(dim=-1)
	predictions = defaultdict(list)
	for tweet_id, m_scores in scores.items():
		for m_score in m_scores:
			logits = torch.tensor([m_score['0_score'], m_score['1_score'], m_score['2_score']], dtype=torch.float)
			preds = get_predictions(logits, args.threshold, score_func)
			predictions[tweet_id].append((m_score['misconception_id'], preds))

	with open(args.output_path, 'w') as f:
		json.dump(predictions, f, indent=2)
