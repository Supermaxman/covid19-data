
import json
import argparse
import torch


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
	parser.add_argument('-hp', '--hera_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', type=float, required=True)
	parser.add_argument('-sna', '--skip_na', default=False, action='store_true')

	args = parser.parse_args()
	scores = {}
	with open(args.input_path) as f:
			# [twitter_id] -> (m_id, m_scores...)
			f_scores = json.load(f)
			for tweet_id, t_scores in f_scores.items():
				# should only be one for each hera tweet
				scores[tweet_id] = t_scores[0]

	score_func = torch.nn.Softmax(dim=-1)
	predictions = {}
	for tweet_id, m_score in scores.items():
		logits = torch.tensor([m_score['0_score'], m_score['1_score'], m_score['2_score']], dtype=torch.float)
		preds = get_predictions(logits, args.threshold, score_func).tolist()
		if args.skip_na and preds == 0:
			continue
		predictions[tweet_id] = preds

	with open(args.hera_path) as f:
		tweets = json.load(f)
		tweets = {t['id_str']: t for t in tweets}

	filtered_tweets = []
	for tweet_id, pred in predictions.items():
		tweet = tweets[tweet_id]
		tweet['misinformation']['predicted_label'] = pred
		filtered_tweets.append(tweet)

	with open(args.output_path, 'w') as f:
		json.dump(filtered_tweets, f, indent=2)
