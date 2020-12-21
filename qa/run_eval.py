
import json
import argparse
from data_utils import label_text_to_id, read_jsonl
from collections import defaultdict


def calculate_metrics(labels, predictions):
	tp = len(labels.intersection(predictions))

	fn = len(labels.difference(predictions))

	fp = len(predictions.difference(labels))

	p = tp / (max(tp + fp, 1))
	r = tp / (max(tp + fn, 1))
	f1 = 2.0 * (p * r) / (max(p + r, 1))
	return p, r, f1


def filter_by_label(data):
	new_data = defaultdict(set)
	for tweet_id, labels in data.items():
		for m_id, m_label in labels:
			new_data[m_label].add((tweet_id, m_id))
	return new_data


def filter_by_misconception(data):
	new_data = defaultdict(set)
	for tweet_id, labels in data.items():
		for m_id, m_label in labels:
			new_data[m_id].add((tweet_id, m_label))
	return new_data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--label_path', required=True)
	parser.add_argument('-r', '--run_path', required=True)
	args = parser.parse_args()

	labels = read_jsonl(args.label_path)
	labels = {t['id_str']: t['misconceptions'] for t in labels}
	# [tweet_id] -> set((m_id, m_label))
	labels = {
		k: set([(m['misconception_id'], label_text_to_id(m['label'])) for m in v]) for k, v in labels.items()
	}

	with open(args.run_path) as f:
		predictions = json.load(f)
		# [tweet_id] -> set((m_id, m_label))
		predictions = {k: set([(m_id, m_l) for m_id, m_l in v]) for k, v in predictions.items()}

	# class eval
	num_labels = 3
	macro_p = 0.0
	macro_r = 0.0
	macro_f1 = 0.0

	class_labels = filter_by_label(labels)
	class_run = filter_by_label(predictions)
	result_titles = []
	result_values = []
	for i in [1, 2, 0]:
		i_precision, i_recall, i_f1 = calculate_metrics(
			class_labels[i],
			class_run[i]
		)
		result_titles.extend([f'{i}-P', f'{i}-R', f'{i}-F1'])
		macro_p += i_precision
		macro_r += i_recall
		macro_f1 += i_f1
		result_values.extend([f'{i_precision*100:.1f}', f'{i_recall*100:.1f}', f'{i_f1*100:.1f}'])

	macro_f1 = macro_f1 / num_labels
	macro_p = macro_p / num_labels
	macro_r = macro_r / num_labels

	result_titles = ['M-P', 'M-R', 'M-F1'] + result_titles
	result_values = [f'{macro_p*100:.1f}', f'{macro_r*100:.1f}', f'{macro_f1*100:.1f}'] + result_values

	titles_text = ' '.join(result_titles)
	values_text = ' '.join(result_values)
	print(titles_text)
	print(values_text)

	# misinformation eval

