
import json
import argparse
from data_utils import label_text_to_id, read_jsonl
from collections import defaultdict


def calculate_metrics(labels, predictions):
	for tweet_id, t_labels in labels.items():
		t_run = predictions[tweet_id]

		tp = len(t_labels.intersection(t_run))

		fn = len(t_labels.difference(t_run))

		fp = len(t_run.difference(t_labels))

		p = tp / (max(tp + fp, 1))
		r = tp / (max(tp + fn, 1))
		f1 = 2.0 * (p * r) / (max(p + r, 1))
		return p, r, f1


def filter_by_label(data):
	new_data = defaultdict(lambda: defaultdict(set))
	for tweet_id, labels in data.items():
		for m_id, m_label in labels:
			new_data[m_label][tweet_id].add(m_id)
	return new_data


def filter_by_misconception(data):
	new_data = defaultdict(lambda: defaultdict(set))
	for tweet_id, labels in data.items():
		for m_id, m_label in labels:
			new_data[m_id][tweet_id].add(m_label)
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
	macro_f1 = 0.0
	macro_p = 0.0
	macro_r = 0.0
	class_labels = filter_by_label(labels)
	class_run = filter_by_label(predictions)
	for i in range(num_labels):
		i_labels = class_labels[i]
		i_run = class_run[i]
		i_precision, i_recall, i_f1 = calculate_metrics(i_labels, i_run)

		macro_f1 += i_f1
		macro_p += i_precision
		macro_r += i_recall

	macro_f1 = macro_f1 / num_labels
	macro_p = macro_p / num_labels
	macro_r = macro_r / num_labels

	# overall eval
	print(f'P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}')


	# misinformation eval

