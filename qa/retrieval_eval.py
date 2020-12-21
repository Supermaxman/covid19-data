
import json
import argparse
from data_utils import read_jsonl, label_text_to_id
from collections import defaultdict


def calculate_mrr(labels, scores):
	mrr = 0.0
	for tweet_id in labels:
		t_mrr = 0.0
		t_labels = labels[tweet_id]
		t_scores = scores[tweet_id]
		for rank, (m_id, m_score) in enumerate(t_scores, start=1):
			r_rank = 1.0 / rank
			if m_id in t_labels:
				t_mrr += r_rank
		t_mrr = t_mrr / len(t_labels)
		mrr += t_mrr
	mrr = mrr / len(labels)
	return mrr


def calculate_hits(labels, scores, h):
	hits = 0.0
	for tweet_id in labels:
		t_hits = 0.0
		t_labels = labels[tweet_id]
		t_scores = scores[tweet_id]
		for m_id, m_score in t_scores[:h]:
			if m_id in t_labels:
				t_hits += 1.0
		# TODO is this correct?
		t_hits = t_hits / min(len(t_labels), h)
		hits += t_hits
	hits = hits / len(labels)
	return hits


def calculate_metrics(labels, scores):
	# H@1 H@5 H@10 MRR
	h1 = calculate_hits(labels, scores, h=1)
	h5 = calculate_hits(labels, scores, h=5)
	h10 = calculate_hits(labels, scores, h=10)
	mrr = calculate_mrr(labels, scores)
	return h1, h5, h10, mrr


def filter_by_label(data, relevant_labels):
	new_data = defaultdict(set)
	for tweet_id, l_misconceptions in data.items():
		for m_label in relevant_labels:
			for m_id in l_misconceptions[m_label]:
				new_data[tweet_id].add(m_id)
	return new_data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--label_path', required=True)
	parser.add_argument('-r', '--run_path', required=True)
	args = parser.parse_args()

	labels = read_jsonl(args.label_path)
	labels = {t['id_str']: t['misconceptions'] for t in labels}
	# [tweet_id][m_label] -> set(m_id)
	modified_labels = defaultdict(lambda: defaultdict(set))
	for tweet_id, misconceptions in labels.items():
		for m in misconceptions:
			m_id = m['misconception_id']
			m_label = label_text_to_id(m['label'])
			modified_labels[tweet_id][m_label].add(m_id)
	labels = modified_labels

	with open(args.run_path) as f:
		# [tweet_id] -> sorted list of (m_id, m_score)
		scores = json.load(f)

	# H@1 H@5 H@10 MRR for AGREE and RELEVANT (AGREE OR DISAGREE)
	result_titles = []
	result_values = []
	agree_labels = filter_by_label(labels, [1])
	a_h1, a_h5, a_h10, a_mrr = calculate_metrics(agree_labels, scores)
	result_titles.extend(['A-H@1', 'A-H@5', 'A-H@10', 'A-MRR'])
	result_values.extend([f'{a_h1 * 100:.1f}', f'{a_h5 * 100:.1f}', f'{a_h10 * 100:.1f}', f'{a_mrr:.2f}'])
	relevant_labels = filter_by_label(labels, [1, 2])
	r_h1, r_h5, r_h10, r_mrr = calculate_metrics(relevant_labels, scores)
	result_titles.extend(['R-H@1', 'R-H@5', 'R-H@10', 'R-MRR'])
	result_values.extend([f'{r_h1 * 100:.1f}', f'{r_h5 * 100:.1f}', f'{r_h10 * 100:.1f}', f'{r_mrr:.2f}'])

	titles_text = ' '.join(result_titles)
	values_text = ' '.join(result_values)
	print(titles_text)
	print(values_text)


