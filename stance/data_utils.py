import json
import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from collections import defaultdict
import senticnet5
import numpy as np
import string
import spacy


def read_jsonl(path):
	examples = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				examples.append(ex)
	return examples


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


class StanceBatchCollator(object):
	def __init__(
			self, tokenizer, max_seq_len: int, force_max_seq_len: bool,
			labeled=True):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len
		self.labeled = labeled

	def __call__(self, examples):
		ids = []
		labels = []
		question_ids = []
		scores = defaultdict(list)
		other_ids = defaultdict(list)
		batch_size = len(examples)
		pad_seq_len = 0
		for ex in examples:
			pad_seq_len = max(pad_seq_len, min(len(ex['input_ids']), self.max_seq_len))

		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		edges = {}

		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			if self.labeled:
				labels.append(ex['label'])
			question_ids.append(ex['question_id'])
			ex_input_ids = ex['input_ids'][:self.max_seq_len]
			ex_attention_mask = ex['attention_mask'][:self.max_seq_len]
			ex_token_type_ids = ex['token_type_ids'][:self.max_seq_len]
			input_ids[ex_idx, :len(ex_input_ids)] = torch.tensor(ex_input_ids, dtype=torch.long)
			attention_mask[ex_idx, :len(ex_attention_mask)] = torch.tensor(ex_attention_mask, dtype=torch.long)
			token_type_ids[ex_idx, :len(ex_token_type_ids)] = torch.tensor(ex_token_type_ids, dtype=torch.long)

			# TODO add back in when new tokenization is stable
			# for edge_name, edge_values in ex['edges'].items():
			# 	# truncation to max_seq_len, still need to pad
			# 	edge_values = edge_values[:self.max_seq_len, :self.max_seq_len]
			# 	batch_edge_name = edge_name + '_edges'
			# 	if batch_edge_name not in edges:
			# 		edges[batch_edge_name] = torch.zeros([batch_size, pad_seq_len, pad_seq_len], dtype=torch.float)
			# 	edges[batch_edge_name][ex_idx, :edge_values.shape[0], :edge_values.shape[1]] = torch.tensor(
			# 		edge_values,
			# 		dtype=torch.float
			# 	)

			for score_name, score_values in ex['scores'].items():
				scores[score_name + '_scores'].append(score_values)
				other_ids[score_name + '_ids'].append([i for i in range(len(score_values))])
			other_ids['stance_ids'].append([i for i in range(3)])

		batch = {
			'id': ids,
			'question_id': question_ids,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
		}
		if self.labeled:
			batch['labels'] = torch.tensor(labels, dtype=torch.long)

		for score_name, score_value in scores.items():
			# [bsize, num_labels]
			batch[score_name] = torch.tensor(score_value, dtype=torch.float)

		for id_name, id_value in other_ids.items():
			batch[id_name] = torch.tensor(id_value, dtype=torch.long)
		for edge_name, edge_value in edges.items():
			batch[edge_name] = edge_value

		return batch


def label_text_to_id(label):
	if label == 'na':
		return 0
	if label == 'pos':
		return 1
	if label == 'neg':
		return 2
	else:
		raise ValueError(f'Unknown label: {label}')


def hera_label_to_id(source, label_name):
	source = source.lower()
	label_name = label_name.lower()
	if 'real' in source:
		# source is a true fact
		if label_name == 'real':
			# agree
			return 1
		elif label_name == 'refutes':
			# refutation of true fact is disagreement
			return 2
		elif 'severe' in label_name:
			# severe misinformation of any level is a disagreement with the source
			return 2
		else:
			raise ValueError(f'Unknown label name: {label_name}')
	else:
		# source is misinformation
		if label_name == 'real':
			# disagree, if source is fake but tweet is real then we have disagreement with source
			return 2
		elif label_name == 'refutes':
			# refutation of misinformation is disagreement
			return 2
		elif 'severe' in label_name:
			# severe misinformation of any level is agreement with the source
			return 1
		else:
			raise ValueError(f'Unknown label name: {label_name}')


def format_predictions(preds, labels):
	values = [None for _ in range(len(labels))]
	for l_name, l_value in preds.items():
		label_idx = labels[l_name]
		values[label_idx] = l_value
	return values


# 0 'pleasantness_value',
# 1 'attention_value',
# 2 'sensitivity_value',
# 3 'aptitude_value',
# 4 'primary_mood',
# 5 'secondary_mood',
# 6 'polarity_label',
# 7 'polarity_value',
# 8 'semantics1',
# 9 'semantics2',
# 10 'semantics3',
# 11 'semantics4',
# 12 'semantics5'
def sentic_expand(sentic_edges, expand_list):
	new_edges = set(sentic_edges)
	for edge in sentic_edges:
		edge_info = senticnet5.senticnet[edge]
		for i in expand_list:
			new_edges.add(edge_info[i])
	return new_edges


def flatten(l):
	return [item for sublist in l for item in sublist]


def filter_tweet_text(tweet_text):
	# TODO consider @<user> and <url> replacing users and urls in tweets
	return tweet_text


def align_tokens(tokens, wpt_tokens, offset=0):
	align_map = {}
	for token in tokens:
		token['wpt_idxs'] = set()
		start = token['start'] + offset
		end = token['end'] + offset
		for char_idx in range(start, end):
			sub_token_idx = wpt_tokens.char_to_token(char_idx)
			# White spaces have no token and will return None
			if sub_token_idx is not None:
				align_map[sub_token_idx] = token
				token['wpt_idxs'].add(sub_token_idx)
	return align_map


def align_token_sequences(m_tokens, t_tokens, wpt_tokens, m_offset, tokenizer):
	print([f'{i}:{m}' for i, m in enumerate(wpt_tokens.tokens)])
	print([f'{m["start"]}:{m["end"]}:{m["text"]}' for m in m_tokens])
	print([f'{m["start"]}:{m["end"]}:{m["text"]}' for m in t_tokens])
	m_align_map = align_tokens(m_tokens, wpt_tokens)
	t_align_map = align_tokens(t_tokens, wpt_tokens, offset=m_offset)

	align_map = {**m_align_map, **t_align_map}
	print('align mapping')
	for key, value in align_map.items():
		print(f'{key} -> {value["start"]}:{value["end"]}:{value["text"]}')
	input()
	t_map = {}
	token_map = {}
	for sub_token_idx in range(len(wpt_tokens['input_ids'])):
		if sub_token_idx not in align_map:
			# CLS, SEP, or other special token
			aligned_token = {
				'pos': 'NONE',
				'dep': 'NONE',
				'head': 'NONE',
				'sentic': None,
				'text': '[CLS]' if sub_token_idx == 0 else '[SEP]'
			}
			align_map[sub_token_idx] = aligned_token
		aligned_token = align_map[sub_token_idx]
		# reverse_map[aligned_token['text']].append(sub_token_idx)
		# token_map[aligned_token['text']] = aligned_token
	print('align mapping')
	for key, value in align_map.items():
		print(f'{key} -> {value["text"]}')
	input()

	return t_map, reverse_map, token_map


def create_adjacency_matrix(edges, size, t_map, r_map):
	adj = np.eye(size, dtype=np.float32)
	for input_idx in range(size):
		input_idx_text = t_map[input_idx]
		i_semantic_edges = set(flatten([r_map[e_txt] for e_txt in edges[input_idx_text]]))
		for edge_idx in i_semantic_edges:
			adj[input_idx, edge_idx] = 1.0
			adj[edge_idx, input_idx] = 1.0
	return adj


def create_edges(m_tokens, t_tokens, wpt_tokens, num_semantic_hops, num_emotion_hops, num_lexical_hops, m_offset, tokenizer):
	seq_len = len(wpt_tokens['input_ids'])
	t_map, r_map, token_map = align_token_sequences(m_tokens, t_tokens, wpt_tokens, m_offset, tokenizer)

	semantic_edges = {}
	emotion_edges = {}
	reverse_emotion_edges = defaultdict(set)
	lexical_edges = {}
	root_text = None

	for token_text, token in token_map.items():
		text = token_text
		# TODO add as features
		pos = token['pos']
		dep = token['dep']
		head = token['head']
		if dep == 'ROOT':
			root_text = text
		sentic = token['sentic']
		if sentic is None:
			semantic_edges[text] = set()
			emotion_edges[text] = set()
		else:
			semantic_edges[text] = set(sentic['semantics'])
			for i in range(num_semantic_hops-1):
				semantic_edges[text] = sentic_expand(semantic_edges[text], [8, 9, 10, 11, 12])
			emotion_edges[text] = {sentic['primary_mood'], sentic['secondary_mood']}
			reverse_emotion_edges[sentic['primary_mood']].add(text)
			reverse_emotion_edges[sentic['secondary_mood']].add(text)

			# for emotion in [sentic['primary_mood'], sentic['secondary_mood']]:
			# 	emotion_edges[text] = emotion_edges[text].union(emotion_nodes[emotion])

			# for i in range(num_emotion_hops - 1):
			# 	new_emotions = sentic_expand(emotion_edges[text], [4, 5])
			# 	for emotion in new_emotions:
			# 		emotion_edges[text] = emotion_edges[text].union(emotion_nodes[emotion])

		lexical_edges[text] = {head}

	lexical_edges['[CLS]'] = {root_text}
	lexical_edges['[SEP]'] = {root_text}

	# TODO implement num_lexical_hops and emotion_hops
	# TODO issue with emotion hops: requires reverse emotion -> all tokens, really slow to compute > 1 hop
	# TODO determine if CLS and SEP should be attached
	# text -> emotion node -> other text in sentence with same emotions
	for text in emotion_edges.keys():
		emotions = emotion_edges[text]
		emotion_edges[text] = emotion_edges[text].union(
			set(flatten(reverse_emotion_edges[emotion] for emotion in emotions))
		)
	semantic_adj = create_adjacency_matrix(
		edges=semantic_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)
	emotion_adj = create_adjacency_matrix(
		edges=emotion_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)
	lexical_adj = create_adjacency_matrix(
		edges=lexical_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)

	# # CLS token is connected to everything
	# semantic_adj[:, 0] = 1.0
	# semantic_adj[0, :] = 1.0
	# # CLS token is connected to everything
	# emotion_adj[:, 0] = 1.0
	# emotion_adj[0, :] = 1.0
	# # CLS token is connected to everything
	# lexical_adj[:, 0] = 1.0
	# lexical_adj[0, :] = 1.0

	edges = {
		'semantic': semantic_adj,
		'emotion': emotion_adj,
		'lexical': lexical_adj,
	}
	return edges


def get_sentic(word_text):
	word_text = word_text.lower()
	if word_text == 'coronavirus' or word_text == 'covid-19' or word_text == 'covid' or word_text == 'covid19':
		word_text = 'virus'
	if word_text not in senticnet5.senticnet:
		word_text = word_text[:-1]
		if word_text not in senticnet5.senticnet:
			word_text = word_text[:-1]
			if word_text not in senticnet5.senticnet:
				return None
	p_v, a_v, s_v, ap_v, p_m, s_m, po_l, po_v, s1, s2, s3, s4, s5 = senticnet5.senticnet[word_text]
	return {
			'pleasantness_value': float(p_v),
			'attention_value': float(a_v),
			'sensitivity_value': float(s_v),
			'aptitude_value': float(ap_v),
			'primary_mood': p_m,
			'secondary_mood': s_m,
			'polarity_label': po_l,
			'polarity_value': float(po_v),
			'semantics': [s1, s2, s3, s4, s5],
	}


def get_token_features(token):
	sentic = get_sentic(token.text)
	token_data = {
		'text': token.text,
		'pos': token.pos_,
		'dep': token.dep_,
		'head': token.head.text,
		'sentic': sentic,
		'start': token.idx,
		'end': token.idx + len(token.text),
	}
	return token_data


class StanceDataset(Dataset):
	def __init__(
			self, documents=None, hera_documents=None,
			keep_real=False,
			sentiment_preds=None, emotion_preds=None, irony_preds=None,
			coaid_sentiment_preds=None, coaid_emotion_preds=None, coaid_irony_preds=None,
			sentiment_labels=None, emotion_labels=None, irony_labels=None,
			tokenizer=None,
			create_edge_features=False,
			num_semantic_hops=None, num_emotion_hops=None, num_lexical_hops=None,
			mis_info=None, add_mis_info=False,
			labeled=True):
		self.examples = []
		self.num_labels = defaultdict(int)
		# TODO follow https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
		# TODO for graph adjacency features
		if sentiment_preds is None:
			sentiment_preds = {}
			coaid_sentiment_preds = {}
		if emotion_preds is None:
			emotion_preds = {}
			coaid_emotion_preds = {}
		if irony_preds is None:
			irony_preds = {}
			coaid_irony_preds = {}
		# if use_token_features:
		# emotion_nodes = defaultdict(set)
		# for key, value in tqdm(senticnet5.senticnet.items(), desc='initializing senticnet emotions...'):
		# 	for emotion in [value[4], value[5]]:
		# 		emotion_nodes[emotion].add(key)
		if create_edge_features:
			nlp = spacy.load("en_core_web_sm")

		if documents is not None:
			misinfo_parse = {}
			for doc in tqdm(documents, desc='loading documents...'):
				tweet_id = doc['id_str']
				tweet_text = doc['full_text'].strip().replace('\r', ' ').replace('\n', ' ')
				if create_edge_features:
					tweet_parse = [get_token_features(x) for x in nlp(tweet_text)]
				for m in doc['misconceptions']:
					m_label = None
					if labeled:
						m_label = label_text_to_id(m['label'])
					tweet_text = filter_tweet_text(tweet_text)
					m_text = m['misconception_text'].strip().replace('\r', ' ').replace('\n', ' ')
					m_id = m['misconception_id']
					if add_mis_info:
						m_info = mis_info[m_id]
						m_type = m_info['type']
						m_concern = m_info['concern']
						m_theme = m_info['theme']
						m_threats = m_info['threats']
						m_text = f'{m_text} (type: {m_type}, concern: {m_concern}, theme: {m_theme}, threats: {m_threats})'

					token_data = tokenizer(
						m_text,
						tweet_text
					)
					ex = {
						'id': tweet_id,
						'text': tweet_text,
						'question_id': m_id,
						'query': m_text,
						'label': m_label,
						'scores': {},
						'edges': {},
						'input_ids': token_data['input_ids'],
						'token_type_ids': token_data['token_type_ids'],
						'attention_mask': token_data['attention_mask'],
					}
					if tweet_id in sentiment_preds:
						ex['scores']['sentiment'] = format_predictions(sentiment_preds[tweet_id], sentiment_labels)

					if tweet_id in emotion_preds:
						ex['scores']['emotion'] = format_predictions(emotion_preds[tweet_id], emotion_labels)

					if tweet_id in irony_preds:
						ex['scores']['irony'] = format_predictions(irony_preds[tweet_id], irony_labels)

					# TODO add back in when tokenization is stable
					if create_edge_features:
						if m_id in misinfo_parse:
							m_parse = misinfo_parse[m_id]
						else:
							m_parse = [get_token_features(x) for x in nlp(m_text)]
							misinfo_parse[m_id] = m_parse
						edges = create_edges(
							m_parse,
							tweet_parse,
							token_data,
							num_semantic_hops,
							num_emotion_hops,
							num_lexical_hops,
							m_offset=len(m_text),
							tokenizer=tokenizer
						)
						ex['edges'] = edges

						import sys
						np.set_printoptions(threshold=sys.maxsize)
						print('semantic_edges')
						for i in range(len(ex['edges']['semantic'])):
							adj_list = []
							for j in range(len(ex['edges']['semantic'])):
								if ex['edges']['semantic'][i, j] > 0:
									adj_list.append(j)
							if len(adj_list) > 0:
								print(f'{i} -> {adj_list}')
						input()
						print('emotion_edges')
						for i in range(len(ex['edges']['emotion'])):
							adj_list = []
							for j in range(len(ex['edges']['emotion'])):
								if ex['edges']['emotion'][i, j] > 0:
									adj_list.append(j)
							if len(adj_list) > 0:
								print(f'{i} -> {adj_list}')
						input()
						print('lexical_edges')
						for i in range(len(ex['edges']['lexical'])):
							adj_list = []
							for j in range(len(ex['edges']['lexical'])):
								if ex['edges']['lexical'][i, j] > 0:
									adj_list.append(j)
							if len(adj_list) > 0:
								print(f'{i} -> {adj_list}')
						input()
						print()

					self.num_labels[m_label] += 1
					self.examples.append(ex)

		if hera_documents is not None:
			for doc in tqdm(hera_documents, desc='loading HERA documents...'):
				tweet_id = doc['id_str']
				tweet_text = doc['full_text']
				tweet_text = filter_tweet_text(tweet_text)

				m = doc['misinformation']
				info = doc['info']
				m_text = info['topic']['title']
				m_id = m['index']
				source = info['source'].lower()
				label_name = m['label_name'].lower()
				if not keep_real and label_name == 'real':
					continue
				if 'predicted_label' in m:
					m_label = m['predicted_label']
				else:
					m_label = hera_label_to_id(source, label_name)

				# TODO annotate news / claims for this:
				if add_mis_info:
					raise NotImplementedError('Not annotated yet!')

				token_data = tokenizer(
					m_text,
					tweet_text
				)
				ex = {
					'id': tweet_id,
					'text': tweet_text,
					'question_id': m_id,
					'query': m_text,
					'label': m_label,
					'scores': {},
					'edges': {},
					'input_ids': token_data['input_ids'],
					'token_type_ids': token_data['token_type_ids'],
					'attention_mask': token_data['attention_mask'],
				}
				if tweet_id in coaid_sentiment_preds:
					ex['scores']['sentiment'] = format_predictions(coaid_sentiment_preds[tweet_id], sentiment_labels)

				if tweet_id in coaid_emotion_preds:
					ex['scores']['emotion'] = format_predictions(coaid_emotion_preds[tweet_id], emotion_labels)

				if tweet_id in coaid_irony_preds:
					ex['scores']['irony'] = format_predictions(coaid_irony_preds[tweet_id], irony_labels)

				self.num_labels[m_label] += 1
				self.examples.append(ex)
		self.num_examples = len(self.examples)
		random.shuffle(self.examples)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class QARetrievalPredictionDataset(Dataset):
	def __init__(self, documents, misconceptions, train_documents):
		self.examples = []
		self.num_docs = len(documents)
		self.train_documents = train_documents
		skip_sets = defaultdict(set)
		for t_doc in train_documents:
			for m in t_doc['misconceptions']:
				skip_sets[t_doc['id_str']].add(m['misconception_id'])

		for doc in documents:
			for m_id, m in misconceptions.items():
				# these represent examples on which this model was trained, therefore do not produce predictions
				if m_id in skip_sets[doc['id_str']]:
					continue
				ex = {
					'id': doc['id_str'],
					'text': doc['full_text'],
					'question_id': m_id,
					'query': m['misconception_question'],
				}
				self.examples.append(ex)

		self.num_examples = len(self.examples)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example
