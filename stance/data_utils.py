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

			for edge_name, edge_values in ex['edges'].items():
				# truncation to max_seq_len, still need to pad
				edge_values = edge_values[:self.max_seq_len, :self.max_seq_len]
				batch_edge_name = edge_name + '_edges'
				if batch_edge_name not in edges:
					edges[batch_edge_name] = torch.zeros([batch_size, pad_seq_len, pad_seq_len], dtype=torch.float)
				edges[batch_edge_name][ex_idx, :edge_values.shape[0], :edge_values.shape[1]] = torch.tensor(
					edge_values,
					dtype=torch.float
				)

			for score_name, score_values in ex['scores'].items():
				scores[score_name + '_scores'].append(score_values)
				other_ids[score_name + '_ids'].append([i for i in range(len(score_values))])

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


def sentic_expand(sentic_edges, expand_list):
	new_edges = set(sentic_edges)
	for edge in sentic_edges:
		edge_info = senticnet5.senticnet[edge]
		for i in expand_list:
			new_edges.add(edge_info[i])
	return new_edges


def flatten(l):
	return [item for sublist in l for item in sublist]


class StanceDataset(Dataset):
	def __init__(
			self, documents=None, hera_documents=None,
			keep_real=False,
			sentiment_preds=None, emotion_preds=None, irony_preds=None,
			sentiment_labels=None, emotion_labels=None, irony_labels=None,
			tokenizer=None,
			token_features=None, misconception_token_features=None,
			num_semantic_hops=None, num_emotion_hops=None, num_lexical_hops=None,
			labeled=True):
		self.examples = []
		self.num_labels = defaultdict(int)
		# TODO follow https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
		# TODO for graph adjacency features
		if sentiment_preds is None:
			sentiment_preds = {}
		if emotion_preds is None:
			emotion_preds = {}
		if irony_preds is None:
			irony_preds = {}
		if token_features is None:
			token_features = {}

		# emotion_nodes = defaultdict(set)
		# for key, value in tqdm(senticnet5.senticnet.items(), desc='initializing senticnet emotions...'):
		# 	for emotion in [value[4], value[5]]:
		# 		emotion_nodes[emotion].add(key)

		if documents is not None:
			for doc in tqdm(documents, desc='loading documents...'):
				for m in doc['misconceptions']:
					m_label = None
					if labeled:
						m_label = label_text_to_id(m['label'])
					tweet_id = doc['id_str']
					m_id = m['misconception_id']
					ex = {
						'id': tweet_id,
						'text': doc['full_text'],
						'question_id': m_id,
						'query': m['misconception_text'],
						'label': m_label,
						'scores': {},
						'edges': {},
					}
					if tweet_id in sentiment_preds:
						ex['scores']['sentiment'] = format_predictions(sentiment_preds[tweet_id], sentiment_labels)

					if tweet_id in emotion_preds:
						ex['scores']['emotion'] = format_predictions(emotion_preds[tweet_id], emotion_labels)

					if tweet_id in irony_preds:
						ex['scores']['irony'] = format_predictions(irony_preds[tweet_id], irony_labels)

					if tweet_id in token_features:
						input_ids = []
						token_type_ids = []
						attention_mask = []
						input_id_texts = []
						input_idx_map = defaultdict(list)
						text_map = {}
						current_token_type_id = 0
						max_input_idx = 0
						semantic_edges = {}
						emotion_edges = {}
						reverse_emotion_edges = defaultdict(set)
						lexical_edges = {}
						root_text = None
						tokens = ['[CLS]'] + misconception_token_features[m_id] + ['[SEP]'] + token_features[tweet_id] + ['[SEP]']
						for token in tokens:
							if isinstance(token, dict):
								text = token['text']
								# TODO add as features
								pos = token['pos']
								dep = token['dep']
								head = token['head']
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

								token_tokens = tokenizer(text=text, add_special_tokens=False)
								for input_id in token_tokens['input_ids']:
									input_ids.append(input_id)
									token_type_ids.append(current_token_type_id)
									attention_mask.append(1)
									input_id_texts.append(text)
									input_idx_map[text].append(max_input_idx)
									text_map[max_input_idx] = text
									max_input_idx += 1
							else:
								if token == '[CLS]' or token == '[SEP]':
									token_tokens = tokenizer(text=token, add_special_tokens=False)
									input_ids.append(token_tokens['input_ids'][0])
									token_type_ids.append(current_token_type_id)
									attention_mask.append(1)
									input_id_texts.append(token)
									input_idx_map[token].append(max_input_idx)
									text_map[max_input_idx] = token
									max_input_idx += 1
									if token == '[SEP]':
										current_token_type_id += 1

						semantic_edges['[CLS]'] = set()
						semantic_edges['[SEP]'] = set()
						emotion_edges['[CLS]'] = set()
						emotion_edges['[SEP]'] = set()
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

						semantic_adj = np.eye(max_input_idx, dtype=np.float32)
						emotion_adj = np.eye(max_input_idx, dtype=np.float32)
						lexical_adj = np.eye(max_input_idx, dtype=np.float32)
						for input_idx in range(max_input_idx):
							input_idx_text = text_map[input_idx]
							i_semantic_edges = set(flatten([input_idx_map[e_txt] for e_txt in semantic_edges[input_idx_text]]))
							for edge_idx in i_semantic_edges:
								semantic_adj[input_idx, edge_idx] = 1.0
								semantic_adj[edge_idx, input_idx] = 1.0

							i_emotion_edges = set(flatten([input_idx_map[e_txt] for e_txt in emotion_edges[input_idx_text]]))
							for edge_idx in i_emotion_edges:
								emotion_adj[input_idx, edge_idx] = 1.0
								emotion_adj[edge_idx, input_idx] = 1.0

							i_lexical_edges = set(flatten([input_idx_map[e_txt] for e_txt in lexical_edges[input_idx_text]]))
							for edge_idx in i_lexical_edges:
								lexical_adj[input_idx, edge_idx] = 1.0
								lexical_adj[edge_idx, input_idx] = 1.0

						# CLS token is connected to everything
						semantic_adj[:, 0] = 1.0
						semantic_adj[0, :] = 1.0
						# CLS token is connected to everything
						emotion_adj[:, 0] = 1.0
						emotion_adj[0, :] = 1.0
						# CLS token is connected to everything
						lexical_adj[:, 0] = 1.0
						lexical_adj[0, :] = 1.0
						ex['edges']['semantic'] = semantic_adj
						ex['edges']['emotion'] = emotion_adj
						ex['edges']['lexical'] = lexical_adj
						ex['input_ids'] = input_ids
						ex['token_type_ids'] = token_type_ids
						ex['attention_mask'] = attention_mask

						# import sys
						# np.set_printoptions(threshold=sys.maxsize)
						# print('text_map')
						# print(text_map)
						# input()
						# print('input_ids')
						# print(input_ids)
						# input()
						# print('token_type_ids')
						# print(token_type_ids)
						# input()
						# print('attention_mask')
						# print(attention_mask)
						# input()
						# print('semantic_edges')
						# print(semantic_adj)
						# input()
						# print('emotion_edges')
						# print(emotion_adj)
						# input()
						# print('lexical_edges')
						# print(lexical_adj)
						# input()
						# print()

					else:
						raise NotImplementedError()
						# tweet_tokens = tokenizer(
						# 	text=ex['query'],
						# 	text_pair=ex['text'],
						# 	add_special_tokens=True,
						# 	padding=False,
						# 	return_tensors='pt',
						# 	truncation='only_second',
						# 	max_length=max_seq_len
						# )
						# ex['tokens'] = tweet_tokens

					self.num_labels[m_label] += 1
					self.examples.append(ex)

		if hera_documents is not None:
			for doc in hera_documents:
				m = doc['misinformation']
				info = doc['info']
				question_id = info['index']
				raise NotImplementedError('Need to implement this for HERA')
				question_text = info['topic']['question']
				source = info['source'].lower()
				label_name = m['label_name'].lower()
				if not keep_real and label_name == 'real':
					continue
				if 'predicted_label' in m:
					m_label = m['predicted_label']
				else:
					m_label = hera_label_to_id(source, label_name)

				ex = {
					'id': doc['id_str'],
					'text': doc['full_text'],
					'question_id': question_id,
					'query': question_text,
					'label': m_label,
				}
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
