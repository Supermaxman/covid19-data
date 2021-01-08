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
import pickle
import zlib
from py_lex import EmoLex


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


def load_dataset(split_path, dataset_args, name):
	args_string = str(zlib.adler32(str(dataset_args).encode('utf-8')))

	cache_path = split_path + f'_{name}_{args_string}.cache'
	if os.path.exists(cache_path):
		with open(cache_path, 'rb') as f:
			dataset = pickle.load(f)
	else:
		dataset = StanceDataset(
			**dataset_args
		)
		with open(cache_path, 'wb') as f:
			pickle.dump(dataset, f)
	return dataset


class StanceBatchCollator(object):
	def __init__(
			self, max_seq_len: int, force_max_seq_len: bool,
			labeled=True):
		super().__init__()
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


def align_tokens(tokens, wpt_tokens, seq_offset=0):
	align_map = {}
	for token in tokens:
		token['wpt_idxs'] = set()
		start = token['start']
		end = token['end']
		for char_idx in range(start, end):
			sub_token_idx = wpt_tokens.char_to_token(char_idx, sequence_index=seq_offset)
			# White spaces have no token and will return None
			if sub_token_idx is not None:
				align_map[sub_token_idx] = token
				token['wpt_idxs'].add(sub_token_idx)
	return align_map


def align_token_sequences(m_tokens, t_tokens, wpt_tokens):
	# print([f'{i}:{m}' for i, m in enumerate(wpt_tokens.tokens())])
	# print([f'{m["start"]}:{m["end"]}:{m["text"]}' for m in m_tokens])
	# print([f'{m["start"]}:{m["end"]}:{m["text"]}' for m in t_tokens])
	m_align_map = align_tokens(m_tokens, wpt_tokens)
	t_align_map = align_tokens(t_tokens, wpt_tokens, seq_offset=1)
	align_map = {**m_align_map, **t_align_map}
	aligned_tokens = []
	for sub_token_idx in range(len(wpt_tokens['input_ids'])):
		if sub_token_idx not in align_map:
			# CLS, SEP, or other special token
			aligned_token = {
				'pos': 'NONE',
				'dep': 'NONE',
				'head': 'NONE',
				'sentic': None,
				'text': '[CLS]' if sub_token_idx == 0 else '[SEP]',
				'wpt_idxs': {sub_token_idx}
			}
			align_map[sub_token_idx] = aligned_token
		aligned_token = align_map[sub_token_idx]
		aligned_tokens.append(aligned_token)

	return align_map, aligned_tokens


def create_adjacency_matrix(edges, size, t_map, r_map):
	adj = np.eye(size, dtype=np.float32)
	for input_idx in range(size):
		input_idx_text = t_map[input_idx]
		i_edges = set(flatten([r_map[e_txt] for e_txt in edges[input_idx_text]]))
		for edge_idx in i_edges:
			adj[input_idx, edge_idx] = 1.0
			adj[edge_idx, input_idx] = 1.0
	return adj


def create_edges(m_tokens, t_tokens, wpt_tokens, num_semantic_hops, num_emotion_hops, num_lexical_hops, emotion_type, emolex, lex_edge_expanded):
	seq_len = len(wpt_tokens['input_ids'])
	align_map, a_tokens = align_token_sequences(m_tokens, t_tokens, wpt_tokens)

	semantic_edges = defaultdict(set)
	emotion_edges = defaultdict(set)
	reverse_emotion_edges = defaultdict(set)
	lexical_edges = defaultdict(set)
	reverse_lexical_dep_edges = defaultdict(set)
	reverse_lexical_pos_edges = defaultdict(set)
	lexical_dep_edges = defaultdict(set)
	lexical_pos_edges = defaultdict(set)
	root_text = None
	r_map = defaultdict(set)
	t_map = {}
	for token in a_tokens:
		text = token['text'].lower()
		head = token['head'].lower()
		for wpt_idx in token['wpt_idxs']:
			t_map[wpt_idx] = text
			r_map[text].add(wpt_idx)
		# TODO consider pos as features
		pos = token['pos']
		dep = token['dep']
		reverse_lexical_dep_edges[dep].add(text)
		reverse_lexical_pos_edges[pos].add(text)
		lexical_dep_edges[text].add(dep)
		lexical_pos_edges[text].add(pos)
		# will be two roots with two sequences
		if dep == 'ROOT':
			root_text = text
		sentic = token['sentic']
		if sentic is not None:
			for sem in sentic['semantics']:
				semantic_edges[text].add(sem)
			for i in range(num_semantic_hops-1):
				semantic_edges[text] = sentic_expand(semantic_edges[text], [8, 9, 10, 11, 12])
			if emotion_type == 'senticnet':
				emotion_edges[text].add(sentic['primary_mood'])
				emotion_edges[text].add(sentic['secondary_mood'])
				reverse_emotion_edges[sentic['primary_mood']].add(text)
				reverse_emotion_edges[sentic['secondary_mood']].add(text)
			elif emotion_type == 'emolex':
				for emotion in emolex.categorize_token(text):
					emotion_edges[text].add(emotion)
					reverse_emotion_edges[emotion].add(text)
			else:
				raise ValueError(f'Invalid emotion type: {emotion_type}')
			# for emotion in [sentic['primary_mood'], sentic['secondary_mood']]:
			# 	emotion_edges[text] = emotion_edges[text].union(emotion_nodes[emotion])

			# for i in range(num_emotion_hops - 1):
			# 	new_emotions = sentic_expand(emotion_edges[text], [4, 5])
			# 	for emotion in new_emotions:
			# 		emotion_edges[text] = emotion_edges[text].union(emotion_nodes[emotion])

		lexical_edges[text].add(head)

	lexical_edges['[CLS]'].add(root_text)
	lexical_edges['[SEP]'].add(root_text)

	# text -> emotion node -> other text in sentence with same emotions
	for text in emotion_edges.keys():
		emotions = emotion_edges[text]
		emotion_edges[text] = emotion_edges[text].union(
			set(flatten(reverse_emotion_edges[emotion] for emotion in emotions))
		)
	if lex_edge_expanded:
		for text in lexical_edges.keys():
			# expand lexical edges to same dependency roles
			text_deps = lexical_dep_edges[text]
			lexical_edges[text] = lexical_edges[text].union(
				set(flatten(reverse_lexical_dep_edges[dep] for dep in text_deps))
			)
			# expand lexical edges to same pos tags
			text_pos = lexical_pos_edges[text]
			lexical_edges[text] = lexical_edges[text].union(
				set(flatten(reverse_lexical_pos_edges[pos] for pos in text_pos))
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
			num_na_examples=None, emotion_type=None,
			mis_info=None, add_mis_info=False, num_hera_na_samples=0, lex_edge_expanded=False,
			labeled=True):
		self.examples = []
		self.num_labels = defaultdict(int)

		if sentiment_preds is None:
			sentiment_preds = {}
			coaid_sentiment_preds = {}
		if emotion_preds is None:
			emotion_preds = {}
			coaid_emotion_preds = {}
		if irony_preds is None:
			irony_preds = {}
			coaid_irony_preds = {}
		if create_edge_features:
			nlp = spacy.load("en_core_web_sm")

		emolex = EmoLex('data/emolex.txt')
		misinfo_parse = {}
		if documents is not None:
			for doc in tqdm(documents, desc='loading documents...'):
				tweet_id = doc['id_str']
				tweet_text = doc['full_text'].strip().replace('\r', ' ').replace('\n', ' ')
				tweet_text = filter_tweet_text(tweet_text)
				if create_edge_features:
					tweet_parse = [get_token_features(x) for x in nlp(tweet_text)]
				for m in doc['misconceptions']:
					m_label = None
					if labeled:
						m_label = label_text_to_id(m['label'])
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
							emotion_type,
							emolex,
							lex_edge_expanded,
						)
						ex['edges'] = edges

						# import sys
						# np.set_printoptions(threshold=sys.maxsize)
						# print('semantic_edges')
						# for i in range(len(ex['edges']['semantic'])):
						# 	adj_list = []
						# 	for j in range(len(ex['edges']['semantic'])):
						# 		if ex['edges']['semantic'][i, j] > 0:
						# 			adj_list.append(j)
						# 	if len(adj_list) > 0:
						# 		print(f'{i} -> {adj_list}')
						# input()
						# print('emotion_edges')
						# for i in range(len(ex['edges']['emotion'])):
						# 	adj_list = []
						# 	for j in range(len(ex['edges']['emotion'])):
						# 		if ex['edges']['emotion'][i, j] > 0:
						# 			adj_list.append(j)
						# 	if len(adj_list) > 0:
						# 		print(f'{i} -> {adj_list}')
						# input()
						# print('lexical_edges')
						# for i in range(len(ex['edges']['lexical'])):
						# 	adj_list = []
						# 	for j in range(len(ex['edges']['lexical'])):
						# 		if ex['edges']['lexical'][i, j] > 0:
						# 			adj_list.append(j)
						# 	if len(adj_list) > 0:
						# 		print(f'{i} -> {adj_list}')
						# input()
						# print()

					self.num_labels[m_label] += 1
					self.examples.append(ex)

		if hera_documents is not None:
			hera_tweet_text = {}
			hera_misinfo = {}
			hera_examples = set()
			for doc in tqdm(hera_documents, desc='loading HERA documents...'):
				tweet_id = doc['id_str']
				tweet_text = doc['full_text'].strip().replace('\r', ' ').replace('\n', ' ')
				tweet_text = filter_tweet_text(tweet_text)
				hera_tweet_text[tweet_id] = tweet_text
				m = doc['misinformation']
				info = doc['info']
				m_text = info['topic']['title'].strip().replace('\r', ' ').replace('\n', ' ')
				m_id = m['index']
				hera_examples.add((tweet_id, m_id))
				hera_misinfo[m_id] = m_text
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

				if create_edge_features:
					tweet_parse = [get_token_features(x) for x in nlp(tweet_text)]
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
						emotion_type,
						emolex,
						lex_edge_expanded,
					)
					ex['edges'] = edges

				self.num_labels[m_label] += 1
				self.examples.append(ex)

			hera_tweet_ids = list(hera_tweet_text.keys())
			hera_mis_ids = list(hera_misinfo.keys())
			num_na = 0
			progress = tqdm(total=num_hera_na_samples, desc='loading HERA samples...')

			while num_na < num_hera_na_samples:
				tweet_id = random.sample(hera_tweet_ids, k=1)[0]
				tweet_text = hera_tweet_text[tweet_id]
				m_id = random.sample(hera_mis_ids, k=1)[0]
				if (tweet_id, m_id) in hera_examples:
					continue
				hera_examples.add((tweet_id, m_id))
				m_text = hera_misinfo[m_id]
				m_label = label_text_to_id('na')
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
						emotion_type,
						emolex,
						lex_edge_expanded,
					)
					ex['edges'] = edges
				self.num_labels[m_label] += 1
				self.examples.append(ex)
				num_na += 1
				progress.update()
			progress.close()
		random.shuffle(self.examples)
		if num_na_examples is not None:
			num_na = 0
			new_examples = []
			for example in self.examples:
				if example['label'] == label_text_to_id('na'):
					if num_na >= num_na_examples:
						continue
					num_na += 1
				new_examples.append(example)
			self.examples = new_examples
		self.num_examples = len(self.examples)

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
