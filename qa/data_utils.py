
import json
import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from collections import defaultdict


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


class QABatchCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool, labeled=True):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len
		self.labeled = labeled

	def __call__(self, examples):
		ids = []
		labels = []
		question_ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
			if self.labeled:
				labels.append(ex['label'])
			question_ids.append(ex['question_id'])
			sequences.append((ex['query'], ex['text']))
		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=sequences,
			add_special_tokens=True,
			padding='max_length' if self.force_max_seq_len else 'longest',
			return_tensors='pt',
			truncation='only_second',
			max_length=self.max_seq_len
		)
		batch = {
			'id': ids,
			'question_id': question_ids,
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
		}
		if self.labeled:
			batch['labels'] = torch.tensor(labels, dtype=torch.long)

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


class QADataset(Dataset):
	def __init__(self, documents=None, hera_documents=None, keep_real=False, labeled=True):
		self.examples = []
		self.num_labels = defaultdict(int)
		if hera_documents is not None:
			for doc in hera_documents:
				m = doc['misinformation']
				info = doc['info']
				question_id = info['index']
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
		if documents is not None:
			for doc in documents:
				for m in doc['misconceptions']:
					m_label = None
					if labeled:
						m_label = label_text_to_id(m['label'])
					ex = {
						'id': doc['id_str'],
						'text': doc['full_text'],
						'question_id': m['misconception_id'],
						'query': m['misconception_question'],
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
