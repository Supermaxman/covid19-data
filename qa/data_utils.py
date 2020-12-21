
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
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		ids = []
		labels = []
		question_ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
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
			'labels': torch.tensor(labels, dtype=torch.long),
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
		}

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


class QALabeledDataset(Dataset):
	def __init__(self, documents):
		self.examples = []
		self.num_docs = len(documents)
		self.num_labels = defaultdict(int)
		for doc in documents:
			for m in doc['misconceptions']:
				ex = {
					'id': doc['id_str'],
					'text': doc['full_text'],
					'question_id': m['misconception_id'],
					'query': m['misconception_question'],
					'label': label_text_to_id(m['label']),
				}
				self.num_labels[m['label']] += 1
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


class QAPredictionDataset(Dataset):
	def __init__(self, documents):
		self.examples = []
		self.num_docs = len(documents)
		for doc in documents:
			for m in doc['misconceptions']:
				ex = {
					'id': doc['id_str'],
					'text': doc['full_text'],
					'question_id': m['misconception_id'],
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


class QAPredictionCollator(object):
	def __init__(self, tokenizer,  max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		ids = []
		question_ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
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

		return batch
