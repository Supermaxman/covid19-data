
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch.distributed as dist
import os

from metrics_utils import F1Score


class QABert(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total,
			torch_cache_dir, predict_mode=False, predict_path=None):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.predict_mode = predict_mode
		self.predict_path = predict_path

		self.bert = BertModel.from_pretrained(
			pre_model_name,
			cache_dir=torch_cache_dir
		)
		self.classifier = nn.Linear(
			self.bert.config.hidden_size,
			3
		)
		self.dropout = nn.Dropout(
			p=self.bert.config.hidden_dropout_prob
		)
		self.config = self.bert.config
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.score_func = torch.nn.Softmax(dim=-1)
		self.save_hyperparameters()

	def forward(self, input_ids, attention_mask, token_type_ids):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		cls_output = contextualized_embeddings[:, 0]
		cls_output = self.dropout(cls_output)
		logits = self.classifier(cls_output)
		return logits

	def _forward_step(self, batch, batch_nb):
		logits = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		if not self.predict_mode:
			labels = batch['labels']
			loss = self._loss(
				logits,
				labels,
			)
			prediction = logits.max(dim=1)[1]
			correct_count = ((labels.eq(1)).float() * (prediction.eq(labels)).float()).sum()
			total_count = (labels.eq(1)).float().sum()
			accuracy = correct_count / total_count
			if accuracy.isnan().item():
				accuracy = torch.zeros(1, dtype=torch.float)

			return loss, logits, prediction, correct_count, total_count, accuracy
		else:
			return logits

	def training_step(self, batch, batch_nb):
		loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

		loss = loss.mean()
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result

	def test_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'test')

	def validation_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'val')

	def _eval_step(self, batch, batch_nb, name):
		if not self.predict_mode:
			loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

			result = {
				f'{name}_loss': loss.mean(),
				f'{name}_batch_loss': loss,
				f'{name}_batch_accuracy': accuracy,
				f'{name}_correct_count': correct_count,
				f'{name}_total_count': total_count,
				f'{name}_batch_logits': logits,
				f'{name}_batch_labels': batch['labels'],
				f'{name}_batch_predictions': prediction,
			}

			return result
		else:
			logits = self._forward_step(batch, batch_nb)
			logits = logits.detach()
			num_labels = logits.shape[-1]
			device_id = get_device_id()
			ex_dict = {
				'id': batch['id'],
				'question_id': batch['question_id'],
			}
			for i in range(num_labels):
				ex_dict[f'{i}_score']: logits[:, i].tolist()
			self.write_prediction_dict(
				ex_dict,
				filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
			)
			result = {
				f'{name}_id': batch['id'],
				f'{name}_question_id': batch['question_id'],
				f'{name}_logits': logits,
			}

			return result

	def _get_predictions(self, logits, threshold):
		# non-zero class probs
		# [bsize, num_labels-1]
		pos_probs = self.score_func(logits)[:, 1:]
		# filter out non-thresholded classes
		# [bsize, num_labels-1]
		pos_probs = pos_probs * ((pos_probs > threshold).float())
		# 1 if any are above threshold, 0 if none are above threshold
		# [bsize]
		pos_any_above = ((pos_probs > threshold).int().sum(dim=-1) > 0).int()
		# if none are above threshold then our prediction will be class 0, otherwise it will be
		# between the classes which have probs above the threshold
		# [bsize]
		# we add one to the class id to account for the [:, 1:] filtering of only positive probs
		pos_predictions = (pos_probs.max(dim=1)[1] + 1)
		# [bsize]
		predictions = pos_predictions * pos_any_above
		return predictions

	def _get_metrics(self, logits, labels, threshold, name):
		metrics = {}
		num_labels = logits.shape[-1]
		macro_f1 = 0.0
		macro_p = 0.0
		macro_r = 0.0
		predictions = self._get_predictions(logits, threshold)
		for i in range(num_labels):
			# label is positive and predicted positive
			i_tp = (predictions.eq(i).int() * labels.eq(i).int()).sum(dim=-1)
			# label is not positive and predicted positive
			i_fp = (predictions.eq(i).int() * labels.ne(i).int()).sum(dim=-1)
			# label is positive and predicted negative
			i_fn = (predictions.ne(i).int() * labels.eq(i).int()).sum(dim=-1)
			i_precision = i_tp / (i_tp + i_fp)
			i_recall = i_tp / (i_tp + i_fn)
			i_f1 = 2.0 * (i_precision * i_recall) / (i_precision + i_recall)
			macro_f1 += i_f1
			macro_p += i_precision
			macro_r += i_recall
			metrics[f'{name}_{i}_f1'] = i_f1
			metrics[f'{name}_{i}_p'] = i_precision
			metrics[f'{name}_{i}_r'] = i_recall

		macro_f1 = macro_f1 / num_labels
		macro_p = macro_p / num_labels
		macro_r = macro_r / num_labels
		metrics[f'{name}_macro_f1'] = macro_f1
		metrics[f'{name}_macro_p'] = macro_p
		metrics[f'{name}_macro_r'] = macro_r
		metrics[f'{name}_threshold'] = threshold
		return metrics

	def _eval_epoch_end(self, outputs, name):
		if not self.predict_mode:
			loss = torch.cat([x[f'{name}_batch_loss'] for x in outputs], dim=0).mean()
			logits = torch.cat([x[f'{name}_batch_logits'] for x in outputs], dim=0)
			labels = torch.cat([x[f'{name}_batch_labels'] for x in outputs], dim=0)

			threshold_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
			max_metric = float('-inf')
			max_metrics = {}
			for threshold in threshold_range:
				t_metrics = self._get_metrics(logits, labels, threshold, name)
				m = t_metrics[f'{name}_macro_f1']
				if m > max_metric:
					max_metric = m
					max_metrics = t_metrics

			for metric, value in max_metrics.items():
				self.log(metric, value)

			correct_count = torch.stack([x[f'{name}_correct_count'] for x in outputs], dim=0).sum()
			total_count = sum([x[f'{name}_total_count'] for x in outputs])
			accuracy = correct_count / total_count
			self.log(f'{name}_loss', loss)
			self.log(f'{name}_accuracy', accuracy)

	def validation_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'test')

	def configure_optimizers(self):
		params = self._get_optimizer_params(self.weight_decay)
		optimizer = AdamW(
			params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.lr_warmup * self.updates_total,
			num_training_steps=self.updates_total
		)
		return [optimizer], [scheduler]

	def _get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay': weight_decay},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		return optimizer_params

	def _loss(self, logits, labels):
		loss = self.criterion(
			logits,
			labels
		)

		return loss


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id
