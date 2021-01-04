
from transformers import BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl
import torch.distributed as dist
import os
import math
import logging

from gcn_layers import GraphConvolution, GraphAttention, TransformerGraphAttention


class BaseCovidTwitterStanceModel(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total,
			torch_cache_dir=None, predict_mode=False, predict_path=None, load_pretrained=False
	):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.predict_mode = predict_mode
		self.predict_path = predict_path
		self.load_pretrained = load_pretrained
		if self.predict_mode or self.load_pretrained:
			# no need to load pre-trained weights since we will be loading whole model's
			# fine-tuned weights from checkpoint.
			self.config = BertConfig.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.bert = BertModel(self.config)
		else:
			self.bert = BertModel.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.config = self.bert.config

		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.score_func = torch.nn.Softmax(dim=-1)
		self.save_hyperparameters()

	def _forward_step(self, batch, batch_nb):
		logits = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			batch=batch
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
			device_id = get_device_id()
			ex_dict = {
				'id': batch['id'],
				'question_id': batch['question_id'],
			}
			for i in range(logits.shape[-1]):
				ex_dict[f'{i}_score'] = logits[:, i].tolist()
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
		pos_predictions = (pos_probs.max(dim=-1)[1] + 1)
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
			i_tp = (predictions.eq(i).float() * labels.eq(i).float()).sum()
			# label is not positive and predicted positive
			i_fp = (predictions.eq(i).float() * labels.ne(i).float()).sum()
			# label is positive and predicted negative
			i_fn = (predictions.ne(i).float() * labels.eq(i).float()).sum()
			i_precision = i_tp / (torch.clamp(i_tp + i_fp, 1.0))
			i_recall = i_tp / torch.clamp(i_tp + i_fn, 1.0)

			i_f1 = 2.0 * (i_precision * i_recall) / (torch.clamp(i_precision + i_recall, 1.0))
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


class CovidTwitterStanceModel(BaseCovidTwitterStanceModel):
	def __init__(
			self, classifier_feature_sizes=None, sentiment_labels=None, emotion_labels=None, irony_labels=None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if classifier_feature_sizes is None:
			classifier_feature_sizes = 0
			# classifier_feature_sizes = self.config.hidden_size
		# 39
		classifier_input_size = self.config.hidden_size
		# before 39
		# classifier_input_size = classifier_feature_sizes

		self.has_sentiment = False
		self.sentiment_labels = sentiment_labels
		if sentiment_labels is not None:
			logging.info('Using sentiment data...')
			self.sentiment_embeddings = nn.Embedding(
				num_embeddings=len(sentiment_labels),
				embedding_dim=self.config.hidden_size
			)
			self.sentiment_pooling = CrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=self.config.hidden_dropout_prob
			)
			classifier_input_size += self.config.hidden_size
			self.has_sentiment = True

		self.has_emotion = False
		self.emotion_labels = emotion_labels
		if emotion_labels is not None:
			logging.info('Using emotion data...')
			self.emotion_embeddings = nn.Embedding(
				num_embeddings=len(emotion_labels),
				embedding_dim=self.config.hidden_size
			)
			self.emotion_pooling = CrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=self.config.hidden_dropout_prob
			)
			classifier_input_size += self.config.hidden_size
			self.has_emotion = True

		self.has_irony = False
		self.irony_labels = irony_labels
		if irony_labels is not None:
			logging.info('Using irony data...')
			self.irony_embeddings = nn.Embedding(
				num_embeddings=len(irony_labels),
				embedding_dim=self.config.hidden_size
			)
			self.irony_pooling = CrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=self.config.hidden_dropout_prob
			)
			classifier_input_size += self.config.hidden_size
			self.has_irony = True
		classifier_input_size += classifier_feature_sizes
		# TODO consider different representation / pooling for each stance type
		self.dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		self.classifier = nn.Linear(
			classifier_input_size,
			3
		)

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		cls_output = contextualized_embeddings[:, 0]

		classifier_inputs = [cls_output]
		if self.has_sentiment:
			s_embeddings = self.sentiment_embeddings(batch['sentiment_ids'])
			# [bsize, emb_size]
			s_outputs = self.sentiment_pooling(
				hidden_states=contextualized_embeddings,
				queries=s_embeddings,
				query_probs=batch['sentiment_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(s_outputs)
		if self.has_emotion:
			e_embeddings = self.emotion_embeddings(batch['emotion_ids'])
			# [bsize, emb_size]
			e_outputs = self.emotion_pooling(
				hidden_states=contextualized_embeddings,
				queries=e_embeddings,
				query_probs=batch['emotion_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(e_outputs)

		if self.has_irony:
			i_embeddings = self.irony_embeddings(batch['irony_ids'])
			# [bsize, emb_size]
			i_outputs = self.irony_pooling(
				hidden_states=contextualized_embeddings,
				queries=i_embeddings,
				query_probs=batch['irony_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(i_outputs)

		classifier_inputs = torch.cat(classifier_inputs, dim=-1)
		classifier_inputs = self.dropout(classifier_inputs)
		logits = self.classifier(classifier_inputs)
		return logits


class CovidTwitterGCNStanceModel(CovidTwitterStanceModel):
	def __init__(self, freeze_lm, gcn_size, gcn_type, graph_names, *args, **kwargs):
		super().__init__(classifier_feature_sizes=gcn_size * len(graph_names), *args, **kwargs)
		self.freeze_lm = freeze_lm
		self.graph_names = graph_names
		if self.config.hidden_size != gcn_size:
			self.gcn_projs = nn.ModuleDict(
				{
					f'{graph_name}_proj': nn.Linear(self.config.hidden_size, gcn_size) for graph_name in self.graph_names
				}
			)

		self.gcn_type = gcn_type.lower()
		# TODO semantic graph, emotion graph, dependency parse graph all possible
		# TODO consider modeling together in same graph or in different graphs
		# TODO different edge types?
		# TODO implement multiple layers?
		if self.gcn_type == 'convolution':
			self.gcns = nn.ModuleDict(
				{
					f'{graph_name}_gcn': GraphConvolution(
						in_features=gcn_size,
						out_features=gcn_size
					) for graph_name in self.graph_names
				}
			)

		elif self.gcn_type == 'attention':
			self.gcns = nn.ModuleDict(
				{
					f'{graph_name}_gcn': GraphAttention(
						in_features=gcn_size,
						out_features=gcn_size,
						dropout=self.config.hidden_dropout_prob,
						alpha=0.2,
						concat=True
					) for graph_name in self.graph_names
				}
			)
		elif self.gcn_type == 'transformer':
			self.gcns = nn.ModuleDict(
				{
					f'{graph_name}_gcn': TransformerGraphAttention(
						in_features=gcn_size,
						out_features=gcn_size,
						dropout_prob=self.config.hidden_dropout_prob,
						activation=True
					) for graph_name in self.graph_names
				}
			)
		else:
			raise ValueError(f'Unknown gcn_type: {self.gcn_type}')

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		if self.freeze_lm:
			with torch.no_grad():
				outputs = self.bert(
					input_ids,
					attention_mask=attention_mask,
					token_type_ids=token_type_ids
				)
		else:
			outputs = self.bert(
				input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
		embedding_output = outputs[0]
		cls_output = embedding_output[:, 0]
		# TODO alternative to cls, consider stance embedding attention pooling + separate classification representations
		classifier_inputs = [cls_output]
		if self.has_sentiment:
			s_embeddings = self.sentiment_embeddings(batch['sentiment_ids'])
			# [bsize, emb_size]
			s_outputs = self.sentiment_pooling(
				hidden_states=embedding_output,
				queries=s_embeddings,
				query_probs=batch['sentiment_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(s_outputs)
		if self.has_emotion:
			e_embeddings = self.emotion_embeddings(batch['emotion_ids'])
			# [bsize, emb_size]
			e_outputs = self.emotion_pooling(
				hidden_states=embedding_output,
				queries=e_embeddings,
				query_probs=batch['emotion_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(e_outputs)

		if self.has_irony:
			i_embeddings = self.irony_embeddings(batch['irony_ids'])
			# [bsize, emb_size]
			i_outputs = self.irony_pooling(
				hidden_states=embedding_output,
				queries=i_embeddings,
				query_probs=batch['irony_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(i_outputs)

		for graph_name in self.graph_names:
			if self.gcn_projs is not None:
				gcn_ctx_input = self.gcn_projs[f'{graph_name}_proj'](embedding_output)
			else:
				gcn_ctx_input = embedding_output
			gcn_edges = batch[f'{graph_name}_edges']
			gcn_output = self.gcns[f'{graph_name}_gcn'](gcn_ctx_input, gcn_edges)
			# [bsize, seq_len, hidden_size] -> [bsize, hidden_size]
			# TODO better GCN pooling
			gcn_output_pool = gcn_output.mean(dim=-2)
			classifier_inputs.append(gcn_output_pool)

		classifier_inputs = torch.cat(classifier_inputs, dim=-1)
		classifier_inputs = self.dropout(classifier_inputs)
		logits = self.classifier(classifier_inputs)
		return logits


class CovidTwitterEmbeddingStanceModel(BaseCovidTwitterStanceModel):
	def __init__(
			self, classifier_feature_sizes=None, embedding_size=100,
			sentiment_labels=None, emotion_labels=None, irony_labels=None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if classifier_feature_sizes is None:
			classifier_feature_sizes = 0
		classifier_input_size = self.config.hidden_size
		self.has_sentiment = False
		self.sentiment_labels = sentiment_labels
		if sentiment_labels is not None:
			logging.info('Using sentiment data...')
			self.sentiment_embeddings = nn.Embedding(
				num_embeddings=len(sentiment_labels),
				embedding_dim=embedding_size
			)
			classifier_input_size += embedding_size
			self.has_sentiment = True

		self.has_emotion = False
		self.emotion_labels = emotion_labels
		if emotion_labels is not None:
			logging.info('Using emotion data...')
			self.emotion_embeddings = nn.Embedding(
				num_embeddings=len(emotion_labels),
				embedding_dim=embedding_size
			)
			classifier_input_size += embedding_size
			self.has_emotion = True

		self.has_irony = False
		self.irony_labels = irony_labels
		if irony_labels is not None:
			logging.info('Using irony data...')
			self.irony_embeddings = nn.Embedding(
				num_embeddings=len(irony_labels),
				embedding_dim=embedding_size
			)
			classifier_input_size += embedding_size
			self.has_irony = True
		classifier_input_size += classifier_feature_sizes
		# TODO consider different representation / pooling for each stance type
		self.dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		self.classifier = nn.Linear(
			classifier_input_size,
			3
		)

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		cls_output = contextualized_embeddings[:, 0]

		classifier_inputs = [cls_output]
		if self.has_sentiment:
			s_emb_idx = torch.argmax(batch['sentiment_scores'], dim=-1)
			s_outputs = self.sentiment_embeddings(s_emb_idx)
			classifier_inputs.append(s_outputs)
		if self.has_emotion:
			e_emb_idx = torch.argmax(batch['emotion_scores'], dim=-1)
			e_outputs = self.emotion_embeddings(e_emb_idx)
			classifier_inputs.append(e_outputs)
		if self.has_irony:
			i_emb_idx = torch.argmax(batch['irony_scores'], dim=-1)
			i_outputs = self.irony_embeddings(i_emb_idx)
			classifier_inputs.append(i_outputs)

		classifier_inputs = torch.cat(classifier_inputs, dim=-1)
		classifier_inputs = self.dropout(classifier_inputs)
		logits = self.classifier(classifier_inputs)
		return logits


class CovidTwitterPoolingStanceModel(BaseCovidTwitterStanceModel):
	def __init__(
			self, classifier_feature_sizes=None, sentiment_labels=None, emotion_labels=None, irony_labels=None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if classifier_feature_sizes is None:
			classifier_feature_sizes = 0
		classifier_input_size = self.config.hidden_size

		self.has_sentiment = False
		self.sentiment_labels = sentiment_labels
		if sentiment_labels is not None:
			logging.info('Using sentiment data...')
			self.sentiment_embeddings = nn.Embedding(
				num_embeddings=len(sentiment_labels),
				embedding_dim=self.config.hidden_size
			)
			self.sentiment_pooling = CrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=self.config.hidden_dropout_prob
			)
			classifier_input_size += self.config.hidden_size
			self.has_sentiment = True

		self.has_emotion = False
		self.emotion_labels = emotion_labels
		if emotion_labels is not None:
			logging.info('Using emotion data...')
			self.emotion_embeddings = nn.Embedding(
				num_embeddings=len(emotion_labels),
				embedding_dim=self.config.hidden_size
			)
			self.emotion_pooling = CrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=self.config.hidden_dropout_prob
			)
			classifier_input_size += self.config.hidden_size
			self.has_emotion = True

		self.has_irony = False
		self.irony_labels = irony_labels
		if irony_labels is not None:
			logging.info('Using irony data...')
			self.irony_embeddings = nn.Embedding(
				num_embeddings=len(irony_labels),
				embedding_dim=self.config.hidden_size
			)
			self.irony_pooling = CrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=self.config.hidden_dropout_prob
			)
			classifier_input_size += self.config.hidden_size
			self.has_irony = True
		classifier_input_size += classifier_feature_sizes

		self.stance_embeddings = nn.Embedding(
			num_embeddings=3,
			embedding_dim=self.config.hidden_size
		)

		self.stance_pooling = AttentionPooling(
			hidden_size=self.config.hidden_size,
			dropout_prob=self.config.hidden_dropout_prob
		)

		self.classifiers = nn.ModuleDict(
			{
				f'{stance_idx}_classifier': nn.Linear(
					classifier_input_size,
					1
				)
				for stance_idx in range(3)
			}
		)

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]

		classifier_inputs = []
		if self.has_sentiment:
			s_embeddings = self.sentiment_embeddings(batch['sentiment_ids'])
			# [bsize, emb_size]
			s_outputs = self.sentiment_pooling(
				hidden_states=contextualized_embeddings,
				queries=s_embeddings,
				query_probs=batch['sentiment_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(s_outputs)
		if self.has_emotion:
			e_embeddings = self.emotion_embeddings(batch['emotion_ids'])
			# [bsize, emb_size]
			e_outputs = self.emotion_pooling(
				hidden_states=contextualized_embeddings,
				queries=e_embeddings,
				query_probs=batch['emotion_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(e_outputs)

		if self.has_irony:
			i_embeddings = self.irony_embeddings(batch['irony_ids'])
			# [bsize, emb_size]
			i_outputs = self.irony_pooling(
				hidden_states=contextualized_embeddings,
				queries=i_embeddings,
				query_probs=batch['irony_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(i_outputs)

		# [bsize, 3, emb_size]
		stance_embs = self.stance_embeddings(batch['stance_ids'])
		# [bsize, 3, emb_size]
		stance_pool_embs = self.stance_pooling(
				hidden_states=contextualized_embeddings,
				queries=stance_embs,
				attention_mask=attention_mask
		)
		logits = []
		for stance_idx in range(3):
			# [bsize, emb_size]
			stance_idx_emb = stance_pool_embs[:, stance_idx]
			stance_idx_classifier_input = torch.cat(classifier_inputs + [stance_idx_emb], dim=-1)
			# [bsize, 1]
			stance_idx_logits = self.classifiers[f'{stance_idx}_classifier'](stance_idx_classifier_input)
			logits.append(stance_idx_logits)
		logits = torch.cat(logits, dim=-1)
		return logits


class CovidTwitterReducedPoolingStanceModel(BaseCovidTwitterStanceModel):
	def __init__(
			self, classifier_feature_sizes=None, sentiment_labels=None, emotion_labels=None, irony_labels=None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if classifier_feature_sizes is None:
			classifier_feature_sizes = 0
			# classifier_feature_sizes = self.config.hidden_size
		# 39
		classifier_input_size = self.config.hidden_size
		# before 39
		# classifier_input_size = classifier_feature_sizes

		self.has_sentiment = False
		self.sentiment_labels = sentiment_labels
		if sentiment_labels is not None:
			logging.info('Using sentiment data...')
			self.sentiment_embeddings = nn.Embedding(
				num_embeddings=len(sentiment_labels),
				embedding_dim=self.config.hidden_size
			)
			self.sentiment_pooling = ReducedCrossAttentionPooling(
				dropout_prob=self.config.hidden_dropout_prob,
				hidden_size=self.config.hidden_size
			)
			classifier_input_size += self.config.hidden_size
			self.has_sentiment = True

		self.has_emotion = False
		self.emotion_labels = emotion_labels
		if emotion_labels is not None:
			logging.info('Using emotion data...')
			self.emotion_embeddings = nn.Embedding(
				num_embeddings=len(emotion_labels),
				embedding_dim=self.config.hidden_size
			)
			self.emotion_pooling = ReducedCrossAttentionPooling(
				dropout_prob=self.config.hidden_dropout_prob,
				hidden_size=self.config.hidden_size
			)
			classifier_input_size += self.config.hidden_size
			self.has_emotion = True

		self.has_irony = False
		self.irony_labels = irony_labels
		if irony_labels is not None:
			logging.info('Using irony data...')
			self.irony_embeddings = nn.Embedding(
				num_embeddings=len(irony_labels),
				embedding_dim=self.config.hidden_size
			)
			self.irony_pooling = ReducedCrossAttentionPooling(
				dropout_prob=self.config.hidden_dropout_prob,
				hidden_size=self.config.hidden_size
			)
			classifier_input_size += self.config.hidden_size
			self.has_irony = True
		classifier_input_size += classifier_feature_sizes
		# TODO consider different representation / pooling for each stance type
		self.dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		self.classifier = nn.Linear(
			classifier_input_size,
			3
		)

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		cls_output = contextualized_embeddings[:, 0]

		classifier_inputs = [cls_output]
		if self.has_sentiment:
			s_embeddings = self.sentiment_embeddings(batch['sentiment_ids'])
			# [bsize, emb_size]
			s_outputs = self.sentiment_pooling(
				hidden_states=contextualized_embeddings,
				queries=s_embeddings,
				query_probs=batch['sentiment_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(s_outputs)
		if self.has_emotion:
			e_embeddings = self.emotion_embeddings(batch['emotion_ids'])
			# [bsize, emb_size]
			e_outputs = self.emotion_pooling(
				hidden_states=contextualized_embeddings,
				queries=e_embeddings,
				query_probs=batch['emotion_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(e_outputs)

		if self.has_irony:
			i_embeddings = self.irony_embeddings(batch['irony_ids'])
			# [bsize, emb_size]
			i_outputs = self.irony_pooling(
				hidden_states=contextualized_embeddings,
				queries=i_embeddings,
				query_probs=batch['irony_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(i_outputs)

		classifier_inputs = torch.cat(classifier_inputs, dim=-1)
		classifier_inputs = self.dropout(classifier_inputs)
		logits = self.classifier(classifier_inputs)
		return logits


class CovidTwitterReducedStancePoolingStanceModel(BaseCovidTwitterStanceModel):
	def __init__(
			self, classifier_feature_sizes=None, sentiment_labels=None, emotion_labels=None, irony_labels=None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if classifier_feature_sizes is None:
			classifier_feature_sizes = 0
		classifier_input_size = self.config.hidden_size

		self.has_sentiment = False
		self.sentiment_labels = sentiment_labels
		if sentiment_labels is not None:
			logging.info('Using sentiment data...')
			self.sentiment_embeddings = nn.Embedding(
				num_embeddings=len(sentiment_labels),
				embedding_dim=self.config.hidden_size
			)
			self.sentiment_pooling = ReducedCrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=0.0
			)
			classifier_input_size += self.config.hidden_size
			self.has_sentiment = True

		self.has_emotion = False
		self.emotion_labels = emotion_labels
		if emotion_labels is not None:
			logging.info('Using emotion data...')
			self.emotion_embeddings = nn.Embedding(
				num_embeddings=len(emotion_labels),
				embedding_dim=self.config.hidden_size
			)
			self.emotion_pooling = ReducedCrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=0.0
			)
			classifier_input_size += self.config.hidden_size
			self.has_emotion = True

		self.has_irony = False
		self.irony_labels = irony_labels
		if irony_labels is not None:
			logging.info('Using irony data...')
			self.irony_embeddings = nn.Embedding(
				num_embeddings=len(irony_labels),
				embedding_dim=self.config.hidden_size
			)
			self.irony_pooling = ReducedCrossAttentionPooling(
				hidden_size=self.config.hidden_size,
				dropout_prob=0.0
			)
			classifier_input_size += self.config.hidden_size
			self.has_irony = True
		classifier_input_size += classifier_feature_sizes

		self.stance_embeddings = nn.Embedding(
			num_embeddings=3,
			embedding_dim=self.config.hidden_size
		)

		self.stance_pooling = ReducedAttentionPooling(
			hidden_size=self.config.hidden_size,
			dropout_prob=0.0
		)

		self.classifiers = nn.ModuleDict(
			{
				f'{stance_idx}_classifier': nn.Linear(
					classifier_input_size,
					1
				)
				for stance_idx in range(3)
			}
		)

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]

		classifier_inputs = []
		if self.has_sentiment:
			s_embeddings = self.sentiment_embeddings(batch['sentiment_ids'])
			# [bsize, emb_size]
			s_outputs = self.sentiment_pooling(
				hidden_states=contextualized_embeddings,
				queries=s_embeddings,
				query_probs=batch['sentiment_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(s_outputs)
		if self.has_emotion:
			e_embeddings = self.emotion_embeddings(batch['emotion_ids'])
			# [bsize, emb_size]
			e_outputs = self.emotion_pooling(
				hidden_states=contextualized_embeddings,
				queries=e_embeddings,
				query_probs=batch['emotion_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(e_outputs)

		if self.has_irony:
			i_embeddings = self.irony_embeddings(batch['irony_ids'])
			# [bsize, emb_size]
			i_outputs = self.irony_pooling(
				hidden_states=contextualized_embeddings,
				queries=i_embeddings,
				query_probs=batch['irony_scores'],
				attention_mask=attention_mask
			)
			classifier_inputs.append(i_outputs)

		# [bsize, 3, emb_size]
		stance_embs = self.stance_embeddings(batch['stance_ids'])
		# [bsize, 3, emb_size]
		stance_pool_embs = self.stance_pooling(
				hidden_states=contextualized_embeddings,
				queries=stance_embs,
				attention_mask=attention_mask
		)
		logits = []
		for stance_idx in range(3):
			# [bsize, emb_size]
			stance_idx_emb = stance_pool_embs[:, stance_idx]
			stance_idx_classifier_input = torch.cat(classifier_inputs + [stance_idx_emb], dim=-1)
			# [bsize, 1]
			stance_idx_logits = self.classifiers[f'{stance_idx}_classifier'](stance_idx_classifier_input)
			logits.append(stance_idx_logits)
		logits = torch.cat(logits, dim=-1)
		return logits


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id


class CrossAttentionPooling(nn.Module):
	def __init__(self, hidden_size, dropout_prob):
		super().__init__()
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob
		self.query = nn.Linear(hidden_size, hidden_size)
		self.key = nn.Linear(hidden_size, hidden_size)
		self.value = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout_prob)
		self.normalizer = nn.Softmax(dim=-1)

	def forward(self, hidden_states, queries, query_probs, attention_mask=None):
		# [bsize, seq_len, hidden_size]
		# hidden_states
		# [bsize, num_queries, hidden_size]
		# queries
		# [bsize, num_queries]
		# query_probs
		# [bsize, seq_len]]
		# attention_mask
		# TODO consider multi-head
		if attention_mask is None:
			attention_mask = torch.ones(hidden_states.shape[:-1])
		attention_mask = attention_mask.float()
		attention_mask = attention_mask.unsqueeze(dim=1)
		attention_mask = (1.0 - attention_mask) * -10000.0
		# [bsize, num_queries, hidden_size]
		q = self.query(queries)
		# [bsize, seq_len, hidden_size]
		k = self.key(hidden_states)
		# [bsize, seq_len, hidden_size]
		v = self.value(hidden_states)
		# [bsize, num_queries, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, num_queries, seq_len]
		attention_scores = torch.matmul(q, k.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.hidden_size)
		attention_scores = attention_scores + attention_mask

		# [bsize, num_queries, seq_len]
		# Normalize the attention scores to probabilities.
		attention_probs = self.normalizer(attention_scores)
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		# [bsize, num_queries, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, num_queries, hidden_size]
		context_layer = torch.matmul(attention_probs, v)
		# [bsize, 1, num_queries] x [bsize, num_queries, hidden_size] -> [bsize, 1, hidden_size]
		final_layer = torch.matmul(query_probs.unsqueeze(1), context_layer)
		# [bsize, hidden_size]
		final_layer = final_layer.squeeze(dim=1)
		return final_layer


class ReducedCrossAttentionPooling(nn.Module):
	def __init__(self, dropout_prob, hidden_size):
		super().__init__()
		self.dropout_prob = dropout_prob
		self.dropout = nn.Dropout(dropout_prob)
		self.normalizer = nn.Softmax(dim=-1)
		self.hidden_size = hidden_size

	def forward(self, hidden_states, queries, query_probs, attention_mask=None):
		# [bsize, seq_len, hidden_size]
		# hidden_states
		# [bsize, num_queries, hidden_size]
		# queries
		# [bsize, num_queries]
		# query_probs
		# [bsize, seq_len]]
		# attention_mask
		# TODO consider multi-head
		if attention_mask is None:
			attention_mask = torch.ones(hidden_states.shape[:-1])
		attention_mask = attention_mask.float()
		attention_mask = attention_mask.unsqueeze(dim=1)
		attention_mask = (1.0 - attention_mask) * -10000.0
		# [bsize, num_queries, hidden_size]
		q = queries
		# [bsize, seq_len, hidden_size]
		k = hidden_states
		# [bsize, seq_len, hidden_size]
		v = hidden_states
		# [bsize, num_queries, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, num_queries, seq_len]
		attention_scores = torch.matmul(q, k.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.hidden_size)
		attention_scores = attention_scores + attention_mask

		# [bsize, num_queries, seq_len]
		# Normalize the attention scores to probabilities.
		attention_probs = self.normalizer(attention_scores)
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		# [bsize, num_queries, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, num_queries, hidden_size]
		context_layer = torch.matmul(attention_probs, v)
		# [bsize, 1, num_queries] x [bsize, num_queries, hidden_size] -> [bsize, 1, hidden_size]
		final_layer = torch.matmul(query_probs.unsqueeze(1), context_layer)
		# [bsize, hidden_size]
		final_layer = final_layer.squeeze(dim=1)
		return final_layer


class AttentionPooling(nn.Module):
	def __init__(self, hidden_size, dropout_prob):
		super().__init__()
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob
		self.query = nn.Linear(hidden_size, hidden_size)
		self.key = nn.Linear(hidden_size, hidden_size)
		self.value = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout_prob)
		self.normalizer = nn.Softmax(dim=-1)

	def forward(self, hidden_states, queries, attention_mask=None):
		# [bsize, seq_len, hidden_size]
		# hidden_states
		# [bsize, num_queries, hidden_size]
		# queries
		# [bsize, num_queries]
		# query_probs
		# [bsize, seq_len]]
		# attention_mask
		# TODO consider multi-head
		if attention_mask is None:
			attention_mask = torch.ones(hidden_states.shape[:-1])
		attention_mask = attention_mask.float()
		attention_mask = attention_mask.unsqueeze(dim=1)
		attention_mask = (1.0 - attention_mask) * -10000.0
		# [bsize, num_queries, hidden_size]
		q = self.query(queries)
		# [bsize, seq_len, hidden_size]
		k = self.key(hidden_states)
		# [bsize, seq_len, hidden_size]
		v = self.value(hidden_states)
		# [bsize, num_queries, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, num_queries, seq_len]
		attention_scores = torch.matmul(q, k.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.hidden_size)
		attention_scores = attention_scores + attention_mask

		# [bsize, num_queries, seq_len]
		# Normalize the attention scores to probabilities.
		attention_probs = self.normalizer(attention_scores)
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		# [bsize, num_queries, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, num_queries, hidden_size]
		context_layer = torch.matmul(attention_probs, v)
		return context_layer


class ReducedAttentionPooling(nn.Module):
	def __init__(self, hidden_size, dropout_prob):
		super().__init__()
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob
		self.dropout = nn.Dropout(dropout_prob)
		self.normalizer = nn.Softmax(dim=-1)

	def forward(self, hidden_states, queries, attention_mask=None):
		# [bsize, seq_len, hidden_size]
		# hidden_states
		# [bsize, num_queries, hidden_size]
		# queries
		# [bsize, num_queries]
		# query_probs
		# [bsize, seq_len]]
		# attention_mask
		# TODO consider multi-head
		if attention_mask is None:
			attention_mask = torch.ones(hidden_states.shape[:-1])
		attention_mask = attention_mask.float()
		attention_mask = attention_mask.unsqueeze(dim=1)
		attention_mask = (1.0 - attention_mask) * -10000.0
		# [bsize, num_queries, hidden_size]
		q = queries
		# [bsize, seq_len, hidden_size]
		k = hidden_states
		# [bsize, seq_len, hidden_size]
		v = hidden_states
		# [bsize, num_queries, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, num_queries, seq_len]
		attention_scores = torch.matmul(q, k.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.hidden_size)
		attention_scores = attention_scores + attention_mask

		# [bsize, num_queries, seq_len]
		# Normalize the attention scores to probabilities.
		attention_probs = self.normalizer(attention_scores)
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		# [bsize, num_queries, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, num_queries, hidden_size]
		context_layer = torch.matmul(attention_probs, v)
		return context_layer

