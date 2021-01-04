
import os
import json
import argparse
import logging
import pytorch_lightning as pl
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from model_utils import *
from data_utils import StanceDataset, StanceBatchCollator, QARetrievalPredictionDataset, load_dataset

import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--split_path', default=None)
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-mn', '--model_name', default='pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-bs', '--batch_size', default=16, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=128, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-eo', '--epochs', default=10, type=int)
	parser.add_argument('-cd', '--torch_cache_dir', default=None)
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')
	parser.add_argument('-csl', '--calc_seq_len', default=False, action='store_true')
	parser.add_argument('-lr', '--learning_rate', default=5e-6, type=float)
	parser.add_argument('-gpu', '--gpus', default='0')
	parser.add_argument('-lt', '--load_trained_model', default=False, action='store_true')
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-hp', '--hera_path', default=None)
	parser.add_argument('-kr', '--keep_real', default=False, action='store_true')
	parser.add_argument('-m', '--mode', default='stance')
	parser.add_argument('-mp', '--misconceptions_path', default='data/misconceptions.json')
	parser.add_argument('-mti', '--misconception_info_path', default=None)
	parser.add_argument('-sp', '--sentiment_path', default=None)
	parser.add_argument('-ep', '--emotion_path', default=None)
	parser.add_argument('-ip', '--irony_path', default=None)
	parser.add_argument('-csp', '--coaid_sentiment_path', default=None)
	parser.add_argument('-cep', '--coaid_emotion_path', default=None)
	parser.add_argument('-cip', '--coaid_irony_path', default=None)
	parser.add_argument('-hs', '--num_semantic_hops', default=3, type=int)
	parser.add_argument('-he', '--num_emotion_hops', default=1, type=int)
	parser.add_argument('-hl', '--num_lexical_hops', default=1, type=int)
	parser.add_argument('-mt', '--model_type', default='lm')
	parser.add_argument('-flm', '--freeze_lm', default=False, action='store_true')
	parser.add_argument('-ami', '--add_mis_info', default=False, action='store_true')
	parser.add_argument('-gs', '--gcn_size', default=100, type=int)
	parser.add_argument('-gd', '--gcn_depth', default=1, type=int)
	parser.add_argument('-es', '--embedding_size', default=100, type=int)
	parser.add_argument('-gt', '--gcn_type', default='convolution')
	parser.add_argument('-gns', '--graph_names', default='semantic,emotion,lexical')
	parser.add_argument('-cef', '--create_edge_features', default=False, action='store_true')
	parser.add_argument('-wf', '--weight_factor', default=1.0, type=float)

	args = parser.parse_args()

	pl.seed_everything(args.seed)

	save_directory = os.path.join(args.save_directory, args.model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	gpus = [int(x) for x in args.gpus.split(',')]
	is_distributed = len(gpus) > 1
	precision = 16 if args.use_tpus else 32
	# precision = 32
	tpu_cores = 8
	num_workers = 4
	deterministic = True

	# Also add the stream handler so that it logs on STD out as well
	# Ref: https://stackoverflow.com/a/46098711/4535284
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(save_directory, "train_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	logging.info(f'Loading tokenizer: {args.pre_model_name}')
	# tokenizer = AutoTokenizer.from_pretrained(args.pre_model_name)
	# tokenizer = BertTokenizer.from_pretrained(args.pre_model_name)
	tokenizer = BertTokenizerFast.from_pretrained(args.pre_model_name)
	train_data = None
	eval_data = None
	if args.split_path is not None:
		logging.info(f'Loading dataset: {args.split_path}')

		with open(args.split_path, 'r') as f:
			split = json.load(f)

		eval_data = split['eval']

	hera_data = None
	if args.hera_path is not None:
		logging.info(f'Loading HERA dataset: {args.hera_path}')
		with open(args.hera_path, 'r') as f:
			hera_data = json.load(f)
		logging.info(f'Loaded {len(hera_data)} HERA tweets.')

	sentiment_preds = None
	sentiment_labels = None
	if args.sentiment_path is not None:
		sentiment_labels = {
			'negative': 0,
			'neutral': 1,
			'positive': 2,
		}
		with open(args.sentiment_path, 'r') as f:
			sentiment_preds = json.load(f)
		logging.info(f'Loaded sentiment predictions.')

	emotion_preds = None
	emotion_labels = None
	if args.emotion_path is not None:
		emotion_labels = {
			'anger': 0,
			'joy': 1,
			'optimism': 2,
			'sadness': 3,
		}
		with open(args.emotion_path, 'r') as f:
			emotion_preds = json.load(f)
		logging.info(f'Loaded emotion predictions.')

	irony_preds = None
	irony_labels = None
	if args.irony_path is not None:
		irony_labels = {
			'non_irony': 0,
			'irony': 1
		}
		with open(args.irony_path, 'r') as f:
			irony_preds = json.load(f)
		logging.info(f'Loaded irony predictions.')

	coaid_sentiment_preds = None
	if args.coaid_sentiment_path is not None:
		with open(args.coaid_sentiment_path, 'r') as f:
			coaid_sentiment_preds = json.load(f)
		logging.info(f'Loaded COAID sentiment predictions.')

	coaid_emotion_preds = None
	if args.coaid_emotion_path is not None:
		with open(args.coaid_emotion_path, 'r') as f:
			coaid_emotion_preds = json.load(f)
		logging.info(f'Loaded COAID emotion predictions.')

	coaid_irony_preds = None
	if args.coaid_irony_path is not None:
		with open(args.coaid_irony_path, 'r') as f:
			coaid_irony_preds = json.load(f)
		logging.info(f'Loaded COAID irony predictions.')

	mis_info = None
	if args.misconception_info_path is not None:
		logging.info(f'Loading misconception info: {args.misconception_info_path}')
		with open(args.misconception_info_path, 'r') as f:
			mis_info = json.load(f)

		logging.info(f'Loaded misconception info.')

	if args.mode == 'stance':
		logging.info('Loading stance dataset...')
		eval_dataset_args = dict(
			documents=eval_data,
			sentiment_preds=sentiment_preds,
			emotion_preds=emotion_preds,
			irony_preds=irony_preds,
			coaid_sentiment_preds=coaid_sentiment_preds,
			coaid_emotion_preds=coaid_emotion_preds,
			coaid_irony_preds=coaid_irony_preds,
			sentiment_labels=sentiment_labels,
			emotion_labels=emotion_labels,
			irony_labels=irony_labels,
			tokenizer=tokenizer,
			create_edge_features=args.create_edge_features,
			num_semantic_hops=args.num_semantic_hops,
			num_emotion_hops=args.num_emotion_hops,
			num_lexical_hops=args.num_lexical_hops,
			mis_info=mis_info,
			add_mis_info=args.add_mis_info,
		)
		eval_dataset = load_dataset(
			args.split_path,
			eval_dataset_args,
			name='val'
		)

		# eval_dataset = StanceDataset(
		# 	documents=eval_data,
		# 	sentiment_preds=sentiment_preds,
		# 	emotion_preds=emotion_preds,
		# 	irony_preds=irony_preds,
		# 	sentiment_labels=sentiment_labels,
		# 	emotion_labels=emotion_labels,
		# 	irony_labels=irony_labels,
		# 	coaid_sentiment_preds=coaid_sentiment_preds,
		# 	coaid_emotion_preds=coaid_emotion_preds,
		# 	coaid_irony_preds=coaid_irony_preds,
		# 	tokenizer=tokenizer,
		# 	create_edge_features=args.create_edge_features,
		# 	num_semantic_hops=args.num_semantic_hops,
		# 	num_emotion_hops=args.num_emotion_hops,
		# 	num_lexical_hops=args.num_lexical_hops,
		# 	mis_info=mis_info,
		# 	add_mis_info=args.add_mis_info,
		# 	labeled=False
		# )
	elif args.mode == 'retrieval':
		logging.info('Loading retrieval dataset...')
		with open(args.misconceptions_path) as f:
			misconceptions = json.load(f)
		eval_dataset = QARetrievalPredictionDataset(
			eval_data,
			misconceptions,
			split['train']
		)
	else:
		raise ValueError(f'Unknown mode: {args.mode}')

	logging.info(f'eval={len(eval_dataset)}')

	eval_data_loader = DataLoader(
		eval_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=StanceBatchCollator(
			args.max_seq_len,
			args.use_tpus,
			labeled=False
		)
	)

	num_batches_per_step = (len(gpus) if not args.use_tpus else tpu_cores)
	updates_epoch = len(eval_dataset) // (args.batch_size * num_batches_per_step)
	updates_total = updates_epoch * args.epochs
	logging.info('Loading model...')
	model_type = args.model_type.lower()
	if model_type == 'lm':
		model = CovidTwitterStanceModel(
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			sentiment_labels=sentiment_labels,
			emotion_labels=emotion_labels,
			irony_labels=irony_labels,
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-gcn':
		model = CovidTwitterGCNStanceModel(
			freeze_lm=args.freeze_lm,
			gcn_size=args.gcn_size,
			gcn_depth=args.gcn_depth,
			gcn_type=args.gcn_type,
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			graph_names=args.graph_names.split(','),
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-gcn-expanded':
		model = CovidTwitterGCNExpandedStanceModel(
			freeze_lm=args.freeze_lm,
			gcn_size=args.gcn_size,
			gcn_depth=args.gcn_depth,
			gcn_type=args.gcn_type,
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			graph_names=args.graph_names.split(','),
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-gcn-expanded-dp':
		model = CovidTwitterGCNExpandedDPStanceModel(
			freeze_lm=args.freeze_lm,
			gcn_size=args.gcn_size,
			gcn_depth=args.gcn_depth,
			gcn_type=args.gcn_type,
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			graph_names=args.graph_names.split(','),
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-emb':
		model = CovidTwitterEmbeddingStanceModel(
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			sentiment_labels=sentiment_labels,
			emotion_labels=emotion_labels,
			irony_labels=irony_labels,
			embedding_size=args.embedding_size,
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-pool':
		model = CovidTwitterPoolingStanceModel(
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			sentiment_labels=sentiment_labels,
			emotion_labels=emotion_labels,
			irony_labels=irony_labels,
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-pool-reduced':
		model = CovidTwitterReducedStancePoolingStanceModel(
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			sentiment_labels=sentiment_labels,
			emotion_labels=emotion_labels,
			irony_labels=irony_labels,
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	elif model_type == 'lm-reduced':
		model = CovidTwitterReducedPoolingStanceModel(
			pre_model_name=args.pre_model_name,
			learning_rate=args.learning_rate,
			lr_warmup=0.1,
			updates_total=updates_total,
			weight_decay=0.0,
			weight_factor=args.weight_factor,
			sentiment_labels=sentiment_labels,
			emotion_labels=emotion_labels,
			irony_labels=irony_labels,
			torch_cache_dir=args.torch_cache_dir,
			predict_mode=True,
			predict_path=args.output_path
		)
	else:
		raise ValueError(f'Unknown model type: {model_type}')

	if args.load_trained_model:
		logging.warning(f'Loading weights from trained checkpoint: {checkpoint_path}...')
		model.load_state_dict(torch.load(checkpoint_path))

	logger = pl_loggers.TensorBoardLogger(
		save_dir=save_directory,
		flush_secs=30,
		max_queue=2
	)

	if args.use_tpus:
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			logger=logger,
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=args.epochs,
			precision=precision,
			deterministic=deterministic,
			checkpoint_callback=False,
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			logger=logger,
			gpus=gpus,
			default_root_dir=save_directory,
			max_epochs=args.epochs,
			precision=precision,
			distributed_backend=backend,
			gradient_clip_val=1.0,
			deterministic=deterministic,
			checkpoint_callback=False,
		)

	try:
		trainer.test(model, eval_data_loader)
	except Exception as e:
		logging.exception('Exception during evaluating', exc_info=e)

