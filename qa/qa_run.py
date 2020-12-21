
import os
import json
import argparse
import logging
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from model_utils import QABert
from data_utils import QAPredictionDataset, QAPredictionCollator

import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--split_path', required=True)
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
	tokenizer = AutoTokenizer.from_pretrained(args.pre_model_name)
	logging.info(f'Loading dataset: {args.split_path}')

	with open(args.split_path, 'r') as f:
		split = json.load(f)

	eval_data = split['eval']

	eval_dataset = QAPredictionDataset(
		eval_data
	)

	logging.info(f'eval={len(eval_dataset)}')

	eval_data_loader = DataLoader(
		eval_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=QAPredictionCollator(
			tokenizer,
			args.max_seq_len,
			args.use_tpus
		)
	)

	num_batches_per_step = (len(gpus) if not args.use_tpus else tpu_cores)
	updates_epoch = len(eval_dataset) // (args.batch_size * num_batches_per_step)
	updates_total = updates_epoch * args.epochs
	logging.info('Loading model...')
	model = QABert(
		pre_model_name=args.pre_model_name,
		learning_rate=args.learning_rate,
		lr_warmup=0.1,
		updates_total=updates_total,
		weight_decay=0.0,
		torch_cache_dir=args.torch_cache_dir,
		predict_mode=True,
		predict_path=args.output_path
	)

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
			max_epochs=0,
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

