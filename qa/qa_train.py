
import os
import json
import argparse
import logging
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from model_utils import QABert, get_device_id
from data_utils import QALabeledDataset, QABatchCollator

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
	parser.add_argument('-hp', '--hera_path', default=None)
	parser.add_argument('-kr', '--keep_real', default=False, action='store_true')
	parser.add_argument('-lt', '--load_checkpoint', default=None)
	parser.add_argument('-ft', '--fine_tune', default=False, action='store_true')

	args = parser.parse_args()

	pl.seed_everything(args.seed)

	save_directory = os.path.join(args.save_directory, args.model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

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
	train_data = None
	if args.fine_tune:
		train_data = split['train']
	val_data = split['eval']

	hera_data = None
	if args.hera_path is not None:
		logging.info(f'Loading HERA dataset: {args.hera_path}')
		with open(args.hera_path, 'r') as f:
			hera_data = json.load(f)

		logging.info(f'Loaded {len(hera_data)} HERA tweets.')

	train_dataset = QALabeledDataset(
		documents=train_data,
		hera_documents=hera_data,
		keep_real=args.keep_real
	)

	val_dataset = QALabeledDataset(
		val_data
	)

	if args.calc_seq_len:
		import numpy as np
		from tqdm import tqdm

		data_loader = DataLoader(
			val_dataset,
			batch_size=1,
			shuffle=True,
			num_workers=1,
			collate_fn=QABatchCollator(
				tokenizer,
				args.max_seq_len,
				False
			)
		)
		logging.info('Calculating seq len stats...')
		seq_lens = []
		for idx, batch in tqdm(enumerate(data_loader)):
			seq_len = batch['input_ids'].shape[-1]
			seq_lens.append(seq_len)
			if idx > 1000:
				break
		p = np.percentile(seq_lens, 95)
		logging.info(f'95-percentile: {p}')
		exit()

	logging.info(f'train={len(train_dataset)}, val={len(val_dataset)}')
	logging.info(f'train_labels={train_dataset.num_labels}')

	train_data_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=QABatchCollator(
			tokenizer,
			args.max_seq_len,
			args.use_tpus
		)
	)
	val_data_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=QABatchCollator(
			tokenizer,
			args.max_seq_len,
			args.use_tpus
		)
	)

	num_batches_per_step = (len(gpus) if not args.use_tpus else tpu_cores)
	updates_epoch = len(train_dataset) // (args.batch_size * num_batches_per_step)
	updates_total = updates_epoch * args.epochs
	logging.info('Loading model...')
	model = QABert(
		pre_model_name=args.pre_model_name,
		learning_rate=args.learning_rate,
		lr_warmup=0.1,
		updates_total=updates_total,
		weight_decay=0.0,
		torch_cache_dir=args.torch_cache_dir,
		load_pretrained=args.load_checkpoint is not None
	)
	tokenizer.save_pretrained(save_directory)
	model.config.save_pretrained(save_directory)
	if args.load_checkpoint is not None:
		# load checkpoint from pre-trained model
		logging.warning(f'Loading weights from trained checkpoint: {args.load_checkpoint}...')
		model.load_state_dict(torch.load(args.load_checkpoint))

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
		logging.info('Training...')
		trainer.fit(model, train_data_loader, val_data_loader)
	except Exception:
		logging.exception('Exception while training:')

	device_id = get_device_id()
	if device_id == 0 or '0' in device_id:
		logging.info(f'Saving checkpoint on device {device_id}...')
		model.to('cpu')
		torch.save(model.state_dict(), checkpoint_path)

