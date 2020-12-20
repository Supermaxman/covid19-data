
import os
import json
import argparse
import tweepy
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm


def convert_date(date_str):
	return datetime.strftime(
		datetime.strptime(
			date_str,
			'%a %b %d %H:%M:%S +0000 %Y'
		),
		'%Y%m%d%H%M'
	)


def convert_date_range(date_str, hours):
	return datetime.strftime(
		datetime.strptime(
			date_str,
			'%a %b %d %H:%M:%S +0000 %Y'
		) + timedelta(hours=hours),
		'%Y%m%d%H%M'
	)


def read_jsonl(path):
	examples = []
	seen_ids = set()
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				if ex['id'] in seen_ids:
					continue
				examples.append(ex)
	return examples


def get_reply_graph(tweet_id, user_name, created_at, api, max_depth=1, max_hours=24, max_results=100):
	replies = tweepy.Cursor(
		api.search_full_archive,
		environment_name='researchall',
		query=f'@{user_name} lang:en',
		fromDate=convert_date(created_at),
		toDate=convert_date_range(created_at, hours=max_hours),
		maxResults=max_results,
	).items(max_results)

	reply_list = defaultdict(list)
	for reply in replies:
		reply = reply._json
		reply_to_id = str(reply['in_reply_to_status_id'])
		reply_id = str(reply['id'])
		reply_list[reply_to_id].append(reply_id)
	direct_replies = reply_list[tweet_id]
	reply_graph = {tweet_id: {}}
	reply_parent = {}
	for parent_id, children_ids in reply_list.items():
		reply_graph[parent_id] = {}
		for child_id in children_ids:
			reply_parent[child_id] = parent_id

	reply_fringe = direct_replies

	for depth in range(max_depth):
		new_reply_fringe = []
		for reply_id in reply_fringe:
			parent_id = reply_parent[reply_id]
			reply_graph[reply_id] = {}
			reply_graph[parent_id][reply_id] = reply_graph[reply_id]
			reply_replies = reply_list[reply_id]
			new_reply_fringe.extend(reply_replies)
		reply_fringe = new_reply_fringe
	return reply_graph[tweet_id]


def get_reply_ids(reply_graph):
	reply_ids = []
	for reply_id, reply_reply_graph in reply_graph.items():
		reply_ids.append(reply_id)
		reply_ids.extend(get_reply_ids(reply_reply_graph))
	return reply_ids


def get_replies(api, tweet, max_depth, max_hours, max_results):
	tweet_id = str(tweet['id'])
	user_name = tweet['user']['screen_name']
	created_at = tweet['created_at']
	reply_graph = get_reply_graph(
		tweet_id,
		user_name,
		created_at,
		api,
		max_depth=max_depth,
		max_hours=max_hours,
		max_results=max_results
	)
	tweet['reply_graph'] = reply_graph
	reply_ids = get_reply_ids(reply_graph)
	tweet['reply_ids'] = reply_ids
	replies = []
	for reply_id in reply_ids:
		try:
			reply_tweet = api.get_status(reply_id, tweet_mode='extended')._json
			replies.append(reply_tweet)
		except Exception as e:
			pass
	tweet['replies'] = replies
	return tweet


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
			"--API_key", help="Your API key", type=str, required=True
	)
	parser.add_argument(
			"--API_secret_key", help="your API secret key", type=str, required=True
	)
	parser.add_argument(
			"--access_token", help="your Access token", type=str, required=True
	)
	parser.add_argument(
			"--access_token_secret", help="your Access secret token", type=str, required=True
	)
	parser.add_argument(
			"--input_file", default='data/downloaded_tweets_extra.jsonl'
	)
	parser.add_argument(
			"--output_file", default='data/downloaded_tweets_with_replies.jsonl'
	)
	parser.add_argument(
			"--num_download", default=None, type=int
	)
	args = parser.parse_args()

	auth = tweepy.OAuthHandler(
		args.API_key,
		args.API_secret_key
	)
	auth.set_access_token(
		args.access_token,
		args.access_token_secret
	)

	api = tweepy.API(
		auth,
		wait_on_rate_limit=True
	)

	tweets = read_jsonl(args.input_file)
	if os.path.exists(args.output_file):
		downloaded_tweets = set([t['id'] for t in read_jsonl(args.output_file)])
	else:
		downloaded_tweets = set()
	tweets = [t for t in tweets if t['id'] not in downloaded_tweets and t['check_replies']][:args.num_download]
	with open(args.output_file, 'a') as fo:
		for tweet in tqdm(tweets):
			tweet_with_replies = get_replies(
				api,
				tweet,
				max_depth=10,
				max_hours=24,
				max_results=100
			)
			tweet_json = json.dumps(tweet_with_replies)
			fo.write(f'{tweet_json}\n')



