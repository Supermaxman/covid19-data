
import os
import json
import csv
import argparse
import tweepy
import glob


def read_csv(path):
    output = []
    with open(path, 'r') as f:
        # creating a csv reader object
        reader = csv.reader(f)
        for row in reader:
            # misconception_id,misconception,tweet_id,label
            misconception_id, misconception, tweet_id, label = row
            output.append(tweet_id)
    return output


def acquire_from_twitter_api(tweet_ids, output_file, error_file, args):

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
        parser=tweepy.parsers.JSONParser(),
        wait_on_rate_limit=True
    )

    if not os.path.exists(output_file):
        raise Exception('Output file already exists!')

    with open(output_file, 'w') as fo:
        with open(error_file, 'w') as fe:
            num_tweets = 0
            num_downloaded = 0
            for idx, tweet_id in enumerate(tweet_ids):
                if idx % 500 == 0:
                    print(f'{num_tweets} IDs processed ({num_downloaded/num_tweets:.2f})')
                    num_tweets = 0
                    num_downloaded = 0
                try:
                    tweet = api.get_status(tweet_id, tweet_mode='extended')
                    tweet_json = json.dumps(tweet)
                    fo.write(f'{tweet_json}\n')
                    num_downloaded += 1
                except tweepy.TweepError as e:
                    tweet_error = json.dumps({
                        'id': tweet_id,
                        'error': e
                    })
                    fe.write(f'{tweet_error}\n')
                num_tweets += 1


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
        "--input_file", default='covid_lies.csv'
    )
    parser.add_argument(
        "--output_file", default='data/downloaded_tweets.jsonl'
    )
    parser.add_argument(
        "--error_file", default='data/error_tweets.jsonl'
    )
    args = parser.parse_args()

    print('Loading ids...')
    input_ids = read_csv(args.input_file)
    acquire_from_twitter_api(
        input_ids,
        args.output_file,
        args.error_file,
        args
    )


