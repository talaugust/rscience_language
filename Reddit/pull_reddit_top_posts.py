
##### Python script to run and collect top ten hot posts from subreddits every day 
##### Collects comments and posts
 
# Load environment vars, expected to have this .env in your directory 
from dotenv import load_dotenv
load_dotenv()

### Import
from apscheduler.schedulers.blocking import BlockingScheduler

import time
import math
import praw
import json
import os
import csv
from termcolor import colored
from datetime import datetime


def pull_wrapper():
    ### set all varaibles, these will not be changed each time the script runs
    client_id = os.environ.get("client_id")
    client_secret = os.environ.get("client_secret")
    user_agent = os.environ.get("user_agent")
    reddit_obj = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
    tested_subs_list = ['science', 'politics', 'Economics', 'depression', 'Cooking', 'pics', 'Naruto', 'BabyBumps']
    field_list = ('title', 'author_fullname', 'url', 'score', 
              'created_utc', 'num_comments', 
              'selftext', 'subreddit_id', 
              'distinguished', 'stickied', 'pinned')
    number_posts = 5
    file_name = 'test_top_posts.csv'
    
    pull_top_posts(tested_subs_list, field_list, number_posts, file_name, reddit_obj)
    

def pull_top_posts(tested_subs, fields, num_posts, filename, reddit):
    # get time to second -- save it up here so all pulls are matched by their timestamp
    pull_time = math.floor(time.time())
    list_of_items = []
    print('Pulling top ', num_posts, ' posts at ' , colored(pull_time, 'red'),' from: ', tested_subs)
    for subreddit in tested_subs:
        sub = reddit.subreddit(subreddit)
        submissions = sub.hot(limit=num_posts)
        print("Pulling from...", colored(subreddit, 'green'))
        for s in submissions:
            submission_dict = vars(s)
            sub_dict = {field:submission_dict[field] for field in fields}
            sub_dict['pulled_timestamp'] = pull_time
            sub_dict
            list_of_items.append(sub_dict)


    print('Saving csv of ', len(list_of_items), ' posts to', colored(filename, 'magenta'))    
    with open(filename, mode='a') as csv_file:
        fieldnames = list(fields) + ['pulled_timestamp']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in list_of_items:
            writer.writerow(row)

# filename = 'test-top-posts-' + str(pull_time) + '.json'
# print('Saving csv of ', len(list_of_items), ' posts to', colored(filename, 'magenta'))    
# with open(filename, 'w') as outfile:  
#     json.dump(list_of_items, outfile, indent=4)
     

scheduler = BlockingScheduler()
scheduler.add_job(pull_wrapper,
                  'interval', minutes=1)
scheduler.start()