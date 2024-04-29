""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Feature extraction based on user profile history timeline. Store collected features in csv file.
####################################################################################################################"""

import sys, ast, re, math, argparse

from dateutil.parser import parse
from tqdm import tqdm

sys.path.insert(0, "../utilities")
from mongoConnector import MongoDB
from labels_loading import load_labels
from collections import defaultdict


DATA_PATH = "../data/"
OUTPUT_FILE = 'profile_features.csv' # change for preferable name and change in utilities/mongoConfig.py as input for ML fine-tuning model


def get_age(created_at, probe_at):
    """Measure account age in days"""
    created_at = parse(created_at) if type(created_at) == str else created_at
    probe_at = parse(probe_at) if type(probe_at) == str else probe_at
    return float((probe_at.replace(tzinfo=None) - created_at.replace(tzinfo=None)).days)

def get_description_urls(item):
    if 'entities' in item and item['entities'] is not None and 'description' in item['entities'] and 'urls' in \
            item['entities']['description']:
        return [elem['url'] for elem in item['entities']['description']['urls']]
    return re.findall(r'(https?://\S+)', item['description']) if item['description'] is not None else []

def get_description_mentions(item):
    if 'entities' in item and item['entities'] is not None and 'description' in item['entities'] and 'mentions' in \
            item['entities']['description']:
        return ["@" + elem['username'] for elem in item['entities']['description']['mentions']]
    return []


def get_description_hashtags(item):
    if 'entities' in item and item['entities'] is not None and 'description' in item['entities'] and 'hashtags' in \
            item['entities']['description']:
        return ["#" + elem['tag'] for elem in item['entities']['description']['hashtags']]
    return []

def get_entropy(text):
    text = text.strip()
    freq = defaultdict(lambda: 0)
    for ch in text:
        freq[ch] += 1
    for ch in freq:
        freq[ch] /= len(text)
    res = 0
    for ch in freq:
        res -= freq[ch] * math.log(freq[ch])
    return res

class profile_features:
    def __init__(self, verbose=False, with_labels=True):
        self.labels = None
        self.verbose = verbose
        """MongoDB connection class"""
        self.mongo = MongoDB()
        self.output_filename = DATA_PATH + f'profile_features.csv'
        self.with_labels = with_labels


    def user_object_features(self, user_id):
        """Get user history item from collection"""

        user_object = self.mongo.getUserProfile(user_id)
        v1_object = True
        if user_object is None:
            print(f'User {user_id}: object  is None')
            return None

        if 'public_metrics' in user_object:
            v1_object = False
            user_object['screen_name'] = user_object['username']

        if not v1_object and type(user_object['entities']) == str:
            user_object['entities'] = ast.literal_eval(user_object['entities'])
        user_data = dict()

        """Measure account age in days"""
        user_data["age"] = get_age(user_object["created_at"], user_object["probe_at"])

        """Get URLS from description"""
        description_urls = get_description_urls(user_object)
        user_data["description_urls"] = len(description_urls)

        """Get user description"""
        description = user_object['description']

        if user_data["description_urls"] != 0:
            for url in set(description_urls):
                description = description.replace(url, ' ')
            description = description.replace("  ", " ")

        """Get Mentions from description"""
        description_mentions = get_description_mentions(user_object)
        user_data["description_mentions"] = len(description_mentions)

        if user_data["description_mentions"] != 0:
            for user_ment in set(description_mentions):
                description = description.replace(user_ment, ' ')
            description = description.replace("  ", " ")

        """Get Hashtags from description"""
        description_hashtags = get_description_hashtags(user_object)
        user_data["description_hashtags"] = len(description_hashtags)

        if user_data["description_hashtags"] != 0:
            for hst in set(description_hashtags):
                description = description.replace(hst, ' ')
            description = description.replace("  ", " ")

        """Measure Jaccard similarity between screen_name and user name"""
        name_set = set(user_object["name"].lower())
        screen_name_set = set(user_object["screen_name"].lower())
        user_data["screen_name_sim"] = len(name_set.intersection(screen_name_set)) / \
                                       len(name_set.union(screen_name_set))

        """Get Statuses/follower/friends/listed as count and as by_age values"""
        elements = ['tweet_count', 'following_count'] if not v1_object else ['statuses_count','friends_count']
        item = user_object if v1_object else user_object['public_metrics']
        for category, key_value in zip(['statuses', 'following', 'followers', 'listed'], elements + ['followers_count','listed_count']):
            user_data[category] = int(item[key_value])
            user_data[f'{category}_by_age'] = user_data[category] / user_data["age"] if user_data["age"] != 0 else 0.0



        """Get len of user: name, screen_name and description"""
        for feature_name in ["name", "screen_name", "description"]:
            user_data[feature_name + "_len"] = len(user_object [feature_name]) if user_object [feature_name] != None else 0

        """
            Identify and measure number , upper/lower case and special characters in 
            name, screen_name and user description
        """

        for f_categ in ["name", "screen_name", "description"]:
            feature_val = user_object[f_categ] if user_object [f_categ] != None else ""
            for case in ["upper", "lower", "digit", "special"]:
                if case == "upper":
                    user_data[f_categ + "_" + case + "_len"] = sum([i.isupper() for i in feature_val])
                elif case == "lower":
                    user_data[f_categ + "_" + case + "_len"] = sum([i.islower() for i in feature_val])
                elif case == "digit":
                    user_data[f_categ + "_" + case + "_len"] = sum([i.isdigit() for i in feature_val])
                else:
                    user_data[f_categ + "_" + case + "_len"] = user_data[f_categ + "_len"] - (
                            user_data[f_categ + "_upper_len"] +
                            user_data[f_categ + "_lower_len"] +
                            user_data[f_categ + "_digit_len"])
                user_data[f_categ + "_" + case + "_pcnt"] = user_data[f_categ + "_" + case + "_len"] / len(
                    feature_val) if len(
                    feature_val) != 0 else 0

        """Compute entropy for screen_name and name"""
        for category in ['screen_name', 'name']:
            user_data[f'{category}_entropy'] = get_entropy(user_object[category])

        """Boolean value if the profile has any characters in location section"""
        user_data["has_location"] = 0 if len(user_object['location'].strip()) == 0 else 1

        """Followers to  friends score"""
        user_data["foll_friends"] = (user_data["followers"] / float(user_data["friends"])) if \
            user_data["friends"] != 0 else 0

        """Found if profile has url link"""
        user_data["profile_url"] = 1 if len(re.findall(r'(https?://\S+)', user_object['url'])) > 0 else 0

        """Found total number of unique urls in the entyre profile object"""
        user_data["total_urls"] = len(set(description_urls).union(set(re.findall(r'(https?://\S+)', user_object['url'])))) if \
            user_data["profile_url"] else len(set(description_urls))

        """Profile image urls 0 if the link is for default url 1 for not default url"""
        user_data["profile_image"] = int(user_object['profile_image_url'].find('default_profile_normal') == -1)

        """Boolean values from user object"""
        user_data["verified"] = 1 if user_object["verified"] else 0
        user_data["protected"] = 1 if user_object["protected"] else 0
        user_data["user_id"] = user_id
        if self.with_labels:
            user_data["target"] = int(self.labels[self.labels["user_id"] == user_id]["target"].values[0])
        return user_data

    def initialize_outfile(self):

        self.feature_names = ["age", "description_urls", "description_mentions", "description_hashtags", "screen_name_sim",
                    "statuses", "statuses_by_age", "following", "following_by_age", "followers", "followers_by_age",
                    "listed", "listed_by_age", "profile_image", "description_upper_len", "description_lower_len",
                    "description_digit_len", "description_spec_len", "description_upper_pcnt", "description_lower_pcnt",
                    "description_digit_pcnt", "description_spec_pcnt", "description_len",
                    "screen_name_upper_len", "screen_name_lower_len", "screen_name_digit_len", "screen_name_spec_len",
                    "screen_name_upper_pcnt", "screen_name_lower_pcnt", "screen_name_digit_pcnt",
                    "screen_name_spec_pcnt",
                    "screen_name_len", "name_upper_len", "name_lower_len", "name_digit_len", "name_spec_len",
                    "name_upper_pcnt", "name_lower_pcnt", "name_digit_pcnt", "name_spec_pcnt", "name_len", "screen_name_entropy",
                    "name_entropy", "has_location",
                    "profile_url", "total_urls", "foll_friends",
                    'verified', 'protected', 'target', "user_id"]

        self.f_out = open(self.output_filename, "w+")
        self.f_out.write("\t".join(self.feature_names) + "\n")

    def dump_user_vector(self, user_data):
        outline = "\t".join(["{}".format(user_data[feature_name]) for feature_name in self.feature_names])
        self.f_out.write(f'{outline}\n')

    def user_extract_features(self, user_id):
        if self.verbose:
            print("-"*100 + f'\nStarting feature extraction for {user_id} user')
        return self.user_object_features(int(user_id))



    def start_extraction(self):
        if self.verbose:
            print("-"*100 + f'\nStarting feature extraction')
            print("-"*2 + ">Loading labels ..")

        """Initialize the output file with feature headers"""
        self.initialize_outfile()

        if self.verbose:
            print("-"*2 + ">Start feature computation and extraction...")

        sequence = tqdm()
        if self.with_labels:
            """Load user labels (alive , suspended, removed and protected) and create progress bar object"""
            self.labels = load_labels(DATA_PATH)
            for user_id in sequence:
                user_data = self.user_object_features(int(user_id))

                if user_data is not None:
                    self.dump_user_vector(user_data)
        else:
            client, db = self.mongo.connect()
            for item in tqdm(db['usersHistory'].find({}, no_cursor_timeout=True)):
                user_id = int(item['user_id'])
                max_date = parse('2000-01-01')
                raw_date = ''
                for date in item['history']:
                    new_date = parse(date)
                    if new_date > max_date:
                        max_date = new_date
                        raw_date = date
                if raw_date == '':
                    continue
                user_data = self.user_object_features(user_id, item['history'][raw_date])
                if user_data is not None:
                    self.dump_user_vector(user_data)

        self.mongo.close()
        self.f_out.close()
        if self.verbose:
            print("All features are extracted and stored in: {} file!".format(self.output_filename))


parser = argparse.ArgumentParser(description='Feature extraction for bot detection')
parser.add_argument('-verbose', dest="verbose", action="store_true", help='Verbose output')
parser.add_argument('--without_labels', dest="nolabels", action="store_true", help='Description for foo argument')



if __name__ == "__main__":
    args = vars(parser.parse_args())
    item = profile_features(verbose=args.verbose, with_labels=(not args.nolabels))
    item.start_extraction()
