import json, math, ijson, os, sys, re, time, ast
import pandas as pd
from argparse import ArgumentParser
import os.path as osp
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from urllib.error import HTTPError

seen_locations = set()
reg_locations = dict()
geolocator = Nominatim(user_agent="myGeocoder")


def remove_emoji(string):
    print(string)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def is_valid_location(location, depth=0):
    global geolocator
    if depth >= 10:
        return None

    is_true = False
    if location is None:
        return 0
    location = remove_emoji(location)
    location = ''.join([ch for ch in location if ch.isalpha() or ch == " "])

    if len(location) < 2:
        return 0

    if location in seen_locations:
        return reg_locations[location]
    try:
        location_data = geolocator.geocode(location)
        print(f"Found for : {location} the:{location_data}")
        is_true = 1 if location_data is not None else 0
    except GeocoderTimedOut:
        pass
    except Exception as e:
        time.sleep(60 * 2)
        print(e)
        geolocator = Nominatim(user_agent="myGeocoder")
        return is_valid_location(location, depth=depth + 1)
    seen_locations.add(location)
    reg_locations[location] = is_true
    return is_true


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset = args.dataset
DF = []

assert dataset in ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']

if not osp.exists('tmp/{}'.format(dataset)):
    os.makedirs('tmp/{}'.format(dataset))

collect_year = dataset.split('-')[-1]
if len(collect_year) == 2:
    collect_year = '20{}'.format(collect_year)

# path = '{}'.format(dataset)
# if not osp.exists(path):
#    raise KeyError

label_data = pd.read_csv('label.csv'.format(dataset))
label_index = {}
for index, item in label_data.iterrows():
    label_index[item['id']] = int(item['label'] == 'bot')
print(len(label_index))


def get_feature(value, segment=None):
    if value is None:
        return 0
    assert segment in ['bot', 'have', 'entropy', 'length', 'profile_image_url', None]
    if segment == 'bot':
        flag = False
        for content in value:
            if content is None:
                continue
            if content.find('bot') != -1:
                flag = True
        return int(flag)
    if segment == 'have':
        value = value.strip()
        if len(value) == 0:
            return 0
        return 1
    if segment == 'entropy':
        value = value.strip()
        p = {}
        for i in value:
            if i not in p:
                p[i] = 0
            p[i] += 1
        for i in p:
            p[i] /= len(value)
        ans = 0
        for i in p:
            ans -= p[i] * math.log(p[i])
        return ans
    if segment == 'length':
        return len(value.strip())
    if segment == 'profile_image_url':
        return int(item['profile_image_url'].find('default_profile_normal') == -1)
    if dataset == 'Twibot-20' and value in ['True ', 'False ']:
        value = (value == 'True ')
    if isinstance(value, bool):
        value = int(value)
    return value


def calc_age(created_at):
    if created_at is None:
        return 365 * 2
    created_at = created_at.strip()
    if dataset in ['Twibot-20', 'gilani-2017', 'cresci-stock-2018', 'cresci-rtbust-2019',
                   'cresci-2017', 'cresci-2015', 'botometer-feedback-2019']:
        mode = '%a %b %d %H:%M:%S %z %Y'
    elif dataset in ['Twibot-22']:
        mode = '%Y-%m-%d %H:%M:%S%z'
    elif dataset in ['midterm-2018']:
        mode = '%a %b %d %H:%M:%S %Y'
    else:
        raise KeyError
    if created_at.find('L') != -1:
        created_time = datetime.fromtimestamp(int(created_at.replace('000L', '')))
    else:
        created_time = datetime.strptime(created_at, mode)
    collect_time = datetime.strptime('{} Dec 31'.format(collect_year), '%Y %b %d')
    created_time = created_time.replace(tzinfo=None)
    collect_time = collect_time.replace(tzinfo=None)
    difference = collect_time - created_time
    return difference.days


def get_description_urls(item):
    if type(item['entities']) == str:
        item['entities'] = ast.literal_eval(item['entities'])
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


"""Return UpperCharacters len , UpperCharacters"""


def get_text_stats(text):
    """Remove spaces in begining and end of the text"""
    text = text.strip()

    """Remove double spaces"""
    while "  " in text:
        text = text.replace("  ", " ")

    string_size = len(text)
    if string_size == 0:
        return 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0

    up_len = sum([ch.isupper() for ch in text])
    low_len = sum([ch.islower() for ch in text])
    dig_len = sum([ch.isdigit() for ch in text])
    spec_len = string_size - (up_len + low_len + dig_len)

    return up_len, low_len, dig_len, spec_len, up_len / string_size, low_len / string_size, dig_len / string_size, spec_len / string_size, string_size


def dump_vector(vector=None):
    global DF
    features = ["age", "description_urls", "description_mentions", "description_hashtags", "screen_name_sim",
                "statuses", "statuses_by_age", "following", "following_by_age", "followers", "followers_by_age",
                "listed", "listed_by_age", "profile_image", "description_upper_len", "description_lower_len",
                "description_digit_len", "description_spec_len", "description_upper_pcnt", "description_lower_pcnt",
                "description_digit_pcnt", "description_spec_pcnt", "description_len",
                "screen_name_upper_len", "screen_name_lower_len", "screen_name_digit_len", "screen_name_spec_len",
                "screen_name_upper_pcnt", "screen_name_lower_pcnt", "screen_name_digit_pcnt", "screen_name_spec_pcnt",
                "screen_name_len", "name_upper_len", "name_lower_len", "name_digit_len", "name_spec_len",
                "name_upper_pcnt",
                "name_lower_pcnt", "name_digit_pcnt", "name_spec_pcnt", "name_len", "screen_name_entropy",
                "name_entropy",
                "has_location",  # "valid_location",
                "profile_url", "total_urls", "foll_friends",
                'verified', 'protected', 'target', "uid"]
    if vector is None:
        print("Creating DataFrame")
        DF = pd.DataFrame(DF, columns=features)
        DF['uid'] = [int(item.replace("u", "")) for item in DF['uid']]
        print(DF.shape)
        DF.to_csv(f"{dataset}_extracted.csv", index=False, header=True, sep="\t")
        split = pd.read_csv('split.csv', header=0)
        split['id'] = [int(item.replace("u", "")) for item in split['id']]
        test_uid = set(split[split["split"] == "test"]["id"])
        DF_test = DF[DF["uid"].isin(test_uid)].copy()

        DF_test.to_csv(f"{dataset}_extracted_hold_out.csv", index=False, header=True, sep="\t")

        train_uid = set(split[split["split"] != "test"]["id"])
        DF_train = DF[DF["uid"].isin(train_uid)].copy()
        DF_train.to_csv(f"{dataset}_extracted_visible.csv", index=False, header=True, sep="\t")
    else:
        DF.append([vector[sf] for sf in features])


if __name__ == '__main__':
    with open('node.json' if dataset != 'Twibot-22' else 'user.json') as f:
        data = ijson.items(f, 'item')
        features = []
        idx = []
        labels = []
        skiped = 0
        for item in tqdm(data, ncols=0):
            user_vector = {}
            uid = item['id']
            if uid.find('u') == -1:
                break

            screen_name = item['username']
            name = item['name']
            if screen_name is None and name is None:
                continue
            user_vector["age"] = calc_age(item['created_at'])
            # user_age = calc_age(item['created_at'])

            description = item['description']

            """Get URLS from description"""
            description_urls = get_description_urls(item)
            user_vector["description_urls"] = len(description_urls)

            if user_vector["description_urls"] != 0:
                for url in set(description_urls):
                    description = description.replace(url, ' ')
                description = description.replace("  ", " ")

            """Get Mentions from description"""
            description_mentions = get_description_mentions(item)
            user_vector["description_mentions"] = len(description_mentions)

            if user_vector["description_mentions"] != 0:
                for user_ment in set(description_mentions):
                    description = description.replace(user_ment, ' ')
                description = description.replace("  ", " ")

            """Get Hashtags from description"""
            description_hashtags = get_description_hashtags(item)
            user_vector["description_hashtags"] = len(description_hashtags)

            if user_vector["description_hashtags"] != 0:
                for hst in set(description_hashtags):
                    description = description.replace(hst, ' ')
                description = description.replace("  ", " ")

            """Measure Jaccard similarity between screen_name and user name"""
            name_set = set(name.lower())
            screen_name_set = set(screen_name.lower())
            user_vector["screen_name_sim"] = len(name_set.intersection(screen_name_set)) / \
                                             len(name_set.union(screen_name_set))

            """Get Statuses/follower/friends/listed as count and as by_age values"""
            user_vector["statuses"] = get_feature(item['public_metrics']['tweet_count'])
            user_vector["statuses_by_age"] = user_vector["statuses"] / user_vector["age"] if user_vector[
                                                                                                 "age"] != 0 else 0.0

            user_vector["following"] = get_feature(item['public_metrics']['following_count'])
            user_vector["following_by_age"] = user_vector["following"] / user_vector["age"] if user_vector[
                                                                                                   "age"] != 0 else 0.0

            user_vector["followers"] = get_feature(item['public_metrics']['followers_count'])
            user_vector["followers_by_age"] = user_vector["followers"] / user_vector["age"] if user_vector[
                                                                                                   "age"] != 0 else 0.0

            user_vector["listed"] = get_feature(item['public_metrics']['listed_count'])
            user_vector["listed_by_age"] = user_vector["listed"] / user_vector["age"] if user_vector[
                                                                                             "age"] != 0 else 0.0

            """Profile image urls 0 if the link is for default url 1 for not default url"""
            user_vector["profile_image"] = get_feature(item['profile_image_url'], 'profile_image_url')

            up_len, low_len, dig_len, spec_len, up_pcnt, low_pcnt, dig_pcnt, spec_pcnt, item_len = get_text_stats(
                description)
            user_vector["description_upper_len"] = up_len
            user_vector["description_lower_len"] = low_len
            user_vector["description_digit_len"] = dig_len
            user_vector["description_spec_len"] = spec_len
            user_vector["description_upper_pcnt"] = up_pcnt
            user_vector["description_lower_pcnt"] = low_pcnt
            user_vector["description_digit_pcnt"] = dig_pcnt
            user_vector["description_spec_pcnt"] = spec_pcnt
            user_vector["description_len"] = item_len

            up_len, low_len, dig_len, spec_len, up_pcnt, low_pcnt, dig_pcnt, spec_pcnt, item_len = get_text_stats(
                screen_name)
            user_vector["screen_name_upper_len"] = up_len
            user_vector["screen_name_lower_len"] = low_len
            user_vector["screen_name_digit_len"] = dig_len
            user_vector["screen_name_spec_len"] = spec_len
            user_vector["screen_name_upper_pcnt"] = up_pcnt
            user_vector["screen_name_lower_pcnt"] = low_pcnt
            user_vector["screen_name_digit_pcnt"] = dig_pcnt
            user_vector["screen_name_spec_pcnt"] = spec_pcnt
            user_vector["screen_name_len"] = item_len

            up_len, low_len, dig_len, spec_len, up_pcnt, low_pcnt, dig_pcnt, spec_pcnt, item_len = get_text_stats(name)
            user_vector["name_upper_len"] = up_len
            user_vector["name_lower_len"] = low_len
            user_vector["name_digit_len"] = dig_len
            user_vector["name_spec_len"] = spec_len
            user_vector["name_upper_pcnt"] = up_pcnt
            user_vector["name_lower_pcnt"] = low_pcnt
            user_vector["name_digit_pcnt"] = dig_pcnt
            user_vector["name_spec_pcnt"] = spec_pcnt
            user_vector["name_len"] = item_len

            """Compute entropy for screen_name and name"""
            user_vector["screen_name_entropy"] = get_feature(item['username'], 'entropy')
            user_vector["name_entropy"] = get_feature(item['username'], 'entropy')

            """Boolean value if the profile has any characters in location section"""
            user_vector["has_location"] = get_feature(item['location'], 'have')

            """Todo check for actual location"""
            # try:
            #    user_vector["valid_location"] = is_valid_location(item['location'])
            # except Exception as e:
            #    dump_vector()
            #    print(e)
            #    sys.exit(-1)

            # if user_vector["valid_location"] is None:
            #    skiped += 1
            #    print(f"Skipped: {skiped}")
            #    continue

            """Found if profile has url link"""
            user_vector["profile_url"] = 1 if len(re.findall(r'(https?://\S+)', item['url'])) > 0 else 0

            """Found total number of unique urls in the entyre profile object"""
            total_unique_urls = len(set(description_urls).union(set(re.findall(r'(https?://\S+)', item['url'])))) if \
            user_vector["profile_url"] else len(set(description_urls))
            user_vector["total_urls"] = total_unique_urls

            """Follower / followers ratio"""
            user_vector["foll_friends"] = (user_vector["followers"] / float(user_vector["following"])) if user_vector[
                                                                                                              "following"] != 0 else 0.0

            user_vector['verified'] = 1 if item['verified'] else 0
            user_vector['protected'] = 1 if item['protected'] else 0
            user_vector['target'] = 2 if uid not in label_index else label_index[uid]

            user_vector["uid"] = uid
            dump_vector(vector=user_vector)
        dump_vector()

    # features = np.array(features)
    # labels = np.array(labels)
    # print(len(idx))
    # print(features.shape)
    # print(labels.shape)
    # json.dump(idx, open('tmp/{}/idx.json'.format(dataset), 'w'))
    # np.save('tmp/{}/features.npy'.format(dataset), features)
    # np.save('tmp/{}/labels.npy'.format(dataset), labels)