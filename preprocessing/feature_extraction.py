""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Feature extraction based on user profile history timeline. Store collected features in csv file.
####################################################################################################################"""

import sys
import numpy as np

from dateutil.parser import parse
from tqdm import tqdm

sys.path.insert(0, "../utilities")
from mongoConnector import MongoDB
from labels_loading import load_labels


DATA_PATH = "../data/"
OUTPUT_FILE = 'profile_features.csv' # change for preferable name and change in utilities/mongoConfig.py as input for ML fine-tuning model


class profile_features:
    def __init__(self, verbose=False, category=None):
        self.labels = None
        self.verbose = verbose
        """MongoDB connection class"""
        self.mongo = MongoDB()
        self.category = category
        self.output_filename = DATA_PATH + f'profile_features.csv'


    def user_object_features(self, user_id):
        """Get user history item from collection"""

        user_object = self.mongo.getUserProfile(user_id)
        if user_object is None:
            print(f'User {user_id}: object  is None')
            return None
        user_data = dict()

        """Measure account age in days"""
        first_activity_day = parse(user_object["created_at"]) if type(user_object["created_at"]) == str else user_object[
            "created_at"]
        first_activity_day = first_activity_day.replace(tzinfo=None)
        user_data["age"] = float((user_object["probe_at"] - first_activity_day).days)

        for feature_name in ["favourites", "listed", "statuses", "followers", "friends"]:
            user_data[feature_name] = user_object[feature_name + "_count"]
            """claculate activity divided by age of account in order to find growth by registered days"""
            user_data[feature_name + "_by_age"] = user_data[feature_name] / user_data["age"] if user_data[
                                                                                                    "age"] != 0.0 else 0

        """Get len of user: name, screen_name and description"""
        for feature_name in ["name", "screen_name", "description"]:
            user_data[feature_name + "_len"] = len(user_object [feature_name]) if user_object [feature_name] != None else 0


        """Measure Jaccard similarity between screen_name and user name"""
        name_set = set(user_object["name"].lower())
        screen_name_set = set(user_object["screen_name"].lower())
        user_data["screen_name_sim"] = len(name_set.intersection(screen_name_set)) / \
                                       len(name_set.union(screen_name_set))

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

        """Followers to  friends score"""
        user_data["foll_friends"] = (user_data["followers"] / float(user_data["friends"])) if \
            user_data["friends"] != 0 else 0

        """Boolean values from user object"""
        user_data["geo"] = 1 if user_object ["geo_enabled"] else 0
        user_data["protected"] = 1 if user_object ["protected"] else 0
        user_data["location"] = 1 if user_object ["location"] != "" else 0
        user_data["background_img"] = 1 if user_object ["profile_use_background_image"] else 0
        user_data["default_prof"] = 1 if user_object ["default_profile"] else 0
        user_data["url"] = 1 if user_object ["url"] != None else 0
        user_data["verified"] = 1 if user_object ["verified"] else 0
        user_data["user_id"] = user_id
        user_data["target"] = int(self.labels[self.labels["user_id"] == user_id]["target"].values[0])
        return user_data

    def initialize_outfile(self):
        feature_names = ["age"]

        for feature_name in ["favourites", "listed", "statuses", "followers", "friends"]:
            feature_names.append(feature_name)
            feature_names.append(feature_name + "_by_age")

        for feature_name in ["name", "screen_name", "description"]:
            feature_names.append(feature_name + "_len")

        feature_names.append("screen_name_sim")

        for f_categ in ["name", "screen_name", "description"]:
            for case in ["upper", "lower", "digit", "special"]:
                if case == "upper":
                    feature_names.append(f_categ + "_" + case + "_len")
                elif case == "lower":
                    feature_names.append(f_categ + "_" + case + "_len")
                elif case == "digit":
                    feature_names.append(f_categ + "_" + case + "_len")
                else:
                    feature_names.append(f_categ + "_" + case + "_len")
                feature_names.append(f_categ + "_" + case + "_pcnt")

        feature_names.append("foll_friends")
        feature_names.append("geo")
        feature_names.append("protected")
        feature_names.append("location")
        feature_names.append("background_img")
        feature_names.append("default_prof")
        feature_names.append("url")
        feature_names.append("verified")
        feature_names.append("user_id")
        feature_names.append("target")

        self.f_out = open(self.output_filename, "w+")
        self.f_out.write("\t".join(feature_names) + "\n")
        return feature_names

    def dump_user_vector(self, user_data, features):
        outline = ""
        for feature_name in features:
            outline += "{}\t".format(user_data[feature_name])

        self.f_out.write("{}\n".format(outline[:-1]))

    def user_extract_features(self, user_id):
        if self.verbose:
            print("-"*100 + f'\nStarting feature extraction for {user_id} user')

        return self.user_object_features(int(user_id))



    def start_extraction(self):
        if self.verbose:
            print("-"*100 + f'\nStarting feature extraction for {self.category} mongo collection')
            print("-"*2 + ">Loading labels ..")

        """
            Load user labels (alive , suspended, removed and protected) and create progress bar object
        """
        self.labels = load_labels(DATA_PATH)

        """Initialize the output file with feature headers"""
        feature_names = self.initialize_outfile()

        if self.verbose:
            print("-"*2 + ">Start feature computation and extraction...")

        for user_id in tqdm(self.labels["user_id"]):
            user_data = self.user_object_features(int(user_id))

            if user_data is not None:
                self.dump_user_vector(user_data, feature_names)

        self.mongo.close()
        self.f_out.close()
        if self.verbose:
            print("All features are extracted and stored in: {} file!".format(self.output_filename))

if __name__ == "__main__":
    item = profile_features()
    item.start_extraction()
