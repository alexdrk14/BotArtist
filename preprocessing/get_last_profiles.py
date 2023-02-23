""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Script parse the Tweets mongoDB collection and create new userHistory collection that contain
user object by day of object sampling (user activity like tweet, retweet and quote)
registered in selected Twitter discussion traffic.

In case of dynamic dataset collection, also allow to re-run same script without any parameters
since it store the last date of parsed data and allow to update userHistory collection from last sampling date.
####################################################################################################################"""

import sys, os.path


from collections import defaultdict
from dateutil.parser import parse
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, "../utilities")
from mongoConnector import MongoDB



class UserHistory:
    def __init__(self):
        """
            Main dictionary that contain user profiles information, where :
            -first layer use as Key user id
            -second layer use as Key the date in format YYYY-MM-DD

            * as Values keep the JSON object of user profile
            IMPORTANT keep only buffer of data since this dictionary requires large amount of memory.
            When buffer is big enough we store data in mongo collection and free memory
        """
        self.profiles = defaultdict()

        """
           Help dictionary that keeps for entire dataset user ids as a Key value 
           and as Values set with parsed dates of user objects fir this particular user.
        """
        self.parsed = defaultdict(lambda: datetime(2000, 1, 1))
        self.in_db = set()
        self.known_user_ids = None
        """MongoDB connection class"""
        self.connector = MongoDB()

        """
         Store initial process start date.
         This date will we changed in case the script is already execute and require only update on new dates
        """
        self.start_date = datetime(2022, 12, 1, 0, 0, 0)
        self.dateFile = "LastProfiles_last_date.txt"

    def main(self):

        """Check if dateFile exist. Used for better performance in case we already parse
        some portion of Tweets data and require only to execute since last execution date."""
        if os.path.exists(self.dateFile):
            date = [int(x) for x in open(self.dateFile, "r").read().split("-")]
            self.start_date = datetime(date[0], date[1], date[2], 0, 0, 0)
        print(f'Last date: {self.start_date}')
        """Count number of Tweets for progress bar"""
        _, db = self.connector.connect()
        progress_full = db.Tweets.count_documents({"created_at": {"$lte": self.start_date + timedelta(days=1)}})
        self.connector.close()

        """Load selected user ids from labels.csv file"""
        self.load_selected_users()
 
        """Load already parsed information from collection (alredy parsed userID's and object dates)    """
        self.load_parsed_users()
        """Progress bar parameters"""
        self.pbar = tqdm(desc="Progress", total=progress_full, unit_scale=True)
      
        """
            Parse tweets collection day by day.
            That helps to track changes and dynamically update the profile history 
            since we track the last parsed date.
        """
        while (self.start_date - timedelta(days=1)) >= datetime(2022, 2, 22):
            print(f'\nWorking on date:{self.start_date}')
            self.get_users_from_tweets()
            self.store()
            self.start_date -= timedelta(days=1)
            print(f"Done {len(self.in_db.intersection(self.users))} of {len(self.users)}")
            if len(self.in_db.intersection(self.users)) == len(self.users):
                print("All Users parsed")
                break
            """Store the start_date for next execution in dateFile"""
            date_out_stream = open(self.dateFile, "w+")
            date_out_stream.write("{}-{}-{}".format(
                self.start_date.year,
                self.start_date.month,
                self.start_date.day))
            date_out_stream.close()
        self.pbar.close()

    def load_selected_users(self):
        """load user from label file in order keep selected user ids"""
        print('Loading selected users:')
        _, db = self.connector.connect()
        self.users = set([])
        for entry in tqdm(db.users.find({}, {"id": 1}, no_cursor_timeout=True)):
            self.users.add(int(entry["id"]))
        self.connector.close()

    def load_parsed_users(self):
        """
           Load user ids and parsed dates.
           Used to avoid duplicates
        """
        print('Loading parsed users:')
        _, db = self.connector.connect()
        self.ignore_users = set()
        for entry in tqdm(db.AllUsersHistory.find({})):
            if entry["probe_at"] >= self.start_date:
                self.ignore_users.add(entry["id"])
            else:
                self.in_db.add(entry["id"])
                self.parsed[entry["id"]] = entry["probe_at"]

        self.users -= self.ignore_users
        self.connector.close()

    def store(self):
        """Store parsed user objects. Used for lower memory usage"""
        _, db = self.connector.connect()
        insert_many = []
        for user_id in self.profiles:
            if user_id not in self.in_db:
                insert_many.append(self.profiles[user_id])
                self.in_db.add(user_id)
        if len(insert_many) != 0:
            db.AllUsersHistory.insert_many(insert_many)
            del (insert_many)
        self.connector.close()
        """Clear memory by re-creating the profiles dictionary"""
        del (self.profiles)
        self.profiles = defaultdict()

    def get_users_from_tweets(self):
        """Get user object from Tweet collection"""
        _, db = self.connector.connect()
        for tweet in db.Tweets.find({"created_at": {
            "$gte": self.start_date,
            "$lt": self.start_date + timedelta(days=1)}
        }, batch_size=5000,
                no_cursor_timeout=True):
            """Parse tweet creator user object"""
            self.object_parse(tweet)

            if "retweeted_status" in tweet and int(tweet["retweeted_status"]["user"]["id"]) in self.users:
                """In case of retweet also parse the retweeted user object"""
                self.object_parse(tweet["retweeted_status"])
            elif "quoted_status" in tweet and int(tweet["quoted_status"]["user"]["id"]) in self.users:
                """In case of quote also parse the quoted user object"""
                self.object_parse(tweet["quoted_status"])

            """Check if profile dictionary require data flush"""
            if len(self.profiles.keys()) >= 1000: 
                if len(self.profiles.keys()) != 0:
                    self.store()
            self.pbar.update(1)
        if len(self.profiles.keys()) != 0:
            self.store()
        self.connector.close()

    def object_parse(self, object):
        date = object["created_at"]
        if type(date) == str:
            date = parse(date)
        date = date.replace(tzinfo=None)

        if object["user"]["id"] in self.users and date > self.parsed[object["user"]["id"]]:
            """
               Add user object to profile dictionary and store userId and 
               date to remember parsed dates of particular user
            """
            object["user"]["probe_at"] = date
            self.profiles[int(object["user"]["id"])] = object["user"]
            self.parsed[int(object["user"]["id"])] = date


if __name__ == "__main__":
    history_parser = UserHistory()
    history_parser.main()
