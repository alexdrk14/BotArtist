# BotArtist: Twitter bot detection Machine Learning model based on Twitter suspension (2023).

Current github repo provides implementation described in paper: "BotArtist: Twitter bot detection Machine Learning model based on Twitter suspension" ....... 

Installation of required packages:
```bash
pip3 install -r requirements.txt
```

Developed Bot Detection ML framework was trained and tested over [2022 Russo-Ukrainian War Twitter dataset](https://github.com/alexdrk14/RussoUkrainianWar_Dataset). More detailed analysis of the dataset and user sentiment is provided on [ParasecurityGroup webpage](https://alexdrk14.github.io/RussiaUkraineWar/).
We manage to extract profile feature characteristics for the collected users, that was used for model creation. This dataset due to the large size is shared separetly [TODO Link]().
Also in the dataset folder we share 3 kaiser re-collected datasets with extracted user profile features and 3 types of labels 1st original paper labeling ([The False positive problem of automatic bot
detection in social science research](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241045)), the 2nd label is provied according to [Botometer](https://botometer.osome.iu.edu/bot-repository/datasets.html) and the 3rd label was collected via [compliance of Twitter API](https://developer.twitter.com/en/docs/twitter-api/compliance).
Beside that we also manage to collect two separate dataset of Twitter user discussions of energy-crisis and conspiracy. For these datasets we also provide 2 types of labels: 1st is labeled according to [Twitter API compliance](https://developer.twitter.com/en/docs/twitter-api/compliance) and 2nd with use of the [Botometer](https://botometer.osome.iu.edu/).  



Citation:
