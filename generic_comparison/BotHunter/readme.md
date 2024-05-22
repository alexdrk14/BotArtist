### Bot-hunter: A Tiered Approach to Detecting & Characterizing Automated Activity on Twitter
- **authors**: David M. Beskow,  Kathleen M. Carley
- **link**: http://www.casos.cs.cmu.edu/publications/papers/LB_5.pdf
----------------------------------------------------------------------------------------------

#### How to reproduce:
Please follow the reproduce steps presented in the [Twibot-22 paper](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotHunter) 
for each of the collected dataset. After the feature extraction steps you will have extracted 9 separate csv datasets 
with additional split information (train/test/validation). 

In order to replicate the generic case scenario training procedure run the train.py file from the current repo :

```angular2html
python3 train.py >> logs.txt
```

After the training, alls results would be stored in the logs.txt file.


#### Result:

| dataset                 | acc     | precision | recall | f1     |
|-------------------------|---------|-----------|--------|--------|
| cresci-2015             | 0.7289  | 0.9617    | 0.5946 | 0.7349 |
| gilani-2017             | 0.5720  | 0.8       | 0.0373 | 0.0714 |
| cresci-2017             | 0.7058  | 0.9926    | 0.6167 | 0.7608 |
| midterm-2018            | 0.9879  | 0.9892    | 0.9964 | 0.9928 |
| cresci-stock-2018       | 0.7722  | 0.8830    | 0.6671 | 0.7600 |
| cresci-rtbust-2019      | 0.6029  | 0.9166    | 0.2972 | 0.4489 |
| botometer-feedback-2019 | 0.6981  | 0.3333    | 0.0666 | 0.1111 |
| Twibot-20               | 0.4826  | 0.6794    | 0.0828 | 0.1476 |
| Twibot-22               | 0.7289  | 0.6422    | 0.1793 | 0.2803 |
| Total                   | 0.7380  | 0.7796    | 0.2986 | 0.4318 |
