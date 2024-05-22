### Bot-detective: An explainable Twitter bot detection service with crowdsourcing functionalities
- **authors**: Maria Kouvela, Ilias Dimitriadis, Athena Vakali
- **link**: https://dl.acm.org/doi/abs/10.1145/3415958.3433075
----------------------------------------------------------------------------------------------


#### How to reproduce:
Please follow the reproduce steps presented in the [Twibot-22 paper](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/Kouvela) 
for each of the collected dataset. After the feature extraction steps you will have extracted 9 separate csv datasets 
with additional split information (train/test/validation). 

In order to replicate the generic case scenario training procedure run the train.py file from the current repo :

```angular2html
python3 train.py
```

After the training, alls results would be stored in the logs.txt file.


#### Result:

| dataset                 | acc    | precision | recall | f1      | auc    |
|-------------------------|--------|-----------|--------|---------|--------|
| cresci-2015             | 0.9449 | 0.9409    | 0.9711 | 0.9558  | 0.9373 |
| gilani-2017             | 0.5843 | 0.65      | 0.1214 | 0.2047  | 0.5350 |
| cresci-2017             | 0.9105 | 0.9964    | 0.9024 | 0.9471  | 0.9384 |
| midterm-2018            | 0.9695 | 0.9771    | 0.9868 | 0.9819  | 0.9329 |
| cresci-stock-2018       | 0.7817 | 0.8403    | 0.7364 | 0.7849  | 0.7858 |
| cresci-rtbust-2019      | 0.7058 | 0.7575    | 0.6756 | 0.7142  | 0.7088 |
| botometer-feedback-2019 | 0.7169 | 0.5       | 0.1333 | 0.2105  | 0.5403 |
| Twibot-20               | 0.5217 | 0.7417    | 0.1766 | 0.2853  | 0.5521 |
| Twibot-22               | 0.7656 | 0.6432    | 0.2504 | 0.3605  | 0.6003 |
| Total                   | 0.7755 | 0.7722    | 0.3925 | 0.5205  | 0.6702 |

