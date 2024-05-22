### Seven Months with the Devils: A Long-Term Study of Content Polluters on Twitter
- **original paper authors**: Kyumin Lee, Brian Eoff, James Caverlee
- **link**: https://ojs.aaai.org/index.php/ICWSM/article/view/14106
----------------------------------------------------------------------------------------------


#### How to reproduce:
Please follow the reproduce steps presented in the [Twibot-22 paper](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/Lee) 
for each of the collected dataset. After the feature extraction steps you will have extracted 9 separate csv datasets 
with additional split information (train/test/validation). 

In order to replicate the generic case scenario training procedure run the train.py file from the current repo :

```angular2html
python3 train.py
```

After the training, alls results would be stored in the logs.txt file.


#### Result:

| dataset                 | acc    | precision | recall | f1     | auc     |
|-------------------------|--------|-----------|--------|--------|---------|
| cresci-2015             | 0.8018 | 0.9427    | 0.7307 | 0.8233 | 0.8273  |
| gilani-2017             | 0.5473 | 0.0       | 0.0    | 0.0    | 0.4889  |
| cresci-2017             | 0.7826 | 0.97451   | 0.7326 | 0.8364 | 0.8362  |
| midterm-2018            | 0.9618 | 0.9891    | 0.9651 | 0.9769 | 0.9547  |
| cresci-stock-2018       | 0.7817 | 0.8500    | 0.7241 | 0.7820 | 0.7868  |
| cresci-rtbust-2019      | 0.7058 | 0.84      | 0.5675 | 0.6774 | 0.7192  |
| botometer-feedback-2019 | 0.6981 | 0.4       | 0.1333 | 0.2    | 0.527   |
| Twibot-20               | 0.4750 | 0.7435    | 0.0453 | 0.0854 | 0.5134  |
| Twibot-22               | 0.7524 | 0.6725    | 0.3101 | 0.4245 | 0.6235  |
| Total                   | 0.7595 | 0.7620    | 0.4054 | 0.5293 | 0.6710  |

