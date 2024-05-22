### Twitter Bot Detection with Reduced Feature Set
- **authors**: Jefferson Viana Fonseca Abreu,Celia Ghedini Ralha, Joao Jose Costa Gondim
- **link** : https://ieeexplore.ieee.org/abstract/document/9280525
----------------------------------------------------------------------------------------------

#### How to reproduce:
Please follow the reproduce steps presented in the [Twibot-22 paper](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/Abreu/RF) 
for each of the collected dataset. After the feature extraction steps you will have extracted 9 separate csv datasets 
with additional split information (train/test/validation). 

In order to replicate the generic case scenario training procedure run the train.py file from the current repo :

```angular2html
python3 train.py
```

After the training, alls results would be stored in the logs.txt file.


#### Result:

| dataset                 | acc     | precision  | recall  | f1       | auc      |
|-------------------------|---------|------------|---------|----------|----------|
| cresci-2015             | 0.8130  | 0.8914     | 0.8017  | 0.8442   | 0.8171   |
| gilani-2017             | 0.5473  | 0.2        | 0.0093  | 0.0178   | 0.4899   |
| cresci-2017             | 0.7460  | 0.9777     | 0.6806  | 0.8025   | 0.8159   |
| midterm-2018            | 0.8235  | 0.9863     | 0.8009  | 0.8840   | 0.8714   |
| cresci-stock-2018       | 0.7817  | 0.8500     | 0.7241  | 0.7820   | 0.7868   |
| cresci-rtbust-2019      | 0.5882  | 0.9090     | 0.2702  | 0.4166   | 0.6190   |
| botometer-feedback-2019 | 0.71698 | 0.5        | 0.0666  | 0.1176   | 0.5201   |
| Twibot-20               | 0.4826  | 0.7121     | 0.0734  | 0.1331   | 0.5192   |
| Twibot-22               | 0.7019  | 0.4856     | 0.2067  | 0.2900   | 0.5576   |
| Total                   | 0.7055  | 0.6210     | 0.3002  | 0.4047   | 0.6042   |