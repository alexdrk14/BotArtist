### Scalable and Generalizable Social Bot Detection through Data Selection
- **authors**: Kai-Cheng Yang, Onur Varol, Pik-Mai Hui, Filippo Menczer
- **link**:  https://arxiv.org/abs/1911.09179
----------------------------------------------------------------------------------------------


#### How to reproduce:
Please follow the reproduce steps presented in the [Twibot-22 paper](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/SGBot) 
for each of the collected dataset. After the feature extraction steps you will have extracted 9 separate csv datasets 
with additional split information (train/test/validation). 

In order to replicate the generic case scenario training procedure run the train.py file from the current repo :

```angular2html
python3 train.py >> logs.txt
```

After the training, alls results would be stored in the logs.txt file.


#### Result:

| dataset                 | acc    | precision | recall | f1    | auc   |
|-------------------------|--------|-----------|--------|-------|-------|
| cresci-2015             | 0.736  | 0.934     | 0.627  | 0.750 | 0.776 |
| gilani-2017             | 0.560  | 0.500     | 0.019  | 0.036 | 0.502 |
| cresci-2017             | 0.741  | 0.979     | 0.673  | 0.798 | 0.814 |
| midterm-2018            | 0.987  | 0.988     | 0.997  | 0.992 | 0.966 |
| cresci-stock-2018       | 0.770  | 0.850     | 0.698  | 0.767 | 0.776 |
| cresci-rtbust-2019      | 0.721  | 0.875     | 0.568  | 0.689 | 0.735 |
| botometer-feedback-2019 | 0.679  | 0.000     | 0.000  | 0.000 | 0.474 |
| Twibot-20               | 0.490  | 0.761     | 0.084  | 0.152 | 0.527 |
| Twibot-22               | 0.759  | 0.704     | 0.313  | 0.433 | 0.629 |
| Total                   | 0.766  | 0.786     | 0.409  | 0.538 | 0.677 |
| Average                 | 0.721  | 0.738     | 0.439  | 0.515 | 0.688 |
