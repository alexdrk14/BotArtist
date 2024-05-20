## The output of the SAMLP pipeline 
We store the output of SAMLP pipeline combined with selected features, performance and selected 
hyper-parameters for each developed model configuration. As explained in the original paper, 
we separate out experiment into two different case scenarios. In first case scenario we compare the performance 
(F1 score) of selected  bot detection methods for each of the nine selected dataset (also known as per dataset). 
In such scenario we train each  model separately per dataset and measure the performance over the hidden data portion (test data).

| Method         |  C-15       | G-17        | C-17        | M-18        | C-S-18      | C-R-19      | B-F-19      | TB-20       | TB-22       | Average   |
|:---------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:----------|
| SGBot          | 77.9        | 72.1        | 94.6        | 99.5        | 82.3        | 82.7        | 49.6        | 84.9        | 36.6        | 75.57     |
| Kudugunta      | 75.3        | 49.8        | 91.7        | 94.5        | 50.9        | 49.2        | 49.6        | 47.3        | 51.7        | 62.22     |
| Hayawi         | 85.6        | 34.7        | 93.8        | 91.5        | 60.8        | 60.9        | 20.5        | 77.1        | 24.7        | 61.06     |
| BotHunter      | 97.2        | 69.2        | 91.6        | <u>99.6</u> | 82.2        | 82.9        | 49.6        | 79.1        | 23.5        | 74.98     |
| NameBot        | 83.4        | 44.8        | 85.7        | 91.6        | 61.1        | 67.5        | 38.5        | 65.1        | 0.5         | 59.80     |
| Abreu          | 76.4        | 66.7        | 95.0        | 97.9        | 76.9        | <u>83.5</u> | <u>53.8</u> | 77.1        | 53.4        | 75.63     |
| **BotArtist**  | 98.3        | <u>76.1</u> | 97.0        | **99.7**    | 80.6        | **88.3**    | **68.4**    | 82.2        | <u>58.2</u> | **83.19** |
| Cresci         | 1.17        | -           | 22.8        | -           | -           | -           | -           | 13.7        | -           | -         |
| Wei            | 82.7        | -           | 78.4        | -           | -           | -           | -           | 57.3        | 53.6        | -         |
| BGSRD          | 90.8        | 35.7        | 86.3        | 90.5        | 58.2        | 41.1        | 13.0        | 70.0        | 21.1        | 56.30     |
| RoBERTa        | 95.8        | -           | 94.3        | -           | -           | -           | -           | 73.1        | 20.5        | -         |
| T5             | 89.3        | -           | 92.3        | -           | -           | -           | -           | 70.5        | 20.2        | -         |
| Efthimion      | 94.1        | 5.2         | 91.8        | 95.9        | 68.2        | 71.7        | 0.0         | 67.2        | 27.5        | 57.95     |
| Kantepe        | 78.2        | -           | 79.4        | -           | -           | -           | -           | 62.2        | **58.7**    | -         |
| Miller         | 83.8        | 59.9        | 86.8        | 91.1        | 56.8        | 43.6        | 0.0         | 74.8        | 45.3        | 60.23     |
| Varol          | 94.7        | -           | -           | -           | -           | -           | -           | 81.1        | 27.5        | -         |
| Kouvela        | 98.2        | 66.6        | <u>99.1</u> | 98.2        | 80.4        | 81.1        | 28.1        | 86.5        | 30.0        | 74.24     |
| Santos         | 78.8        | 14.5        | 83.0        | 92.4        | 65.2        | 75.7        | 21.0        | 60.3        | -           | -         |
| Lee            | <u>98.6</u> | 67.8        | **99.3**    | 97.9        | <u>82.5</u> | 82.7        | 50.3        | 80.0        | 30.4        | 76.61     |
| LOBO           | **98.8**    | -           | 97.7        | -           | -           | -           | -           | 80.8        | 38.6        | -         |
| Moghaddam      | 73.9        | -           | -           | -           | -           | -           | -           | 79.9        | 32.1        | -         |
| Alhosseini     | 92.2        | -           | -           | -           | -           | -           | -           | 72.0        | 38.1        | -         |
| Knauth         | 91.2        | 39.1        | 93.4        | 91.3        | **94.0**    | 54.2        | 41.3        | 85.2        | 37.1        | 69.64     |
| FriendBot      | 97.6        | -           | 87.4        | -           | -           | -           | -           | 80.0        | -           | -         |
| SATAR          | 95.0        | -           | -           | -           | -           | -           | -           | 86.1        | -           | -         |
| Botometer      | 66.9        | **77.4**    | 96.1        | 46.0        | 79.6        | 79.0        | 30.8        | 53.1        | 42.8        | 63.5      |
| Rodrifuez-Ruiz | 87.7        | -           | 85.7        | -           | -           | -           | -           | 63.1        | 56.6        | -         |
| GraphHist      | 84.5        | -           | -           | -           | -           | -           | -           | 67.6        | -           | -         |
| EvolveBot      | 90.1        | -           | -           | -           | -           | -           | -           | 69.7        | 14.1        | -         |
| Dehghan        | 88.3        | -           | -           | -           | -           | -           | -           | 76.2        | -           | -         |
| GCN            | 97.2        | -           | -           | -           | -           | -           | -           | 80.8        | 54.9        | -         |
| GAT            | 97.6        | -           | -           | -           | -           | -           | -           | 85.2        | 55.8        | -         |
| HGT            | 96.9        | -           | -           | -           | -           | -           | -           | **88.2**    | 39.6        | -         |
| SimpleHGN      | 97.5        | -           | -           | -           | -           | -           | -           | **88.2**    | 45.4        | -         |
| BotRGCN        | 97.3        | -           | -           | -           | -           | -           | -           | 87.3        | 57.5        | -         |
| RGT            | 97.8        | -           | -           | -           | -           | -           | -           | <u>88.0</u> | 42.9        | -         |

Additionally, we also compare the best methods from previous step in more general scenario, where models are trained over
 the mix of the dataset. In such case all training data between nine dataset are mixed together allowing for model to 
capture more general patterns. In order to test performance of the developed general models we compute their performances 
(F1 score) over each remaining testing set separately and also as a single union.


| Method           | C-15        | G-17        | C-17        | M-18        | C-S-18      | C-R-19      | B-F-19   | TB-20       | TB-22       | Total       | Average     |
|:-----------------|:------------|:------------|:------------|:------------|:------------|:------------|:---------|:------------|:------------|:------------|:------------|
| BotArtist        | 82.7        | <u>39.9</u> | 87.3        | <u>99.0</u> | **80.6**    | <u>73.8</u> | 16.6     | **80.3**    | **56.9**    | **63.7**    | **68.5**    | 
| Lee              | 82.3        | 0.0         | 83.6        | 97.7        | 78.2        | 67.7        | 20.0     | 8.5         | 42.4        | 52.9        | 53.3        | 
| Abreu            | <u>84.4</u> | 0.3         | 80.1        | 88.4        | 67.1        | 40.8        | 11.7     | 15.6        | 29.0        | 40.4        | 46.3        | 
| SGBot            | 75.0        | 3.6         | 79.8        | **99.2**    | 76.7        | 68.9        | 0.0      | 15.2        | <u>43.3</u> | <u>53.8</u> | 51.5        | 
| BotHunter        | 73.4        | 7.1         | 76.0        | **99.2**    | 76.0        | 44.8        | 11.1     | 14.7        | 28.0        | 43.1        | 47.8        | 
| Kouvela          | **95.5**    | 20.4        | <u>94.7</u> | 98.1        | 78.4        | 71.4        | 21.0     | 28.5        | 36.0        | 52.0        | 60.5        | 
| Botometer        | 66.9        | **77.4**    | **96.1**    | 46.0        | <u>79.6</u> | **79.0**    | **30.8** | <u>53.1</u> | 42.8        | 45.3        | <u>63.5</u> | 


The identified results, selected features and model configuration are stored in PerDataset and General folders.