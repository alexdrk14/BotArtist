###  Botometer 101: Social bot practicum for computational social scientists
- **authors**: Kai-Cheng Yang, Emilio Ferrara, Filippo Menczer
- **link**: [https://arxiv.org/abs/2201.01608](https://arxiv.org/abs/2201.01608)
- **introduction**: Botometer (formerly BotOrNot) is a public website to checks the activity of a Twitter account and gives it a score, where higher scores mean more bot-like activity. Botometer's classification system generates more 1,000 features using available meta-data and information extracted from interaction patterns and content.
----------------------------------------------------------------------------------------------

#### How to reproduce:
Please follow the reproduce steps presented in the [Twibot-22 paper](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/Botometer).


### Results

| dataset                 | accuracy | precision | recall | f1    |
|:------------------------|:---------|:----------|:-------|:------|
| Botometer-feedback-2019 | 50.00    | 21.05     | 57.14  | 30.77 |
| Cresci-2015             | 57.92    | 50.54     | 98.95  | 66.90 |
| Cresci-2017             | 94.16    | 93.35     | 99.69  | 96.12 |
| Cresci-rtbust-2019      | 69.23    | 65.22     | 100.0  | 78.95 |
| Cresci-stock-2018       | 72.62    | 68.50     | 94.96  | 79.59 |
| Gilani-2017             | 71.56    | 62.99     | 87.91  | 77.39 | 
| Midterm-2018            | 89.46    | 31.18     | 87.88  | 46.03 |
| Twibot-20               | 53.09    | 55.67     | 50.82  | 53.13 |
| Twibot-22               | 49.87    | 30.81     | 69.80  | 42.75 |
