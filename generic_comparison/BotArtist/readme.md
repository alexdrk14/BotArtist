### BotArtist: Generic approach for bot detection in Twitter via semi-automatic machine learning pipeline
- **authors**: Anonymous Author
- **link**: TBD
----------------------------------------------------------------------------------------------

#### How to reproduce:
In order to reproduce model results, need to follow the steps:
 - Extract required set of features by executing ```bash fs.sh``` this script will extract the features for each of the dataset. 
 - Each would be stored in form of 2 files: 
   - DATASETNAME_extracted_visible.csv (aka train and validation portion)
   - DATASETNAME_extracted_hold_out.csv (aka test set)
 - In order to train the model we need to merge all training set into single file by executing: ```python3 merge_train.py```.
As an output it would merge all train files into 'generic_DF_extracted_visible.csv' and all test into 
'generic_DF_extracted_hold_out.csv'.
 - At the next step we should start the SAMLP pipeline with the provided configuration.py file and the terminal 
arguments in order to measure the model performance over multiple testing files before storing the model. 
```
#copy the provided configuration file
cp configuration.py /Path/To/SAMLP/

#change directory to the local SAMLP path
cd /Path/To/SAMLP/

#Execute pipeline in order to fine tune the model, find best features and 
# measure the performance of best model over different testing sets
python3 pipeline.py -f  generic_DF_extracted.csv -p /Path/To/Extracted/CSV/Location/ -t target -o /Path/To/Desired/Output/Folder --test Twibot-22_extracted_hold_out.csv,Twibot-20_extracted_hold_out.csv,midterm-2018_extracted_hold_out.csv,gilani-2017_extracted_hold_out.csv,cresci-stock-2018_extracted_hold_out.csv,cresci-rtbust-2019_extracted_hold_out.csv,cresci-2017_extracted_hold_out.csv,cresci-2015_extracted_hold_out.csv,botometer-feedback-2019_extracted_hold_out.csv
```
Executed script will create 'multi_test_logs.txt' file in the selected output directory where all results will be stored per testing dataset.


#### Result:

| dataset                 | acc    | precision | recall  | f1      | roc-auc |
|-------------------------|--------|-----------|---------|---------|---------|
| cresci-2015             | 0.7850 | 0.8409    | 0.81360 | 0.8270  | 0.7748  |
| gilani-2017             | 0.6543 | 0.8484    | 0.2616  | 0.3999  | 0.6124  |
| cresci-2017             | 0.8235 | 0.9536    | 0.8065  | 0.8739  | 0.8416  |
| midterm-2018            | 0.9839 | 0.9837    | 0.9974  | 0.9905  | 0.9554  |
| cresci-stock-2018       | 0.7839 | 0.7826    | 0.8315  | 0.8063  | 0.7797  |
| cresci-rtbust-2019      | 0.7500 | 0.8571    | 0.6486  | 0.7384  | 0.7598  |
| botometer-feedback-2019 | 0.6226 | 0.2222    | 0.1333  | 0.1666  | 0.4745  |
| Twibot-20               | 0.7616 | 0.7254    | 0.9000  | 0.8033  | 0.7492  |
| Twibot-22               | 0.7451 | 0.5667    | 0.5713  | 0.5690  | 0.6945  |
| Total                   | 0.7577 | 0.6359    | 0.6399  | 0.6379  | 0.7283  |


