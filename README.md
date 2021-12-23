### Hi there 👋
READ ME

- model : bert base
- input : if len(number of character) > 511 cut by sentence below 511 and each cut sample is all predicted
- output : among prections best answer is choosed
- optimizer : use RADAM
- hyperparameter 
  --MAX_LEN = 511 
  --EPOCHS = 6 
  --VERBOSE = 2 
  --BATCH_SIZE = 8 (because we don't have big memory we just use only 8) 
  --learning_rate = 1e-5 



- model_train.ipynb : train the model with train data
- loaded_model_3.ipynb : it is the way we can load model
  - if u load weights, use '6_epochs_extract_radam_weights.h5'
- predict.ipynb : make prediction with input(it takes pretty long time)
- evaluate.ipynb : F1 and EM score
- etc
  - 1.other files are for EDA
  - 2.some of them is used to only try the 'sliding window' with bert not applied in train_model because when i use this method in your data, score is not good enough.

