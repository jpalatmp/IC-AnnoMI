# -*- coding: utf-8 -*-


import accelerate
print(accelerate.__version__)
!python --version

pip install transformers datasets evaluate accelerate

# For tensor flow
!pip install datasets

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import load_dataset

#import evaluate

# Imports when using tensorflow
import tensorflow as tf
from transformers import create_optimizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import PushToHubCallback
from transformers.keras_callbacks import KerasMetricCallback
from transformers import T5Tokenizer, T5ForConditionalGeneration #For T5 model

# Imports when using PyTorch
from transformers import DataCollatorWithPadding
from transformers import BertTokenizer, TFBertModel, BartTokenizer, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

#evaluation

import matplotlib.pyplot as plt
import seaborn as sns

#Sci-Kit Library
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


from google.colab import drive
drive.mount('/content/drive')

# Reading the Training  Data
train_data= pd.read_csv('load_the_file.csv') 
print(f"The shape of train data: {train_data.shape}")
print(len(train_data), '\n', type(train_data), '\n', train_data.head(3))

# Reading the Test  Data
#test_df = pd.read_csv("test.csv")
test_data=  pd.read_csv('load_the_file.csv')
print(f"The shape of test data: {test_data.shape}")
print(len(test_data), '\n', type(test_data), '\n', test_data.head(3))


#  Preparing train and test dataset to tokenize
train_df=pd.DataFrame(columns=['text', 'label'])
train_df['text'] = train_data['processed_utternace_text']
train_df['label'] = train_data["numeric_mi_quality"]
print('The train_data is ', '\n', train_df.shape, len(train_df),'\n', train_df[:5])

train_label_counts = train_df['label'].value_counts()
print(f"Total Train Labels: \n{train_label_counts}", '\n', "Total classes in train data" , len(train_label_counts.unique()))

test_label_counts = test_df['label'].value_counts()
print(f"Total test instances per class : \n{test_label_counts}", '\n', "Total classes in test data" , len(test_label_counts.unique()))


train_df = train_df.dropna()
print(len(train_df))
test_df = test_df.dropna()
print(len(test_df))

for index, (text, label) in train_df.iterrows():
    print(text)
    print(label)
    break


#Defining models

choose_model = None
choose_model = 'choose a model'
Tokenizer=None

if choose_model == 'BERT':
    print('BERT base Model is Selected')
    model_id = "bert-base-uncased"
    print(f"The selected model-id is: \n{model_id}")
    Tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"The selected tokenizer is: \n{Tokenizer}")

elif choose_model == 'bart':
    print('BART Model is Selected')
    model_id = 'facebook/bart-base'
    print(f"The selected model-id is: \n{model_id}")
    Tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"The selected tokenizer is: \n{Tokenizer}")

elif choose_model == 'distillbert':
    print('Distillbert Model is Selected')
    model_id = "distilbert/distilbert-base-uncased" 
    print(f"The selected model-id is: \n{model_id}")
    Tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"The selected tokenizer is: \n{Tokenizer}")

elif choose_model == 'roberta':
    print('Roberta Model is Selected')
    model_id = 'roberta-base'
    print(f"The selected model-id is: \n{model_id}")
    Tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"The selected tokenizer is: \n{Tokenizer}")

elif choose_model == 't5':
    print(' Model is Selected')
    model_id = 'google-t5/t5-small'
    print(f"The selected model-id is: \n{model_id}")
    Tokenizer = AutoTokenizer.from_pretrained(model_id) 
    print(f"The selected tokenizer is: \n{Tokenizer}")

elif choose_model == 'llama':
    print(' Model is Selected')
    model_id = 'enoch/llama-65b-hf'
    print(f"The selected model-id is: \n{model_id}")


else:
    print('Please use a model')

# Tokenization and preprocessing of data
tokenizer = Tokenizer
print(tokenizer)
print("the length of tokenizer is:", len(tokenizer))


# Selecting the input text length
final_max_len = 60 #210 etc. 

# Defining Preprocessing function 
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation= True, max_length=final_max_len) 

#  Using Dataset object
train_dataset_dict = Dataset.from_pandas(train_df)
test_dataset_dict = Dataset.from_pandas(test_df)

print(f"The shape of train data Dictonary: {train_dataset_dict.shape}", '\n', f"The type of train data: {type(train_dataset_dict)}",  '\n', train_dataset_dict, '\n', f"The first datapoint of train data: {train_dataset_dict[0]}") 
print(f"The shape of test data Dictonary: {test_dataset_dict.shape}", '\n', f"The type of train data: {type(test_dataset_dict)}",  '\n', test_dataset_dict, '\n', f"The first datapoint of test data: {test_dataset_dict[0]}") 

train_dataset = train_dataset_dict.map(preprocess_function, batched=True)

test_dataset = test_dataset_dict.map(preprocess_function, batched=True)

print(train_dataset, "\n", type(train_dataset))
print(test_dataset, "\n", type(train_dataset))

pre_tokenizer_columns_train= set(train_dataset_dict.features)
tokenizer_columns_train = list(set(train_dataset.features) - pre_tokenizer_columns_train)
print("Columns added by tokenizer:", tokenizer_columns_train)
print(train_dataset.features["label"])

pre_tokenizer_columns_test= set(test_dataset_dict.features)
tokenizer_columns_test = list(set(test_dataset.features) - pre_tokenizer_columns_test)
print("Columns added by tokenizer:", tokenizer_columns_test)


# Assigning high and low-quality numeric labels
miquality_id2numeric = {"low":0, "high":1}
miquality_numeric2id = {val: key for key, val in miquality_id2numeric.items()}

id2label= miquality_id2numeric
label2id = miquality_numeric2id


import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Splitting train dataset into train and validation
tokenized_dataset_total = train_dataset.train_test_split(test_size=0.1)
print(f'The train split data is {tokenized_dataset_total}')
tokenized_train_dataset = tokenized_dataset_total['train']
tokenized_val_dataset = tokenized_dataset_total['test']
print(f'The train dataset is split into train and val {tokenized_train_dataset}')
print(f'The newly created valdiation dataset {tokenized_val_dataset}')
print(f'The existing test data set is dataset {test_dataset}')

# Configuring model
model= None
batch_size = 8
num_epochs = 20
batches_per_epoch = len(tokenized_train_dataset) // batch_size 
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

#from transformers import TFAutoModelForSequenceClassification
#. Note: FAutoModelForSequenceClassification is for Trnsor Flow and ModelForSequenceClassificatio is for Pytorch

if model_id =='google-t5/t5-small':
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small") #for T5
else:
  model = TFAutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, id2label=id2label, label2id=label2id) #for distillbert/roberta/bert

print(model)
print(model.summary())

# Define data_collator for tensor flow
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

#Convert your datasets to the tf.data.Dataset format with prepare_tf_dataset():

# train data = tokenized_train_dataset
tf_train_set = model.prepare_tf_dataset(
    train_dataset,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)
# validation dataset = tokenized_val_dataset

tf_validation_set = model.prepare_tf_dataset(
    tokenized_val_dataset,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=2, mode='auto', restore_best_weights=True)
model.compile(optimizer=optimizer, metrics=['accuracy'])  # No loss argument!

from transformers.keras_callbacks import KerasMetricCallback
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set) #eval_dataset=tf_validation_set

history = model.fit(x=tf_train_set, validation_data=tf_validation_set, verbose=2, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])

# Plotting test and validation loss
def plot_history(history):
  plt.figure(figsize=(6, 4)) # this remain always on begining of plot
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Accuracy of Model')
  plt.ylabel('Accuracy')
  plt.xlabel('Number of epoch')
  plt.legend(['training set', 'validation set'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.figure(figsize=(6, 4)) # this remain always on begining of plot
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Loss of Model')
  plt.ylabel('Loss')
  plt.xlabel('Number of epoch')
  plt.legend(['training set', 'validation set'], loc='upper left')
  plt.show()
print(plot_history(history))

"""### Model Inference"""

# Prediction function
def predict_test_data(model, test_text_data):
  outputs = model(test_text_data).logits
  predictions = np.argmax(outputs, axis=1)
  return predictions

#Defining test_test for inference
test_texts= test_df['text'].to_list()
print(type(test_texts), len(test_texts), '\n', test_texts[0])

# Tokenizing the test text 
tokenized_test_text = tokenizer(test_texts, return_tensors="np", padding='max_length', truncation=True, max_length =60) 
print(type(tokenized_test_text), '\n', len(tokenized_test_text), '\n', tokenized_test_text[:1])

#Defining test data label
y_true = test_df['label'] #Already defined in begining
print(type(y_true), len(y_true), '\n', y_true.head(5))

predictions = None
print('Prediction Started:')
predictions = predict_test_data(model, tokenized_test_text) 
print('=== Printing predicted labels ====')
print(type(predictions), '\n', predictions[0:10], '\n', len(predictions))

y_pred = pd.DataFrame(predictions)
print(type(y_pred), len(y_pred), y_pred.head(10))

"""### Classification report"""

def result_plot_viv(y_test, y_pred, cluster_codes, cluster_codes_inv, choose_model):
    from google.colab import files
    print('-->> The Classification Report of : {}, is:'.format(choose_model))

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_micro =  f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print('The model accuracy is:', accuracy)
    print('Balanced Accuracy of the model is:', balanced_accuracy)
    print('The model precision is:', precision)
    print('The model recall is:', recall)
    print('--->>> F1:', f1)
    print('--->>> F1-micro:', f1_micro)
    print('--->>> F1-macro:', f1_macro)
    cm = confusion_matrix(y_test, y_pred)

    #Classification report and confusion matrix
    print('The confusion matrix is', '\n', cm)
    cr = classification_report(y_test, y_pred, digits=4)
    print('The classification report is:', '\n', cr)

    labels = id2label

    # Visualize confusion matrix 
    plt.figure(figsize=(6, 4)) 
    cm_labels = cluster_codes.keys()

    cm_array_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels) 
    conf_matrix_plot = sns.heatmap(cm_array_df, annot=True, fmt='d', cmap='Blues') 
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    conf_matrix_plot.figure.savefig(choose_model+ '_conf_matrix'+ ".png")
    files.download(choose_model+ '_conf_matrix' ".png")

    # Plotting bar chart for metrics
    metrics = ['Accuracy', 'Bal. Acc.', 'Precision', 'Recall', 'F1 Score', 'F1 micro', 'F1 macro'] 
    values = [accuracy, balanced_accuracy, precision, recall, f1, f1_micro, f1_macro]
    plt.figure(figsize=(6, 4))  

    barplot = sns.barplot(x=values, y=metrics, palette='colorblind') 

    plt.xlim(0, 1.0)
    plt.axvline(x=0.5, color='black', linestyle='--') 

    # Displaying values in front of each bar
    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f'{v:.2f}', color='Black')
    plt.xlabel('Metric Score')
    plt.ylabel('Metric')
    plt.title('Performance Metrics')
    plt.show()
    barplot.figure.savefig(choose_model+ '_barchart' + ".png")
    files.download(choose_model+ '_barchart'+ ".png")
    #return matrix, barchart

result_metrics = result_plot_viv(y_true, y_pred, id2label, label2id, choose_model) # has to be in sequence as defiend in function # casual_id2label, casual_reverse,

print(result_metrics)