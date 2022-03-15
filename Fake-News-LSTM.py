
######################
# LIBRARIES
######################
import time
start = time.time()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
plt.style.use('ggplot')
print("Tensorflow version " + tf.__version__)

# sklearn hata verirse apple silicon
# pip install --upgrade --force-reinstall scikit-learn

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


######################
# DATA
######################
fake_df = pd.read_csv('FAKE-NEWS-LSTM/fake_news_datasets/Fake.csv')
real_df = pd.read_csv('FAKE-NEWS-LSTM/fake_news_datasets/True.csv')



real_df.head()


######################
# DATA PRE-PROCESSING
######################
# We load Fake.csv and True.csv.




# Remove the useless columns, we only need title & text.
fake_df = fake_df[['title', 'text']]
real_df = real_df[['title', 'text']]

fake_df.isnull().sum()
real_df.isnull().sum()

# Label fake news as 0, and real news as 1.
fake_df['class'] = 0
real_df['class'] = 1


plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fake_df), color='orange')
plt.bar('Real News', len(real_df), color='green')
plt.title('Distribution of Fake News and Real News', size=12)
plt.xlabel('News Type', size=12)
plt.ylabel('# of News Articles', size=12);



# Concatenate two data frames into one.
df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
df.head()
df.shape


# Combine title & text
# into one column.
df['title_text'] = df['title'] + ' ' + df['text']
df.drop(['title', 'text'], axis=1, inplace=True)


# The way we split training and testing data must be the same for deep learning model and Logistic Regression.

X = df['title_text']
y = df['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Standard text cleaning process such as lower case, remove extra spaces and url links.
def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

X_train = normalize(X_train)
X_test = normalize(X_test)


# We put the parameters at the top like this to make it easier to change and edit.
vocab_size = 10000
embedding_dim = 64
max_length = 256
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'


# Tokenization
# Tokenizer does all the heavy lifting for us. In our articles (aka title + text) that it was tokenizing, it will take 10,000 most common words. oov_tok is to put a special value in where an unseen word is encountered. This means I want “OOV” in bracket to be used to for words that are not in the word index. fit_on_text will go through all the text and create dictionary.
# After tokenization, the next step is to turn those tokens into lists of sequence.
# When we train neural networks for NLP, we need sequences to be in the same size, that’s why we use padding. Our max_length is 256, so we use pad_sequence to make all of our articles (aka title + text) the same length which is 256.
# In addition, there are padding type and truncating type, we set both of them “post”., meaning for example, if one article at 200 in length, we padded to 256, and we padded at the end, add 56 zeros.

## tokenizer = Tokenizer(num_words=max_vocab)
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding=padding_type, truncating=trunc_type, maxlen=max_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding=padding_type, truncating=trunc_type, maxlen=max_length)


# BUILDING MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.summary()





early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])





# Visualize training over time and the results were good.
history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

plt.figure(figsize=(10,6))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss', size=15)
plt.xlabel('Epochs', size=15)
plt.ylabel('Loss', size=15)
plt.legend(prop={'size': 15})
plt.show()

plt.figure(figsize=(10,6))
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy', size=15)
plt.xlabel('Epochs', size=15)
plt.ylabel('Accuracy', size=15)
plt.legend(prop={'size': 15})
plt.ylim((0.5,1))
plt.show()



model.evaluate(X_test, y_test)



pred = model.predict(X_test)

binary_predictions = []

for i in pred:
    if i >= 0.5:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0)



print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
print('Precision on testing set:', precision_score(binary_predictions, y_test))
print('Recall on testing set:', recall_score(binary_predictions, y_test))


matrix = confusion_matrix(binary_predictions, y_test, normalize='all')
plt.figure(figsize=(10, 6))
ax= plt.subplot()
sns.heatmap(matrix, annot=True, ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted Labels', size=15)
ax.set_ylabel('True Labels', size=15)
ax.set_title('Confusion Matrix', size=15)
ax.xaxis.set_ticklabels([0,1], size=15)
ax.yaxis.set_ticklabels([0,1], size=15);


end = time.time()
print(end - start)


# You can find the original codes and lesson from these links
# https://medium.com/@actsusanli/fake-news-classification-with-lstm-or-logistic-regression-82a3527aaf13
# https://github.com/susanli2016/dsProject_LogReg_DL/blob/main/FakeNew_LogReg.ipynb