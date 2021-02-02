# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/gmihaila/stock_risk_prediction/blob/master/notebooks/train_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Info
# 
# * Main Dataset: [S&P 500 stock data](https://www.kaggle.com/camnugent/sandp500)
# 
# * Download detailes for each company: [S&P 500 Companies with Financial Information](https://datahub.io/core/s-and-p-500-companies-financials#resource-s-and-p-500-companies-financials_zip)
# 
# Stock prices are flutuated in every day. So, in each day, put those stocks in order of price change to one sentence. Then, with certain window size, each stock will show up with highly related stock frequently, because they tend to move their prices together. Source: [stock2vec repo](https://github.com/kh-kim/stock2vec)
# %% [markdown]
# # Imports

# %%
import sys
import pandas as pd
import operator
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

# %% [markdown]
# # Helper Functions

# %%
def sort_dict(mydict, reversed=False):
  return sorted(mydict.items(), key=operator.itemgetter(1), reverse=reversed)

# %% [markdown]
# # Read Data

# %%
# companies' description
desc_df = pd.read_csv('constituents.csv')
print('\nCompanies Details')
print(desc_df.head())

# stocks details
stocks_df = pd.read_csv('all_stocks_5yr.csv')#, parse_dates=['date'])
print('\nCompanies Stocks')
print(stocks_df.head())

# %% [markdown]
# # Preprocess

# %%
# dicitonary for companies name and sector
companies_names = {symbol:name for symbol, name in desc_df[['Symbol', 'Name']].values}
companies_sector = {symbol:sector for symbol, sector in desc_df[['Symbol', 'Sector']].values}

# get all companies symbols
symbols = stocks_df['Name'].values
dates = set(stocks_df['date'].values)
dates = sorted(dates)

# store each individual date and all its stocks
dates_dictionary = {date:{} for date in dates}

# %% [markdown]
# # Data for Word Embeddings
# 
# For each date in out dataset we rearrange each company in ascending order based on the **change in price**.
# 
# Formula for **change in price** [source](https://pocketsense.com/calculate-market-price-change-common-stock-4829.html):
# * (closing_price - opening_price) / opening_price
# 
# We can change the formula to use highest price and lowest price. This is something we will test out.

# %%
# calculate price change for each stock and sort them in each day
for date, symbol, op, cl, in stocks_df[['date', 'Name', 'open', 'close']].values:
  # CHANGE IN PRICE: (closing_price - opening_price) / opening_price
  dates_dictionary[date][symbol] = (cl - op)/op
# sort each day reverse order
dates_dictionary = {date:sort_dict(dates_dictionary[date]) for date in dates}

stocks_w2v_data = [[value[0] for value in dates_dictionary[date]] for date in dates]

# print sample
print(stocks_w2v_data[0])

# %% [markdown]
# # Classifiers
# We utilized classifiers to predict a stock's sector based on its daily price change to determine the quality of the Word2Vec embedding and optimal number of dimensions. After testing 4 classifiers for up to 30 dimensions, we found that the accuracy of our classifiers begins to plateau at 4 dimensions.

# %%
stocks_ordered = [[value[0] for value in dates_dictionary[date]] for date in dates]


# %%
featureNumber = 15
labels = ['Industrials' ,'Health Care' ,'Information Technology' ,'Utilities','Financials','Materials', 
                     'Consumer Discretionary','Real Estate', 'Consumer Staples','Energy',
                     'Telecommunication Services']
counter = []
classifier1_array = []
classifier2_array = []
classifier3_array = []
classifier4_array = []

# test accuracy at various number of dimensions
for j in range(1,21):
    model = Word2Vec(stocks_w2v_data, min_count=1, size=j)
    words = list(model.wv.vocab)
    X = model[model.wv.vocab]
    Y = list()
    for word in words:
        Y.append(companies_sector[word])

    # split data for cross validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # predict sectors using GaussianNB, SVM, DecisionTreeClassifier and RandomForestClassifier
    model1 = GaussianNB()
    model1.fit(X_train, Y_train)
    preds1 = model1.predict(X_test)
    
    model2 = svm.SVC()
    model2.fit(X_train, Y_train)
    preds2 = model2.predict(X_test)
    
    model3 = tree.DecisionTreeClassifier()
    model3.fit(X_train, Y_train)
    preds3 = model3.predict(X_test)
    
    model4 = RandomForestClassifier()
    model4.fit(X_train, Y_train)
    preds4 = model4.predict(X_test)
    
    classifier1_array.append(accuracy_score(Y_test, preds1))
    classifier2_array.append(accuracy_score(Y_test, preds2))
    classifier3_array.append(accuracy_score(Y_test, preds3))
    classifier4_array.append(accuracy_score(Y_test, preds4))
    
    counter.append(j)

np.set_printoptions(threshold=sys.maxsize)

pyplot.plot(counter,classifier1_array)
pyplot.plot(counter,classifier2_array)
pyplot.plot(counter,classifier3_array)
pyplot.plot(counter,classifier4_array)
pyplot.ylabel('Accuracy')
pyplot.xlabel('Dimensions')
gnb_patch=mpatches.Patch(color='blue', label='GaussianNB')
svm_patch=mpatches.Patch(color='orange', label='svm')
dtc_patch=mpatches.Patch(color='green', label='Decision Tree')
rfc_patch=mpatches.Patch(color='red', label='Random Forest')
pyplot.legend(handles=[gnb_patch,svm_patch, dtc_patch, rfc_patch], loc='best')
pyplot.show()

# %% [markdown]
# # Plot 4D Representation
# Each stock is now represented in 4 dimensions, or four numbers. To visualize these dimensions, we graphed the 4 dimensions in two separate 2D plots.

# %%
# confusion matrix function
def plot_cm(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = pyplot.subplots(figsize=(20, 15)) 
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap=sns.diverging_palette(230, 30, n=9),
        ax=ax,
        annot_kws={"fontsize":20}
    )
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    pyplot.ylabel('Actual', fontsize = 20)
    pyplot.xlabel('Predicted', fontsize = 20)
    ax.set_title('Confusion Matrix', fontsize = 40, y = -.02)
    ax.set_xticklabels(class_names, fontsize=20, rotation=90)
    ax.set_yticklabels(class_names, rotation=0, fontsize = 20)
    b, t = pyplot.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    pyplot.ylim(b, t) # update the ylim(bottom, top) values
    pyplot.show()


# %%
# recreate model with 4 dimensions(this is the model that will be used for the rest of the code)
model = Word2Vec(stocks_w2v_data, min_count=1, size=4)
words = list(model.wv.vocab)
X = model[model.wv.vocab]
Y = list()
for word in words:
    Y.append(companies_sector[word])

# split data set for cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# %%
# plot confusion matrix for SVM classifier at 4 dimensions
model2.fit(X_train, Y_train)
preds2= model2.predict(X_test)
acc = accuracy_score(Y_test, preds2)
print("Accuracy is ", acc)
plot_cm(Y_test,preds2,labels)


# %%
array1X = []
array1Y = []
array2X = []
array2Y = []

# color code by sector
sector_color_dict = {'Industrials':'red','Health Care':'orange','Information Technology':'yellow','Utilities':'green',
                     'Financials':'blue','Materials':'purple','Consumer Discretionary':'cyan','Real Estate':'magenta',
                     'Consumer Staples':'pink','Energy':'brown','Telecommunication Services':'gray'}
cvec = [sector_color_dict[companies_sector[word]] for word in words]

# legend
red_patch=mpatches.Patch(color='red', label='Industrials')
orange_patch=mpatches.Patch(color='orange', label='Health Care')
yellow_patch=mpatches.Patch(color='yellow', label='Information Technology')
green_patch=mpatches.Patch(color='green', label='Utilities')
blue_patch=mpatches.Patch(color='blue', label='Financials')
purple_patch=mpatches.Patch(color='purple', label='Materials')
cyan_patch=mpatches.Patch(color='cyan', label='Consumer Discretionary')
magenta_patch=mpatches.Patch(color='magenta', label='Real Estate')
pink_patch=mpatches.Patch(color='pink', label='Consumer Staples')
brown_patch=mpatches.Patch(color='brown', label='Energy')
gray_patch=mpatches.Patch(color='gray', label='Telecommunication Services')

# graph dimensions 1 & 2
for d in range(0,505):
    array1X.append((model[model.wv.vocab])[d,0])
    array1Y.append((model[model.wv.vocab])[d,1])
pyplot.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
pyplot.scatter(array1X[:], array1Y[:], c=cvec)
pyplot.legend(handles=[red_patch,orange_patch,yellow_patch,green_patch,blue_patch,purple_patch,cyan_patch,magenta_patch,
                       pink_patch,brown_patch,gray_patch],loc='best')
pyplot.show()

# graph dimensions 3 & 4
for f in range(0,505):
    array2X.append((model[model.wv.vocab])[f,2])
    array2Y.append((model[model.wv.vocab])[f,3])
pyplot.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
pyplot.scatter(array2X[:], array2Y[:], c=cvec)
pyplot.legend(handles=[red_patch,orange_patch,yellow_patch,green_patch,blue_patch,purple_patch,cyan_patch,magenta_patch,
                       pink_patch,brown_patch,gray_patch],loc='best')
pyplot.show()   

# %% [markdown]
# # Train Word Embeddings
# We're using PCA dimensionality reduction to reduce our 4D model into 2 dimensions that can easily be graphed on a coordinate plane.

# %%
# fit a 2D PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
newResultX = []
newResultY = []
newWords = list()
newWordToken = ""
# read stock names from an input file (format: %AAL%)
with open('../data/stocks.txt') as stockFile:
    contents = stockFile.read()
    for i in range(0,505):
        newWordToken = "%" + words[i] + "%"
        #if newWordToken in contents:
        newWords.append(words[i])
        newResultX.append(result[i,0])
        newResultY.append(result[i,1])

# %% [markdown]
# # Plot 2D Representation

# %%
#increase size of figure
pyplot.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')

# color code by sector
sector_color_dict = {'Industrials':'red','Health Care':'orange','Information Technology':'yellow','Utilities':'green',
                     'Financials':'blue','Materials':'purple','Consumer Discretionary':'cyan','Real Estate':'magenta',
                     'Consumer Staples':'pink','Energy':'brown','Telecommunication Services':'gray'}
cvec = [sector_color_dict[companies_sector[word]] for word in newWords]

# create a scatter plot of the projection
pyplot.scatter(newResultX[:], newResultY[:], c = cvec)

# labels(company symbol)
'''
for i, word in enumerate(newWords):
	#pyplot.annotate(companies_names[word], xy=(newResultX[i], newResultY[i]), fontsize = 12)
    #pyplot.annotate(word, xy=(newResultX[i], newResultY[i]))
'''

pyplot.legend(handles=[red_patch,orange_patch,yellow_patch,green_patch,blue_patch,purple_patch,cyan_patch,magenta_patch,
                       pink_patch,brown_patch,gray_patch],loc='best')
pyplot.show()

# display eigen pairs(coordinate points)
print(result)
pca.fit(X)
eigen_vals = pca.explained_variance_
eigen_vecs = pca.components_
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W;\n',w)

# %% [markdown]
# # Analyze Features

# %%
# compute average feature value for each sector(15 total features)
np.set_printoptions(threshold=sys.maxsize)

# output info to averageFeatures file
with open("../notebooks/averageFeatures.txt", 'w') as averageFile:
    for k in range(0, len(sectors)):
        companiesInSector = 0
        averages = []
        for i in range (0, featureNumber):
            averages.append(0.0)
        for i in range(0,505):
            if companies_sector[words[i]] == sectors[k]:
                companiesInSector += 1
                for j in range(0,featureNumber):
                    averages[j] += model[model.wv.vocab][i,j]
        for i in range (0,featureNumber):
            averages[i] /= companiesInSector;
        averageFile.write(sectors[k])
        averageFile.write(" Average Feature Numbers: ")
        averageFile.write("\n")
        for i in range(0, featureNumber):
            averageFile.write(str(averages[i]) + " ")
    
        averageFile.write("\n\n")


# %%
# print similar stocks
target_symb = 'ALXN'

print('Symbol:%s\tName:%s\tSector: %s'%(target_symb, companies_names[target_symb], companies_sector[target_symb]))
top_similar = model.similar_by_word(target_symb, topn=20)
print('Most Similar')
for similar in top_similar:
  symb = similar[0]
  name = companies_names[symb]
  sect = companies_sector[symb]
  print('Symbol: %s\tName: %s\t\t\tSector: %s'%(symb, name, sect))


# %%
# access vector for one word
print(model['AAL'])


