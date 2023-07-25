
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Read the data set
books = pd.read_csv("Books.csv", encoding='Latin1')

# Drop all three Image URL features.
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)

# Replace these three empty cells with ‘Other’.
books.at[187689, 'Book-Author'] = 'Other'

books.at[128890, 'Publisher'] = 'Other'
books.at[129037, 'Publisher'] = 'Other'

# Check for the unique years of publications
books['Year-Of-Publication'].unique()

books.at[209538, 'Publisher'] = 'DK Publishing Inc'
books.at[209538, 'Year-Of-Publication'] = 2000
books.at[209538, 'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'
books.at[209538, 'Book-Author'] = 'Michael Teitelbaum'

books.at[221678, 'Publisher'] = 'DK Publishing Inc'
books.at[221678, 'Year-Of-Publication'] = 2000
books.at[
    209538, 'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
books.at[209538, 'Book-Author'] = 'James Buckley'

books.at[220731, 'Publisher'] = 'Gallimard'
books.at[220731, 'Year-Of-Publication'] = '2003'
books.at[209538, 'Book-Title'] = 'Peuple du ciel - Suivi de Les bergers '
books.at[209538, 'Book-Author'] = 'Jean-Marie Gustave Le ClÃ?Â©zio'


# Converting year of Publication to numbers
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)

# In[17]:


# Replacing Invalid years with max year
from collections import Counter

cnt = Counter(books['Year-Of-Publication'])
[k for k, v in cnt.items() if v == max(cnt.values())]

# In[18]:


books.loc[books['Year-Of-Publication'] > 2021, 'Year-Of-Publication'] = 2002
books.loc[books['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002

# In[19]:


# Uppercasing all alphabets in ISBN
books['ISBN'] = books['ISBN'].str.upper()
books.sample(2)

# In[20]:


# Drop duplicate rows
books.drop_duplicates(keep='last', inplace=True)
books.reset_index(drop=True, inplace=True)

# Read user data set
user = pd.read_csv("Users.csv", encoding='Latin1')
# user.columns

# user.info()
user.boxplot(column='Age')

percent_missing = user['Age'].isnull().sum() * 100 / len(user['Age'])
missing_value_df = pd.DataFrame({'column_name': user['Age'],
                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True)

percent_missing = user['Age'].isnull().sum() * 100 / len(user['Age'])

req = user[user['Age'] <= 70]
req = req[req['Age'] >= 15]

median = round(req['Age'].median())

# outliers with age grater than 70 are substituted with median
user.loc[user['Age'] > 70, 'Age'] = median
# outliers with age less than 15 years are substitued with median
user.loc[user['Age'] < 15, 'Age'] = median
# filling null values with median
user['Age'] = user['Age'].fillna(median)

user['Age'] = pd.cut(user.Age, bins=[2, 17, 65, 99], labels=['Child', 'Adult', 'Elderly'])

user["Age"].value_counts(normalize=True)

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
user['Age'] = label_encoder.fit_transform(user['Age'])

user["Age"].value_counts(normalize=True)

# Drop duplicate rows
user.drop_duplicates(keep='last', inplace=True)
user.reset_index(drop=True, inplace=True)

user['Age'].value_counts().plot(kind='bar');

user['Location'].unique()

for i in user:
    user['Country'] = user.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')

user['Country'] = user['Country'].astype('str')

d = list(user.Country.unique())
d = set(d)
d = list(d)
d = [x for x in d if x is not None]
d.sort()

user = user.drop(columns=['Location'], axis=1)

user['Country'].replace(
    ['', '01776', '02458', '19104', '23232', '30064', '85021', '87510', 'alachua', 'america', 'austria', 'autralia',
     'cananda', 'geermany', 'italia', 'united kindgonm', 'united sates', 'united staes', 'united state',
     'united states', 'us'],
    ['other', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'usa', 'australia', 'australia', 'canada',
     'germany', 'italy', 'united kingdom', 'usa', 'usa', 'usa', 'usa', 'usa'], inplace=True)

# Read ratings data set
ratings = pd.read_csv("Ratings.csv", encoding='Latin1')

# checking all ratings number or not
from pandas.api.types import is_numeric_dtype

# Uppercasing all alphabets in ISBN
ratings['ISBN'] = ratings['ISBN'].str.upper()

# Drop duplicate rows
ratings.drop_duplicates(keep='last', inplace=True)
ratings.reset_index(drop=True, inplace=True)

# ratings.sample()

# Hence segregating implicit and explict ratings datasets
ratings_explicit = ratings[ratings['Book-Rating'] != 0]
ratings_implicit = ratings[ratings['Book-Rating'] == 0]

# In[55]:


# Let's find the top 5 books which are rated by most number of users.
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())
rating_count.sort_values('Book-Rating', ascending=False).head()

most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'],
                                index=np.arange(5), columns=['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')

data = pd.merge(books, ratings, on='ISBN', how='inner')
data = pd.merge(data, user, on='User-ID', how='inner')


data.loc[(data['Book-Rating'] == 0), 'Book-Rating'] = np.NaN

# Replacing null data with median
data['Book-Rating'] = data['Book-Rating'].fillna(data.groupby('Age')['Book-Rating'].transform('median'))
data['Book-Rating'].unique()

# In[63]:


# data['Book-Rating'].value_counts()


def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
        columns={df.index.name: 'col_name', 0: 'Missing Values', 1: '% of Total Values'})
    mz_table['Data_type'] = df.dtypes
    mz_table = mz_table.sort_values('% of Total Values', ascending=False)
    return mz_table.reset_index()


# missing_values(data)

year_count = books['Year-Of-Publication'].value_counts()
year_count = pd.DataFrame(year_count)

import plotly.express as px




from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot



colors = ['#494BD3', '#E28AE2', '#F1F481', '#79DB80', '#DF5F5F',
          '#69DADE', '#C2E37D', '#E26580', '#D39F49', '#B96FE3']


df = data[data['Book-Rating'] != 0]

# top rated books
books_top20 = df['Book-Title'].value_counts().head(20)
books_top20 = list(books_top20.index)

top20_books = pd.DataFrame(columns=data.columns)

for book in books_top20:
    cond_df = data[data['Book-Title'] == book]

    top20_books = pd.concat([top20_books, cond_df], axis=0)

top20_books = top20_books[top20_books['Book-Rating'] != 0]
top20_books = top20_books.groupby('Book-Title')['Book-Rating'].agg('mean').reset_index().sort_values(by='Book-Rating',
                                                                                                     ascending=False)

top10_books = top20_books.head(10)


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prYellow(skk): print("\033[93m {}\033[00m".format(skk))


def popular_books():
    for (book, ratings) in zip(top10_books['Book-Title'], top10_books['Book-Rating']):
        st.markdown('<span style="color: red;">{}</span>'.format(book), unsafe_allow_html=True)
        st.markdown('<span style="color: green;">Ratings -></span>\n'.format(book), unsafe_allow_html=True)
        st.markdown('<span style="color: red;">{}</span>'.format(str(round(ratings, 1))), unsafe_allow_html=True)
        st.markdown("-" * 50)


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# We are going to count the ratings of the books to classify the common and the rare ones
count_rate = pd.DataFrame(df['Book-Title'].value_counts())

rare_books = count_rate[count_rate["Book-Title"] <= 100].index
print()

# so if the book is not included in the rare books we are going to classify it as common one

common_books = df[~df["Book-Title"].isin(rare_books)]
# common_books.head()

item_based_cb = common_books.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")



# We created the recommendation system
# item based collaborative recommendation system is ready, next we have to put it all together to bulild our RS

def item_based_coll_rs(book_title):
    book_title = str(book_title)
    if book_title in data['Book-Title'].values:

        count_rate = pd.DataFrame(df['Book-Title'].value_counts())
        rare_books = count_rate[count_rate["Book-Title"] <= 100].index

        common_books = df[~df["Book-Title"].isin(rare_books)]

        if book_title in rare_books:
            prYellow("A rare book, so u may try our popular books: \n ")
            return None

        else:

            item_based_cb = common_books.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
            sim = item_based_cb[book_title]
            recommendation_df = pd.DataFrame(item_based_cb.corrwith(sim).sort_values(ascending=False)).reset_index(
                drop=False)

            if not recommendation_df['Book-Title'][recommendation_df['Book-Title'] == book_title].empty:
                recommendation_df = recommendation_df.drop(
                    recommendation_df[recommendation_df["Book-Title"] == book_title].index[0])

            less_rating = []
            for i in recommendation_df["Book-Title"]:
                if df[df["Book-Title"] == i]["Book-Rating"].mean() < 5:
                    less_rating.append(i)

            if recommendation_df.shape[0] - len(less_rating) > 5:
                recommendation_df = recommendation_df[~recommendation_df["Book-Title"].isin(less_rating)]
                recommendation_df.columns = ["Book-Title", "Correlation"]

            return recommendation_df



    else:
        prYellow("This book is not in our library, check out our most popular books:")
        print()
        return None



