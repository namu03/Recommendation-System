from django.shortcuts import render
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from numpy import savez_compressed
from numpy import load
from sklearn.feature_extraction.text import CountVectorizer

global li1
global li2


# Create your views here.
def login(request):
    return render(request, 'App/login.html')


def register(request):
    return render(request, 'App/signup.html')


def home(request):
    df = pd.read_csv('C:\\Users\\Admin\\PycharmProjects\\Engage_1\\Project\\App\\templates\\App\\Netflix.csv')
    title_list = ["title"]
    df1 = pd.read_csv("C:\\Users\\Admin\\PycharmProjects\\Engage_1\\Project\\App\\templates\\App\\Netflix.csv",
                      usecols=title_list)
    df2 = pd.read_csv('C:\\Users\\Admin\\PycharmProjects\\Engage_1\\Project\\App\\templates\\App\\Netflix.csv')
    df2.dropna(inplace=True)
    df3 = pd.read_csv('C:\\Users\\Admin\\PycharmProjects\\Engage_1\\Project\\App\\templates\\App\\Netflix.csv')
    df3.dropna(inplace=True)
    # adding new columns to make dataset more accurate
    df['date_added'] = pd.to_datetime(df['date_added'])
    df['month_added'] = df['date_added'].dt.month
    df['month_name_added'] = df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year
    a = preprocessing(df, df1)
    movie = '0'
    # if request.method=="POST":
    movie = "Kota Factory"
    lst = top_recommendation(df2)
    new = new_release(df3)
    comedies=get_genre_wise_list(df3,'Comedies')
    horror=get_genre_wise_list(df3,'Horror')
    shape = df.shape
    mydict = {
        "df": df.to_html(),
        "shape": shape,
        "list": lst,
        "new_release": new,
        "comedies":comedies,
        "horror":horror,
    }
    return render(request, 'App/home.html', mydict)


# Function for combining the values of these columns into a single string.
def combine_features(row):
    return row['title'] + ' ' + row['listed_in'] + ' ' + row['director'] + ' ' + row['cast']


# Functions to get movie title from movie index.
def get_title_from_index(df, index):
    return df[df.index == index]["title"].values[0]


def preprocessing(df, df1):
    # Removing Duplicates
    dup_bool = df.duplicated(
        ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration',
         'listed_in', 'description', 'ratingdescription', 'user_rating_score', 'user_rating_size'])
    dups = sum(dup_bool)  # by considering all columns.
    features = ['listed_in', 'title', 'cast', 'director']
    df['cast'].isnull().values.any()
    li1 = []
    for i in range(0, len(df1.index)):
        li1.append(df1['title'][i])
    df['title'] = df['title'].str.replace(" ", "")
    li2 = []
    for j in range(0, len(df.index)):
        li2.append(df['title'][j])
    df['listed_in'] = df['listed_in'].str.replace(" ", "")
    df['director'] = df['director'].str.replace(" ", "")
    # Filling the NaN values with string.
    # Calling this function over each row of our dataframe.
    for feature in features:
        df[feature] = df[feature].fillna('')

    # applying combine_feature method over each row of Dataframe and storing the combined string in "combined_features" column.
    df['combined_features'] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])

    start = datetime.now()
    if os.path.isfile('cosine_sim2.npz'):
        print("It is already present in my local repository. Loading...\n\n")
        dict_data = load("cosine_sim2.npz")
        cosine_sim1 = dict_data['arr_0']
        print("DONE..")
    else:
        print("File is not present in my Local Repository..Creating....\n\n")
        cosine_sim1 = cosine_similarity(count_matrix)
        print('Saving it into my Local Repository....\n\n')
        savez_compressed("cosine_sim2.npz", cosine_sim1)
        print("DONE..\n")

    print(datetime.now() - start)
    return cosine_sim1


# Functions to get movie index from movie title.
def get_index_from_title(df, title1):
    index = df.index
    condition = df["title"] == title1
    indices = index[condition]
    indices_list = indices.tolist()
    print(indices_list)
    return indices_list


def recommendation(df, movie, cosine_sim):
    m1 = movie.replace(" ", "")
    # getting the movie index
    m1 = get_index_from_title(df, m1)
    similar_movies = list(enumerate(
        cosine_sim[m1[0]]))  # We will sort the list similar_movies according to similarity scores in descending order.
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]
    lst = []
    i = 0
    print("Top 10 similar movies to " + movie + " are:\n")
    for element in sorted_similar_movies:
        ind = li2.index(df, get_title_from_index(element[0]))
        lst.append(li1[ind])
        i = i + 1
        if i > 10:
            break
    return lst


def top_recommendation(df2):
    df2 = df2.sort_values(by="user_rating_score", ascending=False)
    # len(df2)
    df2.head(15)
    a = df2['title']
    a = a.tolist()
    lst = []
    for i in range(0, 10):
        lst.append(a[i])
    return lst


def new_release(df3):
    df3.sort_values(by=['date_added'], ascending=False)
    # Recommending the movie_title on the top release_date.
    a = df3['title']
    a = a.tolist()
    lst = []
    for i in range(0, 15):
        lst.append(a[i])
    return lst


# Function to get list of the movie_title for the given genre parameter.
def get_genre_wise_list(df3, genre):
    # global df3
    titles = []
    dict = {genre: '1'}
    df3['Value'] = df3['listed_in'].str.split().apply(lambda x: [dict[i] for i in x if i in dict.keys()])
    i = 0
    for ind in df3.index:
        if df3['Value'][ind]:
            if (df3['Value'][ind][0] == '1'):
                titles.append(df3['title'][ind])
                i = i + 1
        if i > 15:
            break
    return titles
