import os, re
import json
import string
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import ktrain
from ktrain import text

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

import matplotlib.pyplot as plt

fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')


preproc_name = os.path.join(fpath, 'tf_model.preproc')
preproc_isfile = os.path.isfile(preproc_name) 
st.write(preproc_name)
st.write(preproc_isfile)
predictor = ktrain.load_predictor(fpath)
#with open(preproc_name, 'rb') as f: preproc = pickle.load(f)
#preproc = joblib.load(preproc_name)

#model = preproc.get_model(fpath=fpath)
#predictor = text.predictor.TextPredictor(model, preproc, batch_size=32)

def get_twitter_api():
    
    ckey=os.environ['ckey']
    csecret=os.environ['csecret']
    atoken=os.environ['atoken']
    asecret=os.environ['asecret']

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    api = tweepy.API(auth)
    return api

def clean_text(text):
    regex = re.compile('[%s]' % re.escape('|'))
    text = regex.sub(" ", text)
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    return words

def getTweets(ID, api):

    twitter_id = ID

    tweets = tweepy.Cursor(api.user_timeline,id=twitter_id).items(50)
    tweets_list = [[tweet.text, tweet.created_at, tweet.id_str, tweet.user.screen_name, tweet.coordinates, tweet.place, tweet.retweet_count, tweet.favorite_count, tweet.lang, tweet.source, tweet.in_reply_to_status_id_str, tweet.in_reply_to_user_id_str, tweet.is_quote_status] for tweet in tweets]
    tweets_df = pd.DataFrame(tweets_list,columns=['text', 'Tweet Datetime', 'Tweet Id', 'Twitter @ Name', 'Tweet Coordinates', 'Place Info', 'Retweets', 'Favorites', 'Language', 'Source', 'Replied Tweet Id', 'Replied Tweet User Id Str', 'Quote Status Bool'])

    return tweets_df[['text', 'Tweet Datetime', 'Twitter @ Name']]

def lookup_twitter_ID(ID, api):
    try:
        api.get_user(ID)
        return True

    except:
        return False

def startAnalyzing(texts,source):

    st.write(texts)
    test_text = clean_text(texts)

    with st.spinner('Predicting...'):
        predicted = predictor.predict(test_text)
        probabilities = predictor.predict_proba(test_text)
        probabilities_df = pd.DataFrame(probabilities, categories)
        probabilities_df.columns = ['Probabilities']
        probabilities_df['Probabilities_pct'] = pd.Series(["{0:.2f}%".format(val * 100) for val in probabilities_df['Probabilities']], index = probabilities_df.index)
        st.write(predicted)
        st.write(probabilities_df[['Probabilities_pct']].T)

    st.subheader('Based on {}, this person is {} with probability of {:.2%}'.format(source, predicted, max(probabilities)))

    fig, ax = plt.subplots()
    ax.bar(probabilities_df.index, probabilities_df.Probabilities)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)

categories = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

def main():

    st.title("What's your Twitter Persona's MBTI?")
    inputType = st.radio("Do you want to give Twitter ID or raw text?",("Twitter", "Raw Text"))

    if inputType == "Twitter":
        st.header("Please enter the twitter ID that you want to search")

        ID = st.text_input("Enter the ID: ", "@")
        api = get_twitter_api()
        lookup_result = lookup_twitter_ID(ID, api)
        checkID = st.button("Check ID")

        if checkID:

            if lookup_result:
                st.success("{0} works!".format(ID))

                number_of_tweets = 5
                data_df = getTweets(ID,api)
                st.write(data_df.head(number_of_tweets))
                texts = ''

                for tweet_text in data_df.text.values.tolist():
                    texts += tweet_text
                    texts += ' '
                
                startAnalyzing(texts,'recent 5 tweets')
        
            else:
                st.error("Try Different ID")
            
    else:

        texts = st.text_area("Enter Your Message:", "Type Here")
        if st.button("Submit"):
            result = texts
            st.success("Text Submitted Successfully!")

            startAnalyzing(texts,'given texts')

    st.sidebar.header("About")
    st.sidebar.text("This is a work of Hojin Lee.")
    st.sidebar.text("Contact me if any questions")
    st.sidebar.text("hjlee9295@gmail.com")

    st.markdown('For source code see: https://github.com/hjlee9295/MBTI_twitter')


main()
