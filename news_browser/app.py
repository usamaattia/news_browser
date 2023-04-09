from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
# import urllib.request
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from bm25 import preprocess_text, df

app = Flask(__name__)


df = pd.read_json('News_Dataset.json', lines = True)
df = df.drop_duplicates() 

words = set(nltk.corpus.words.words())

def preprocess_text(text):
    str1 = " "
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words or not w.isalpha())
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered = [w for w in word_tokens if not w.lower() in stop_words]
    com = str1.join(filtered)
    return com

cols = ['headline', 'short_description']
df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)



words = set(nltk.corpus.words.words())

def preprocess_text(text):
    str1 = " "
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words or not w.isalpha())
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered = [w for w in word_tokens if not w.lower() in stop_words]
    com = str1.join(filtered)
    return com

df["combined"] = df.combined.apply(preprocess_text)

vectorizer = CountVectorizer(stop_words='english')
documents_vectorized = vectorizer.fit_transform(df['combined'])
vocabulary = vectorizer.get_feature_names_out()
vocabulary
print("")
print("")
print ('We have a {} document corpus with a {} term vocabulary'.format(*documents_vectorized.shape))
print("")
print("")

dataframe = pd.DataFrame(documents_vectorized.toarray(), columns=vocabulary)
doc_ids = dataframe.index.values

def BM25_IDF_df(df):
  """
  This definition calculates BM25-IDF weights before hand as done last week
  """

  dfs = (df > 0).sum(axis=0)
  N = df.shape[0]
  idfs = -np.log(dfs / N)
  
  k_1 = 1.2
  b = 0.8
  dls = df.sum(axis=1) 
  avgdl = np.mean(dls)

  numerator = np.array((k_1 + 1) * df)
  denominator = np.array(k_1 *((1 - b) + b * (dls / avgdl))).reshape(N,1) \
                         + np.array(df)

  BM25_tf = numerator / denominator

  idfs = np.array(idfs)

  BM25_score = BM25_tf * idfs
  return pd.DataFrame(BM25_score, columns=vocabulary)



bm25_df_arr = []
    
for i in range(0, 6000, 1000):
    subset_df = dataframe.iloc[i:i+1000]
    bm25_df1 = BM25_IDF_df(subset_df)  # a dataframe with BM25-idf weights
    bm25_df_arr.append(bm25_df1)

bm25_df = pd.concat(bm25_df_arr, ignore_index=True)



def retrieve_ranking(query, bm25_df):
  q_terms = query.split(' ')
  q_terms_only = bm25_df[q_terms]
  score_q_d = q_terms_only.sum(axis=1)
  return sorted(zip(bm25_df.index.values, score_q_d.values),
                key = lambda tup:tup[1],
                reverse=True)



def query_expansion(query):
    expanded_query = []
    for term in query.split():
        synonyms = []
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            expanded_query.append(' '.join(set(synonyms)))
        else:
            expanded_query.append(term)
    return ' '.join(expanded_query)


# bm25_df = pd.read_csv('BM25_data1.csv')

@app.route("/", methods =["GET", "POST"])
def home():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       final = []
       query1 = request.form.get("search")
      #  query2 = query_expansion(query1)
      #  print(query2)
       query = preprocess_text(query1)
       print(query)
       try:
          results = retrieve_ranking(query, bm25_df)
       except:
          print("this query is not in our index")
          return render_template("errorMessage.html", query = query1)
       for i in results[:10]:
         final.append(i[0])
       print(final)
       
       headline = df.loc[final, 'headline']
       link = df.loc[final, 'link']
       disc = df.loc[final, 'short_description']
    else:
       link = []
       headline =[]
       final = []
       disc = []
    return render_template("index.html", link = link, headline= headline, final = final, disc = disc)


@app.route("/results")
def res():
    return render_template("results.html")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
