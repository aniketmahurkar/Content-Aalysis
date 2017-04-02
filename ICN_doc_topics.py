import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

dirSrc = 'doc_01'
num_clusters = 4

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.EnglishStemmer()
punc = string.punctuation

def getDoc():
	res = []
	for r,d,files in os.walk(dirSrc):
		res = res + [f for f in files if f.endswith('.txt')]	
	return sorted(res)

def tkz(text):
	res = []
	for sent in nltk.sent_tokenize(text):
		for word in nltk.word_tokenize(sent):
			res.append(stemmer.stem(word))
	return res

if __name__=="__main__":
	docs = getDoc()
	print 'Doc List:',docs
	### calculate tf-idf matrix
	tfidf_vectorizer = TfidfVectorizer(input='filename',max_df=0.8,min_df=0.2,max_features=200000,stop_words='english',use_idf=True, tokenizer=tkz, ngram_range=(1,1))
	tfidf_matrix = tfidf_vectorizer.fit_transform([dirSrc+'/'+f for f in docs])
	print 'matrix shape',tfidf_matrix.shape
	### calculate the distance between documents
	dist = 1 - cosine_similarity(tfidf_matrix)
	#print dist
	### cluster documents via K-means
	km = KMeans(n_clusters=num_clusters)
	km.fit(tfidf_matrix)
	clusters = km.labels_.tolist()
	print 'Clustering through K-means:',clusters
	### hierachical clustering
	linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
	fig, ax = plt.subplots(figsize=(3, 4)) # set size
	ax = dendrogram(linkage_matrix, orientation="right", labels=docs);
	plt.tick_params(axis= 'x',which='both',bottom='off',top='off',labelbottom='off')
	plt.tight_layout() #show plot with tight layout
	plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
	print 'Hierachical Clustering'
	plt.show()

