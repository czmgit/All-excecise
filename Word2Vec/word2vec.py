import gensim,logging,os
import nltk
nltk.download("brown")
logging.basicConfig(format = "%(asctime)s:%(levelname)s:%(message)s",level = logging.INFO)
corpus = nltk.corpus.brown.sents()
fname = 'brown_skigram.model'
if os.path.exists(fname):
    model = gensim.models.Word2Vec.load(fname)
else:
    model = gensim.models.Word2Vec(corpus,size=100,min_count = 5,workers = 2,iter = 50)
    model.save(fname)

mywords = "The animal didn't cross the street because it was too tired".split()
for w1 in mywords:
    for w2 in mywords:
        print(w1,w2,model.similarity(w1,w2))
