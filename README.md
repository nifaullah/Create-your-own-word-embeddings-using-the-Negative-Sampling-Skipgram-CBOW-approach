# Create-Your-Own-Word-Embeddings
Create your own word embeddings using the Negative Sampling, Skipgram & CBOW approach.

In this project I am trying to create my own word embeddings provided a corpus or a dataset

<b>Goal:</b> The main goal of this project is to understand how word embeddings can be developed using different development methods and to develop generic methods which can be used to build locally contextualized word embeddings. The best way to test if you've learned something properly or not, is to implement the ideas learned and with this excercise I am trying to improve my understanding of word embeddings.

<b>Data:</b> Although the methods developed here are generic and can be used with any corpus, I have used the <a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie Review</a> dataset from Stanford used for demonstration purposes.

<b>Approach:</b> First I have put the entire dataset in the dataframe with 2 columns Review & Sentiment. I am seprately using another file(IMDB_utils) to deal with all the 50K textfiles and put each of this reviews as row in the dataframe. In total I have 2 dataframes test & train. I only used the train dataset to develop the embeddings. First I cleaned the dataset and created a corpus from the dataframe. Using this corpus first I created a ContexTarget pairs for Skipgram methods, only to later realize the enormity of having to do multiple softmax regression for millions of times. I then decided, based on the original paper, succesfully proceeded with Negative Sampling.

<b>Scope: </b> All though the I have mainly used negative sampling approach due to memory limitations in my personal computer but I have still added the code for Skipgram(Not tested), if you have resources to develop embeddings using skipgram approach please go ahead and use the code and if you want to use the Continuous Bag of words approach, for data you can use the skipgram code, but the data & label would interchange. Additionally I have provided detailed description above each method.

<b>Results: </b> Currentl, I am able to derive word vectors from the model but I have only done basic manual testing to complete few basic test cases. Learninng wise this project has been a revelation on many fronts, I am not sure if I ever fully read a paper before, this excercised has revealed many faulty approaches to learning I have developed and if any one wants to understand the concept substantially to try building the word embeddings on their own.

<b>Future Developments: </b> First I want to create a test utils where I will build all the automated test methods which would generically test the embeddings irrespective of the corpus. Additionally I want to do a analysis comparing the results from this word embeddings with embeddinngs from the famous Gensim library.

<b>Citation: </b> I have used following resources to develop the word embeddings

1.  <a href="https://ai.stanford.edu/~amaas/papers/wvSent_acl2011">Learning Word Vectors for Sentiment Analysis</a>
2.  <a href="https://arxiv.org/abs/1301.3781">Efficient Estimation of Word Representations in Vector Space</a>
2.  <a href="https://www.coursera.org/specializations/deep-learning">Deep Learning Specialization</a>
4.  <a href="https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72">Word2vec from Scratch with NumPy</a>
