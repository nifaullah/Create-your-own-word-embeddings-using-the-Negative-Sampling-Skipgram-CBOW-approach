# Create-My-Own-Word-Embeddings

In this project I am trying to create my own word embeddings.

<b>Goal:</b> The main goal of this project is to understand how word embeddings can be developed using different development methods.

<b>Data:</b> Although the methods developed here are generic and can be used with any corpus, I have used the <a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie Review</a> dataset from Stanford used for this analysis.

<b>Approach:</b> First I have put the entire dataset in the dataframe with 2 columns Review & Sentiment. I am seprately using another file to deal with all the 50K textfiles and put each of this reviews as row in the dataframe. In total I have 2 dataframes test & train. I only used the train dataset to develop the embeddings. First I cleaned the dataset and created a corpus from the dataframe. Using this corpus first I created a ContexTarget pairs for Skipgram methods, only to later realize the enormity of having to do multiple softmax regression for millions of times. I then decided based on the original paper I went ahead with Negative Sampling.

<b>Scope: </b> All though the I have mainly used negative sampling approach due to memory limitations in my personal computer but I have still added the code for Skipgram(Not tested), if you have resources to develop embeddings using skipgram approach please go ahead and use the code and if you want to use the Continuous Bag of words approach, for data you can use the skipgram code, but the data & label would interchange.

<b>Result: </b> Currently I am still training the model to develop the word embeddings, I'll update the files as I move forward.

<b>Citation: </b> I have used following resources to develop word embeddings

1.  <a href="https://ai.stanford.edu/~amaas/papers/wvSent_acl2011">Learning Word Vectors for Sentiment Analysis</a>
2.  <a href="https://arxiv.org/abs/1301.3781">Efficient Estimation of Word Representations in Vector Space</a>
2.  <a href="https://www.coursera.org/specializations/deep-learning">Deep Learning Specialization</a>
4.  <a href="https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72">Word2vec from Scratch with NumPy</a>
