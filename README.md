# Book-Rating-Estimation

## Introduction

In this project I have decided to use two methods of unsupervised learning - spectral and dbscan - in order to estimate books ratings, based on their reviews. The dataset can be found [https://www.kaggle.com/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis](here). Each lines contains an ID, product ID, reviewer ID, reviewer name, review time, review text, review summary, review helpfulness and product rating. The product rating represents the clusters we want our data to be separated in, so it will only be used for estimating each model's performance. From the other columns, I decided to train each model only on the preprocessed review texts.

## Preprocessing

Before testing the unsupervised models I tried finetuning the preprocessing on a supervised model - SVC -, which will later be used as the supervised baseline. I first tested the SVC with tfidf using the original text phrases, in order to have a start point for comparison. The results are as follows: 0.847 on train and 0.494 on test.

The embeddings were trained similarly to the first project: TFIDF was trained only on the train texts, while word2vec was trained on a merge between these texts and nltk.corpus.brown.

After that I removed all newline characters, digits and extra spaces and separated the text into lowercase tokens. This got the following results:

punctuation | stopwords | lemmatized | no. features | train score | test score
:---: | :---: | :---: | :---: | :---: | :---:
no | no | no | 500 | 0.845 | 0.493
no | yes | no | 500 | 0.845 | 0.493
yes | no | no | 500 | 0.866 | 0.477
yes | no | yes | 500 | 0.866 | 0.474
yes | no | no | 100 | 0.693 | 0.415
yes | no | no | 300 | 0.808 | 0.467
yes | yes | no | 500 | 0.845 | 0.494
yes | no | no | 1000 | **0.889** | **0.507**

We can see that the best preprocessing has punctuation, no stopwords, and the words are not lemmatized. For word2vec I also removed punctuation and stop words, which gave a train score of 0.458 and test score of 0.448. This will be later used for the unsupervised models.

## Models

### Spectral

Spectral only requires the number of clusters.

Eigen\_solver \ assign\_labels | kmeans | discretize
:---: | :---: | :---:
arpack | 0.217 | 0.173
amg | 0.168 | 0.15
lobpcg | 0.158 | 0.167

We can see that we get the best with kmeans and  arpack.

By checking one clusterization we can see that one cluster groups romantic books, the second one has opinions about the book's structure and references to other reviews, the third one has book synopsis, usually drama, the fourth one contains short reviews and the fifth one contains reviews with clear feelings towards the book.

### DBSCAN

data \ epsilon | 0.1 | 0.21 | 0.25 | 0.29 | 0.32 | 0.36 | 0.4 | 0.44 | 0.47 | 0.52 | 0.55 | 0.587 | 0.625 | 0.662 | 0.7 | 0.77 | 0.81 | 0.85 | 0.88 | 0.92 | 0.96 | 1.0
:--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
TFIDF | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 5 | 7 | 10 | 12 | 7 | 4 | 15
Word2vec | 0 | 1 | 10 | 9 | 8 | 5 | 8 | 2 | 7 | 5 | 3 | 6 | 5 | 5 | 3 | 1 | 1 | 3 | 3 | 2 | 3 | 2

By applying DBSCAN only with the epsilons that give 5 clusters (from this data), we'll see that TFIDF only has one option (epsilon 0.77) and it has the accuracy 0.257. For word2vec we have 0.36, 0.52, 0.625 and 0.662 which give 0.257, 0.255, 0.26 and 0.259 respectively. So, the best accuracy is found using word2vec with epsilon 0.662.

After looking at the dataset it is obvious that the unsupervised clusterization has no connection to the reviewers' sentiments, but rather to the main theme of the book. TFIDF with 5 clusters for example groups in the first cluster reviews about fairytales, in the second one reviews about mature subjects, the third one has violent themes, the fourth one - books about John - and the fifth one is about science fiction. While this is an interesting classification by theme, it was not the scope of this project.

## Conclusion

It seems the preprocessing was not appropriate for the task at hand. More attention should have been paid to which elements showcase the reader's attitude towards a book, rather than theme.

The dataset itself wasn't perfect for the task either, since we can find the following review that has rating 3:
"Seriously a throwback to the old 1960's-70's Harlequin romances set in Europe. Obnoxious, autocratic male lead, weak thoughtless female who has every intention of standing up to Mr. Obnoxious but her big moments seem to fizzle. The more obnoxious he is towards her, the more she gets drawn in until she falls in love with the jerk. This is more like a case of Stockholm Syndrome. Please get this author a proof-reader and an editor before she publishes anything else!"


Although the accuracy was somehow better with DBSCAN, looking at the clusters makes it clear that spectral was closer to the actual scope. Spectral also had less outliers. It was expected to work better, since it is made for medium datasets with a small number of clusters, while DBSCAN is made for very large datasets with a medium number of clusters.
