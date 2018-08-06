---
layout: post
title:  "Topic Modeling for DEEP"
date:   2018-08-05 12:22:59 +0545
categories: deep nlp
---

### Topic modeling
Topic modeling is a part of Natural Language Processing which helps in finding out hidden topics present in the documents, as group of words. Topic modeling gives insights about what the documents are composed of
and which words compose the topics. In short, we can understand and summarize large collection of textual data using topic modeling.  

### Topic modeling in DEEP
There are a lot of textual data collected in DEEP by various analysts around the world. Although all such texts and documents are humanitarian or crisis related, we are always interested in and in need of more
precise insight of the documents. Basically, we want to find the hierarchical composition of doucments collected.  

The library that we used for this purpose is `gensim` which is very popular for topic modeling. We chose `gensim` because of its elegant implementation of algorithms and simplicity. The following is the sample 
code for performing topic modeling.  
```python
from gensim import models


def find_topics(documents, num_topics):
    """
    Return the keywords for topics discovered
    @documents: documents to be analyzed
    @num_topics: number of topics we wish to find
    """
    texts = [
        pre_processor(document).split() for document in documents
    ]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = models.ldamodel.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes
        )
    lda_output = {}
    topics = ldamodel.get_topics()
    for i, topic_words in enumerate(topics):
        # sort keywords based on their weights/contributions to the topic
        sorted_words = sorted(
            list(enumerate(topic_words)),
            key=lambda x: x[1],
            reverse=True
        )
        lda_output['Topic {}'.format(i)] = {
            'keywords': [
                (dictionary.get(i), x)
                for i, x in sorted_words[:num_words]
            ]
        }
    return lda_output
```
Now, given some documents, the result of running the above function would result in output similar to the following:
<pre>
{
    "Topic 1": {
        "keywords": ["kw11, "kw12", ...]
    },
    "Topic 2": {
        "keywords": ["kw21, "kw22", ...]
    }, ...
}
</pre>

This gives the topics present in the documents as important keywords. One thing that can be noticed here is that the topics returned are not the exact topics, we do not have name for the topics. We needed the 
exact topic names for our application. We tackled this problem by a very simple but good-to-go approach: taking the first most relevant keyword for the topic.  


### Hierarchical Topic Modeling
The next problem with the above result is that we are also interested in hierarchical composition of the topics in the documents but this just gives a flat topics composition. Unfortunately, we could not find any library that fitted 
our purpose of hierarchical modeling. And, we applied another simple approach to achieve the hierarchy: **recursion**.  

Our idea about achieving hierarchical topic modeling consists of the following steps:
1. Perform topic modeling on the documents to get initial topics composition.
2. For each topic, group all the documents belonging to the topic.
3. For each topic, run the topic modeling algorithm on the grouped documents to obtain subtopics composition for the topic.
4. Repeat the process from step 2 until the desired hierarchy level is obtained.

The following is the simplified code for the above steps.
```python
def find_topics_and_subtopics(documents, num_topics, depth=5, dictionary=None, corpus=None):
    """
    Return the keywords for topics discovered
    @documents: documents to be analyzed
    @num_topics: number of topics we wish to find
    @depth: depth of hierarchy
    """
    texts = [
        pre_processor(document).split() for document in documents
    ]
    if dictionary is None:
        dictionary = corpora.Dictionary(texts)
    if corpus is None:
        corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = models.ldamodel.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes
        )
    lda_output = {}
    topics = ldamodel.get_topics()
    for i, topic_words in enumerate(topics):
        # sort keywords based on their weights/contributions to the topic
        sorted_words = sorted(
            list(enumerate(topic_words)),
            key=lambda x: x[1],
            reverse=True
        )
        lda_output['Topic {}'.format(i)] = {
            'keywords': [
                (dictionary.get(i), x)
                for i, x in sorted_words[:num_words]
            ]
            'subtopics': {}
        }
    if depth <= 1:
        return lda_output

    # Group documents
    topic_documents = {}
    for j, corp in enumerate(corpus):
        topics_prob = sorted(
            ldamodel.get_document_topics(corp),
            key=lambda x: x[1]
        )
        for i, topic in enumerate(topics_prob):
            topic = 'Topic {}'.format(i)
            curr_set = topic_documents.get(topic, set())
            curr_set.add(j)
            topic_documents[topic] = curr_set

    # Now the recursive call
    for topic in lda_output:
        if topic_documents.get(topic):
            topic_docs = [corpus[x] for x in topic_documents[topic]]
            lda_output[topic]['subtopics'] = find_topics_and_subtopics(
                topic_docs,
                num_topics,
                depth-1,
                dictionary,
                corpus
            )
    return lda_output
```
The result of running the above function would result in output similar to the following:
<pre>
{
    "Topic 1": {
        "keywords": ["kw11, "kw12", ...],
        "subtopics": {
            "Topic 1": {
                "keywords": ["kw1.11", "kw1.12", ... ],
                "subtopics": { ... }
            }, ...
        }
    },
    "Topic 2": {
        "keywords": ["kw21, "kw22", ...],
        "subtopics": {
            "Topic 1": {
                "keywords": ["kw2.21", "kw2.22", ... ],
                "subtopics": { ... }
            }, ...
        }
    }, ...
}
</pre>


### The Problems and Possible Solutions
The following are the problems that we've faced:
- The first and the foremost, `gensim` has implemented **Latent Dirichlet Allocation(LDA)** which is a statistical method which therefore does random initialization of topics and keywords. So, if we don't do more 
iterations, running the algorithm multiple times with same documents result in quite different results.
- Although more iteration means more stable and predictable output, it causes the algorithm to run slow for large documents. Finding a balance between these is a challenge and completely experimental. It also 
depends on the kinds of documents we have.
- Also, since we've introduced recursive solution for hierarchical modeling, the complexity grows exponentially with depths of topics. And with large documents, it becomes very slow.  

The first two problems are inherent in the algorithm itself and is just a matter of finding out the balance between stable results and faster calculation. However, for the third problem mentioned above, we
've implemented caching in server side to prevent running the algorithm multiple times if the documents have not changed.
