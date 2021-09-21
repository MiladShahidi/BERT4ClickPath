# Clickstream-Transformer

Clickstream-Transformer is a Python library that has emerged out of the various models I have built around online
shopping use cases. It implements a configurable Transformer-based model for clickstream data and can be easily
configured and trained for a wide range of tasks. The following are some examples:

- Product recommendation (See below)

    - Given a sequence of items, the model can learn to recommend related products to the user.

- Predicting return of purchased products

    - In this setup, the model can predict whether purchased products will be returned, given the user's behaviour
    during the online shopping session.  

- Predicting purchase intention

    - The model can also be configured to predict whether an ongoing online shopping session will result in a purchase. 

The architecture and implementation of the model is inspired by BERT, the pioneer NLP model.

In its simplest configuration, the model takes in a sequence of items that the user has interacted with
(e.g. clicked on) as its input. An additional "Head Unit" can be used to further process the output of the Transformer
for the specific task at hand. This architecture facilitates transfer learning from one task to another. Once the model
is trained for, say recommendation, it can quickly be fine-tuned to perform another task on the same set of products.

![alt text](https://github.com/MiladShahidi/Clickstream-Transformer/blob/master/doc/images/Clickstream%20Transformer.svg "Clickstream-Transformer")

Alternatively, it can also accept multiple variables per sequence. For example, the
clickstream data might consist of (action, item) pairs, such as ("view", "gloves") or ("size change", "shoe"). In this
case, it learns representations (embeddings) of both actions and items resulting in a richer model of user behaviour.

# Example: Clickstream-Transformer as A Sequential Recommendation Model

The repository also provides a demonstration of how this model can be configured and trained as a sequential product
recommendation model. This is implemented under `examples`. As the table below shows, it outperforms some of the
best-known recommendation algorithms.

[Table here]

Note: Amazon Beauty data and performance figures for other models are from Sun et. al (2019) that introduces BERT4Rec.

Following ideas from the NLP literature, and BERT in particular, the model is trained using the Cloze task (aka Masked
Language Model). More specifically a random subset of each sequence is masked and the task is to predict what those
items were. This results in a model of user's sequential behaviour given the context of a particular session and enables
it to make next item recommendations.