# PubMedTranslator
## Server

Listens for requests from the client and sends the translated request.

The model is a sequence to sequence model with a global attention layer.
Inputs are tokenized and represented as a vector of word identifiers. Outputs are represented by a one hot vector for each word. Before the outputs are sent, they are converted to a human-readable, space-separated tokens.