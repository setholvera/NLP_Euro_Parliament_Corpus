# NLP_Euro_Parliament_Corpus
A NLP project using the European Parliament Proceedings Corpus

Used a bag of words model in identifying the 100 most common wordsq in each language dictionary

Normalized the test and training sets by eliminating punctuation and numerical characters using the built-in function isalpha to return strings containing only alphabetic characters

Implemented TF-IDF (Term Frequency-Inverse Document Frequency) model for language detection

Created contingency tables for each language and a confusion matrix for comparing the language dictionaries

Identified the 100 most common words in each language minimizing conflicts between languages, reducing noise between the training and test sets

Used the cleaned language dictionaries to produce new contingency tables and a confusion matrix. This strategy significantly reduced the misclassifications seen between French and Spanish previously. 

Implemented a language identification technique using n-grams. Detected bigrams within language dictionaries.

Europarl Corpus can be found here: https://www.statmt.org/europarl/
