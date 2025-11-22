HINDI CHUNKING

1. Transformers for sequence classification

For this segment I mainly copy pasted the code from https://huggingface.co/learn/llm-course/chapter7/2#token-classification
I additionally used the following sources to understand the topic better:
    -> https://reybahl.medium.com/token-classification-in-python-with-huggingface-3fab73a6a20e
    -> 

2. Using BERT with our data

For this part I had to adapt the data processing step from the previous step to fit the Hindi dataset. To create my Hindi Dataset class I first explored the data further:
    -> https://github.com/UniversalDependencies/UD_Hindi-HDTB
    -> https://nlp.johnsnowlabs.com/models?task=Word+Segmentation
    -> https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/chinese/word_segmentation/words_segmenter_demo.ipynb
    -> https://medium.com/john-snow-labs/tokenizing-asian-texts-into-words-with-word-segmentation-models-42e04d8e03da
    ->https://github.com/EmilStenstrom/conllu
Based on code from step 1 and insights from above sources I developed the HindiDataset class. At this point I also considered whether to keep the 88 labels given in the data or translate them somehow so both the English and Hindi modelling would have the same labels (or at least the same numer of labels).
    -> https://github.com/cfiltnlp/HiNER/blob/main/README.md
However, as using the same labels for both datasets would mean using another dataset annotated differently I decided against it.

Then I had to find a model which worked for Hindi. Datasets I found included:
    -> https://huggingface.co/l3cube-pune/hindi-bert-v2
    -> https://huggingface.co/monsoon-nlp/hindi-bert
    -> https://huggingface.co/google-bert/bert-base-multilingual-cased
I chose the 'bert-base-multilingual-cased' model due to the integration simplicity as it is included in the pytorch library.
Then I modified the code from part one to work for both the English and Hindi dataset.

3. Performance Analysis

============================================================
RESULTS SUMMARY: English NER (BERT)

============================================================

 ENTITY-LEVEL METRICS (Seqeval):
  Precision:  0.8798
  Recall:     0.9132
  F1-Score:   0.8962
  Accuracy:   0.9719

 TOKEN-LEVEL METRICS:
  Accuracy:             0.9719
  Precision (macro):    0.8851
  Recall (macro):       0.9025
  F1-Score (macro):     0.8936
  Precision (weighted): 0.9728
  Recall (weighted):    0.9719
  F1-Score (weighted):  0.9723

 PER-CLASS PERFORMANCE (Top 10 by support):
----------------------------------------------------------------------
Class                 Precision     Recall         F1    Support
----------------------------------------------------------------------
O                        0.9940     0.9870     0.9905      47925
I-PER                    0.9620     0.9748     0.9684       4082
I-ORG                    0.9047     0.9429     0.9234       3172
I-LOC                    0.9129     0.9170     0.9150       1748
B-LOC                    0.9388     0.9376     0.9382       1668
B-ORG                    0.9045     0.9175     0.9109       1661
B-PER                    0.9582     0.9629     0.9605       1617
I-MISC                   0.5938     0.6467     0.6191        886
B-MISC                   0.7976     0.8362     0.8164        702

============================================================
RESULTS SUMMARY: Hindi Chunking (DistilBERT)

============================================================

 ENTITY-LEVEL METRICS (Seqeval):
  Precision:  0.0003
  Recall:     0.0009
  F1-Score:   0.0004
  Accuracy:   0.0043

 TOKEN-LEVEL METRICS:
  Accuracy:             0.0043
  Precision (macro):    0.0041
  Recall (macro):       0.0021
  F1-Score (macro):     0.0017
  Precision (weighted): 0.0149
  Recall (weighted):    0.0043
  F1-Score (weighted):  0.0041

 PER-CLASS PERFORMANCE (Top 10 by support):
----------------------------------------------------------------------
Class                 Precision     Recall         F1    Support
----------------------------------------------------------------------
NP15                     0.0000     0.0000     0.0000      11462
JJP3                     0.0000     0.0000     0.0000      11281
NP18                     0.0000     0.0000     0.0000      11097
NP19                     0.0600     0.0062     0.0112       9554
NP2                      0.0583     0.0024     0.0047       7781
NP26                     0.0000     0.0000     0.0000       6865
NP20                     0.0000     0.0000     0.0000       5736
NP21                     0.0000     0.0000     0.0000       4388
NP35                     0.0089     0.0006     0.0010       3588
NP22                     0.0268     0.0690     0.0386       3302

As can be seen in the results above (and in in the documents in the evaluation_results directory) the English model performs very well and the hindi model performs catastroffically. For the 'O' and person entities labels the English model is close to perfect, with a F1-score of ~0.99 and ~0.97/0.96 respectively. The great metrics for 'O' is not supprising considering the numer of entries of said class in the data compared to the other classes. The miscellaneous labels for the English model are the metrics the model struggles to identify the most, which is understandable as the 'MISC' category in and of itself is vague. 
In contrast to the English model, the Hindi model performs badly. Considering the Hindi dataset had 88 classes compared to 9 for the English model it is expected that the Hindi model performs worse due to the increased complexity of the task. However, the Hindi model does not just perform worse than the English model, it also performs worse than if it would have just been randomly guessing the classes. Ergo, the Hindi model did not learn anything (which the loss score also shows). This might be due to the few training epochs (only 3) -- an increase in training epochs might result in better performance -- but I highly doubt it as the loss did not improve even with 10 epochs. Hence, the poor performance might be due to the model chosen for the task. Unlike the model for the English dataset, the model chosen for Hindi is not monolingual -- it is multilingual and might not have seen enough Hindi data to perform well on this task. Out of the models I mentioned above 'monsoon-nlp/hindi-bert' might have been a better fit as it is a monolingual BERT model and not multilingual. 
Why the Hindi model predicted as it did I am unsure, as it even performed poorly on the classes with the most instances -- my first guess being that it would predict the most frequent classes the most. 
To better compare the two models some form of normalization would need to take place due to the difference in complexity of the task. However, with the Hindi model basically learning nothing as is I would suggest exploring other model options to find a model which actually lerns something before conisdering normalization.