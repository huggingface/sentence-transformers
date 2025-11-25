# CT

Carlsson et al. present in [Semantic Re-Tuning With Contrastive Tension (CT)](https://openreview.net/pdf?id=Ov_sMNau-PF) ([Github](https://github.com/FreddeFrallan/Contrastive-Tension)) an unsupervised learning approach for sentence embeddings that just requires sentences.

## Background

During training, CT builds two independent encoders ('Model1' and 'Model2') with initial parameters shared to encode a pair of sentences. If Model1 and Model2 encode the same sentence, then the dot-product of the two sentence embeddings should be large. If Model1 and Model2 encode different sentences, then their dot-product should be small.

The original CT paper uses batchs that contain multiple mini-batches. For the example of K=7, each mini-batch consists of sentence pairs (S_A, S_A), (S_A, S_B), (S_A, S_C), ..., (S_A, S_H) and the corresponding labels are 1, 0, 0, ..., 0. In other words, one identical pair of sentences is viewed as the positive example and other pairs of different sentences are viewed as the negative examples (i.e. 1 positive + K negative pairs). The training objective is the binary cross-entropy between the generated similarity scores and labels. This example is illustrated in the figure (from the Appendix A.1 of the CT paper) below:

![CT working](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/CT.jpg)

After training, the model 2 will be used for inference, which usually has better performance.

In **[CT_Improved](../CT_In-Batch_Negatives/README.md)** we propose an improvement to CT by using in-batch negative sampling.

## Performance

In some preliminary experiments, we compate performance on the STSbenchmark dataset (trained with 1 million sentences from Wikipedia) and on the paraphrase mining task for the Quora duplicate questions dataset (trained with questions from Quora).

| Method | STSb (Spearman) | Quora-Duplicate-Question (Avg. Precision) |
| --- | :---: | :---:
| CT | 75.7 | 36.5
| CT-Improved | 78.5 | 40.1

Note: We used the code provided in this repository, not the official code from the authors.

## CT from Sentences File

**[train_ct_from_file.py](train_ct_from_file.py)** loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

## Further Training Examples

- **[train_stsb_ct.py](train_stsb_ct.py)**: This example uses 1 million sentences from Wikipedia to train with CT. It evaluate the performance on the [STSbenchmark dataset](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark).
- **[train_askubuntu_ct.py](train_askubuntu_ct.py)**: This example trains on [AskUbuntu Questions dataset](https://github.com/taolei87/askubuntu), a dataset with questions from the AskUbuntu Stackexchange forum.

**Note:**
This is a re-implementation of CT within sentence-transformers. For the official CT code, see: [FreddeFrallan/Contrastive-Tension](https://github.com/FreddeFrallan/Contrastive-Tension)
