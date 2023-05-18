# SPGNN
Sequence Pre-training-based Graph Neural Network (SPGNN) is a novel graph-based pre-training scheme that offers a promising approach to discover associations between lncRNAs and miRNAs. SPGNN utilizes a sequence-to-vector technique to generate pre-trained embeddings based on the sequences of all RNAs during the pre-training stage. In the fine-tuning stage, SPGNN uses Graph Neural Network to learn node representations from the heterogeneous graph constructed using lncRNA-miRNA association information. The source code of the proposed scheme is available on this Github repository.

![A detailed overview of our proposed \ours{} scheme which includes the $k$-mers processing, pre-training and fine-tuning stages](/images/cerna.pdf)

## To train and test SPGNN
`python main.py`

