## Neural Machine Translation with Encoder-Decoder and Attention

### Summary

Here we implemented Seq-2-Seq Encoder-Decoder mechanism with Attention to Machine Tranlation Task. The goal is to translate English to French and use cross-entropy as loss function. At the end we calculate BLEU “BiLingual Evaluation Understudy” score of our model for evaluation.

### Model Architecture

**Encoder**
For Encoder we used an Embedding Size of 64 and Bidirectional GRU unit. The goal of the Encoder is to create a great representation of the input and create a fixed size context vector.

**Decoder**
The Decoder is made out of Embedding Vector of size 64 and GRU cell. The decoder’s RNN model takes in the context vector and generates an output.  The context vector which at each time step is the sum of hidden states of input sentence weighted by **alignment scores**.

In training for each time step it looks at the real target value meaning it has a teacher at every time step that show the real value but we don't have this option on testing.


### Training
For training we used eng-french simple dataset with 100k examples. Trained it on Nvidia RTX 2070 for 2 hours for 100 epochs. We also implemented early stopping when the loss is not decreasing anymore.

**Results**  
