# Introducing, AhmetBERT

A BERT model specifically designed for "AhmetGPT" purposes.

Yes, GPTs are decoder only. "AhmetGPT" is just a cool name.
It is not aimed to represent architectural features of the
actual thing.

# The Model

AhmetBERT, is a BERT variant that I have created just to have
an encoder that may represent prompts to a chatbot copy of myself.

AhmetBERT is a bidirectional transformer model of 6 transformer blocks,
each with 8 attention heads and 512 units for their FNNs. It is trained 
on all of the message data from defense blocks I have created for [AhmetGPT](https://github.com/ahmeterdem1/ahmetgpt).

BPE tokenizer with vocabulary size of 30000 is trained on all data gathered
from all defense blocks. "[START]", "[STOP]", "[PAD]" and "[MASK]" special
tokens are added to the vocabulary. Whitespace pre-tokenizer is used alongside
BPE.

Embeddings are learned through the BERT training.

Prompts are counted as the collection of messages in a defense block that
I have received from others. Masked language modeling is applied with the
prompts. ~%15 of prompts are masked with "[MASK]" special token. But, there 
is a problem with that.

Most of the messages are short. It is impossible to mask %15 of a message
that is 2 tokens or less. Special cases are created for these situations
that favors masking some parts of the message. 

From length 2 to 6, there is ~%50 probability that one of the tokens get masked.
We stop at 6, because 1/6 is very close to %15.

All sequences are converted to length 32. For prompts, if they are shorter than 32
tokens, "[PAD]" tokens are appended to the beginning. Otherwise, tokens are popped
from beginning. "[PAD]", "[START]" and "[STOP]" tokens are never masked when applying 
masking.

A total of, 1, epochs are given to the training. Exactly. Around 10 GPU minutes of A100.

The collection of all the data is not enough to train a BERT. After 1 epoch, the
model reaches its maximum performance, and does not improve any more no matter
the epoch count. BERTs require a huge amount of data to be trained. WhatsApp
messages are not enough on their own. 

The model reaches %35 accuracy on the test set. Test set in our case is very
much powerful, for testing purposes. It is the *chronologically* last %10 of
all messages. Stacked LSTMs' performance tops at %87.8 with the same data.
I will publish the LSTM-only encoder soon, and probably, I will build the
decoder on top of it. Not this BERT. 

The model can be downloaded from [here](https://drive.google.com/file/d/1--wx6RgENX2OLjwdKjW7hOjbBzUvms0I/view?usp=sharing).
When loading the model, don't forget the import custom layer classes created here.
You will probably want to use Tensorflow 2.15 and Keras 2.15. These were the
versions that I have used to train the model with.



