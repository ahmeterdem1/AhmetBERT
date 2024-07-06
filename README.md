# Introducing, AhmetBERT

A BERT model specifically designed for "AhmetGPT" purposes.

Yes, GPTs are decoder only. "AhmetGPT" is just a cool name.
It is not aimed to represent architectural features of the
actual thing.

## The Model

AhmetBERT, is a BERT variant that I have created just to have
an encoder that may represent prompts to a chatbot copy of myself.

AhmetBERT is a bidirectional transformer model of 6 transformer blocks,
each with 8 attention heads and 512 units for their FNNs. It is trained 
on all of the message data from defense blocks I have created for [AhmetGPT](https://github.com/ahmeterdem1/ahmetgpt).

BPE tokenizer with vocabulary size of 30000 is trained on all data gathered
from all defense blocks. "[START]", "[STOP]", "[PAD]", "[MASK]" and "[CLS]" special
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
We stop at 6, because 1/6 is very close to %15. Effective masking ratio over the
sequence length of 32 is probably lower than %15 because of the datas specific
nature given the circumstances.

All sequences are converted to length 32. For prompts, if they are shorter than 32
tokens, "[PAD]" tokens are appended to the beginning. Otherwise, tokens are popped
from beginning. "[PAD]", "[START]" and "[STOP]" tokens are never masked when applying 
masking. There is not a single instance of "[CLS]" token in the training and test data.
This model is designed to be a base model that can be later fine tuned for specific
purposes.

The model is trained for around 10 epochs, computation-wise ~2 GPU hours of L4.
A batch size of 32 is used. Masked sparse categorical cross entropy is used to
calculate the loss. Optimizer is chosen to be Adam instead of RMSprop this time.
The reason is, the dataset is not that big and models training is very slow.
Adam has momentum and RMSprop does not. Momentum increases the duration that 
model keeps learning after the dataset "fades out".


### AhmetBERT 1.0

Accuracy of this model on the test set reaches over %76. Accuracy is calculated over
the count of all masked tokens. The model is only trained to learn the masked tokens.

The model can be downloaded from [here](https://drive.google.com/file/d/1-jOd5pTU_RkIAqO92Fr2sLfLnwXatDv2/view?usp=share_link).
When loading the model, don't forget the import custom layer classes created here.
You will probably want to use Tensorflow 2.15 and Keras 2.15. These were the
versions that I have used to train the model with.

You can also find the tokenizer [here](https://drive.google.com/file/d/1xAnL0W9_TuMcS_Gcz4tH537DX0jk-tqo/view?usp=share_link)



