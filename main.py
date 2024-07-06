from keras import Sequential, layers, optimizers
from transformer_blocks import TransformerBlock
from positional_embeddings import PositionalEncoding
from loss import masked_sparse_categorical_crossentropy

sequence_length = 32
vocab_size = 30000
train_limit = 81822 - 81822 // 10
embedding_dim = 64
batch_size = 32

loss = masked_sparse_categorical_crossentropy
optimizer = optimizers.Adam()

model = Sequential([
    layers.Input((sequence_length,)),
    layers.Embedding(vocab_size, embedding_dim),
    PositionalEncoding(sequence_length, embedding_dim),

    TransformerBlock(units=512, activation="relu", num_heads=8, sequence_length=sequence_length, key_dim=embedding_dim, value_dim=embedding_dim),
    TransformerBlock(units=512, activation="relu", num_heads=8, sequence_length=sequence_length, key_dim=embedding_dim, value_dim=embedding_dim),
    TransformerBlock(units=512, activation="relu", num_heads=8, sequence_length=sequence_length, key_dim=embedding_dim, value_dim=embedding_dim),

    TransformerBlock(units=512, activation="relu", num_heads=8, sequence_length=sequence_length, key_dim=embedding_dim, value_dim=embedding_dim),
    TransformerBlock(units=512, activation="relu", num_heads=8, sequence_length=sequence_length, key_dim=embedding_dim, value_dim=embedding_dim),
    TransformerBlock(units=512, activation="relu", num_heads=8, sequence_length=sequence_length, key_dim=embedding_dim, value_dim=embedding_dim),

    layers.Dense(vocab_size, activation="softmax")
])

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model.summary()


