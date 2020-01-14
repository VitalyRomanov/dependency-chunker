
import sys
import tensorflow as tf
# from gensim.models import Word2Vec
# import numpy as np
# from collections import Counter
# from scipy.linalg import toeplitz
# from gensim.models import KeyedVectors
from pprint import pprint

from data_reader import DataReader
from model import Chunker

BATCH_SIZE = 128
MAX_SEQ_LEN = 100
EPOCHS = 100


data_p = sys.argv[1]
test_p = sys.argv[2]

print("Initializing data_reader...", end="")
reader = DataReader(data_p,
                    test_p,
                    BATCH_SIZE=BATCH_SIZE,
                    MAX_SEQ_LEN=MAX_SEQ_LEN)
print("done")

print("Assembling model...", end="")
chunker = Chunker(MAX_SEQ_LEN=MAX_SEQ_LEN,
                  N_TOKEN_FEATURES=reader.token2id.size,
                  N_UNIQUE_TAGS=reader.label2id.size)
print("done")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # summary_writer = tf.summary.FileWriter("model/", graph=sess.graph)

    terminals = chunker.terminals

    hold_out = reader.test_data

    for e in range(EPOCHS):
        batches = reader.batches
        for ind, batch in enumerate(batches):
            sentences, pos_tags, lens = batch

            sess.run(terminals['train'], {
                    terminals['input']: sentences,
                    terminals['labels']: pos_tags,
                    terminals['lengths']: lens
                })
            print("Batch {}\r".format(ind), end="")

            # if ind % 10 == 0:
        sentences, pos_tags, lens = hold_out

        loss_val, acc_val, am = sess.run([terminals['loss'], terminals['accuracy'], terminals['argmax']], {
            terminals['input']: sentences,
            terminals['labels']: pos_tags,
            terminals['lengths']: lens
        })
        print("\nEpoch {}, loss {}, acc {}".format(e, loss_val, acc_val))

