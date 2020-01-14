from Token2id import Token2id
from Label2id import Label2id
import tensorflow as tf
import numpy as np
#from sklearn.preprocessing import OneHotEncoder

def read_data(data_path):

    # sent_w stores list of token features
    # sent_c stores list of labels
    # sents_w stores list of sent_w
    sents_w = [];    sent_w = []
    sents_c = [];    sent_c = []

    with open(data_path, "r") as conll:
        for line in conll.read().strip().split("\n"):
            if line == '':
                sents_w.append(sent_w)
                sent_w = []

                sents_c.append(sent_c)
                sent_c = []
            else:
                token_info, chunk_tag = read_token(line)

                sent_w.append(token_info)
                sent_c.append(chunk_tag)

    return sents_w, sents_c


def to_sparse(sentences, max_seq_len, n_params):
    indices = []
    values = []

    for ind_1, content_1 in enumerate(sentences):
        for ind_2, content_2 in enumerate(content_1):
            for val in content_2:
                indices.append([ind_1, ind_2, val])
                values.append(1.0)

    batch_size = len(sentences)
    shape = np.array([batch_size, max_seq_len, n_params], dtype=np.int64)

    return tf.SparseTensor(indices=np.array(indices, dtype=np.int32),
                           values=np.array(values),
                           dense_shape=shape)


def convert(sentences, max_seq_len, n_params):

    indices = []

    for ind_1, content_1 in enumerate(sentences):
        for ind_2, content_2 in enumerate(content_1):
            if type(content_2) == list:
                for val in content_2:
                    indices.append([ind_1, ind_2, val])
            else:
                indices.append([ind_1, content_2])

    # arr_ind = np.array(indices, dtype=np.int32)

    if len(indices[0]) == 3:
        encoded_sentences = np.zeros((len(sentences), max_seq_len, n_params))
    elif len(indices[0]) == 2:
        encoded_sentences = np.zeros((len(sentences), max_seq_len))
    else:
        raise NotImplementedError("rank should be 2 or 3")

    arr_ind = np.array(indices, dtype=np.int32)

    if len(indices[0]) == 3:
        encoded_sentences[arr_ind[:,0], arr_ind[:,1], arr_ind[:,2]] = 1.
    else:
        encoded_sentences[arr_ind[:,0], arr_ind[:,1]] = 1.

    return encoded_sentences

def create_batches(batch_size, seq_len, sents, tags, token2id, label2id):
    # verify number of data samples
    assert len(sents) == len(tags)

    c_pos = 0

    b_sents = []
    b_tags = []
    b_lens = []

    while c_pos < len(sents):
        # verify length of each sentence matches length of labels
        assert len(sents[c_pos])==len(tags[c_pos])

        # used to truncate sentence if too long
        c_seq_len = min(seq_len, len(sents[c_pos]))

        # map token features to ids on the fly
        # saves memory, but wastes computation
        # allows to map new sentences
        b_sents.append(
            token2id.transform(
                [sents[c_pos]]
            )[0][:c_seq_len]
        )
        b_tags.append(
            label2id.transform(
                [tags[c_pos]]
            )[0][:c_seq_len]
        )
        b_lens.append(c_seq_len)

        c_pos += 1

        assert len(b_sents) == len(b_tags) == len(b_lens)


        if len(b_sents) == batch_size:
            # yield b_sents, b_tags, b_lens
            # print(b_tags)
            # print(np.array(b_tags, dtype=np.int32))
            yield convert(b_sents, seq_len, token2id.size), \
                  convert(b_tags, seq_len, label2id.size), \
                  np.array(b_lens, dtype=np.int32)
            b_sents = [];   b_tags = [];    b_lens = []

    if len(b_lens) != 0:
        yield convert(b_sents, seq_len, token2id.size), \
              convert(b_tags, seq_len, label2id.size), \
              np.array(b_lens, dtype=np.int32)


def read_token(line):
    parts = line.split()

    token = parts[0]  # token itself not included

    chunk_tag = parts[-1]

    pos = parts[1]
    position = parts[2]
    head = parts[3]
    dep_tag = parts[4]
    morph = parts[5:-1]
    if len(token) > 3:
        morph.append("<"+token[:2])
        morph.append(token[-2:]+">")
    else:
        morph.append("token")
    return (pos, position+"_pos", head+"_head", dep_tag, morph), chunk_tag
    # try addign head properties here
    # return (token, pos, position, head, dep_tag, morph), chunk_tag


class DataReader:
    def __init__(self,
                 train_data,
                 test_data,
                 BATCH_SIZE,
                 MAX_SEQ_LEN):
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_SEQ_LEN = MAX_SEQ_LEN

        train_sents, train_chunks = read_data(train_data)
        test_sents, test_chunks = read_data(test_data)

        self.train_sents = train_sents#[:1000]
        self.train_chunks = train_chunks#[:1000]
        self.test_sents = test_sents[:2000]
        self.test_chunks = test_chunks[:2000]

        # map every unique token feature to a unique id
        self.token2id = Token2id()
        # map every token label to a unique id
        self.label2id = Label2id()

        self.token2id.fit(train_sents)
        self.label2id.fit(train_chunks)

    @property
    def batches(self):
        return create_batches(self.BATCH_SIZE,
                              self.MAX_SEQ_LEN,
                              self.train_sents,
                              self.train_chunks,
                              self.token2id,
                              self.label2id)

    @property
    def test_data(self):
        return list(create_batches(len(self.test_sents),
                              self.MAX_SEQ_LEN,
                              self.test_sents,
                              self.test_chunks,
                              self.token2id,
                              self.label2id))[0]


    def map_new(self, sentence):
        c_seq_len = min(self.MAX_SEQ_LEN, len(sentence))


        return self.token2id.transform(
            [sentence]
        )[0][:c_seq_len]

    def __call__(self, sentence):
        return self.map_new(sentence)
