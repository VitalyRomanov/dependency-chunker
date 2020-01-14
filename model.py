import tensorflow as tf

default_params = {
    "d_win": 7,
    "param_dim": 500,
    "h1_dim": 200,
    "h2_dim": 100
}

def assemble_model(seq_len,
                   n_params,
                   n_tags,
                   lr=0.001,
                   train_embeddings=False,
                   parameters=default_params):

    d_win = default_params["d_win"]
    param_dim = default_params["param_dim"]
    h1_dim = default_params["h1_dim"]
    h2_dim = default_params["h2_dim"]

    kernel_shape = (d_win, param_dim)

    with tf.variable_scope("Inputs") as inputs:
        # input_params = tf.sparse.placeholder(dtype=tf.float32,
        #                                      shape=(None, seq_len, n_params),
        #                                      name="input")
        input_params = tf.placeholder(dtype=tf.float32,
                                         shape=(None, seq_len, n_params),
                                         name="input")

        output_labels = tf.placeholder(dtype=tf.int32,
                                       shape=(None, seq_len),
                                       name="out_labels")

        lengths = tf.placeholder(shape=(None,),
                                 dtype=tf.int32,
                                 name="lengths")

    with tf.variable_scope("Embedding") as embedding:
        # reshape input tensor (None, seq_len, n_params) to
        # (None * seq_len, n_params)
        # input_unrolled = tf.sparse.reshape(input_params,
        #                                    shape=(-1, n_params),
        #                                    name="inputs_unrolled")
        input_unrolled = tf.reshape(input_params,
                                           shape=(-1, n_params),
                                           name="inputs_unrolled")

        # create embedding for every token feature
        # param_emb = tf.get_variable("W_param",
        #                             shape=(n_params, param_dim),
        #                             dtype=tf.float32)

        # Embed every token with a non-linearity
        # words_embedded_unrolled = tf.nn.relu(
        #     tf.sparse_tensor_dense_matmul(input_unrolled,
        #                                   param_emb)
        # )
        # words_embedded_unrolled = tf.nn.relu(
        #     tf.matmul(input_unrolled,
        #               param_emb)
        # )
        words_embedded_unrolled = tf.layers.dense(input_unrolled,
                                                    param_dim,
                                                    activation=None,
                                                    name="non_lin_emb")

        # reshape back to (None * seq_len, n_params)
        words_embedded = tf.reshape(words_embedded_unrolled, shape=(-1,seq_len, param_dim), name="embedded_sents")



    def convolutional_layer(input, units, cnn_kernel_shape, activation=None):
        with tf.variable_scope("Convolutional_Layer") as cl:
            # pad according to window size
            padded = tf.pad(input,
                            tf.constant([[0, 0], [d_win//2, d_win//2], [0, 0]])
                            )
            # add axis where convolutional channels go
            emb_sent_exp = tf.expand_dims(padded, axis=3)

            convolve = tf.layers.conv2d(emb_sent_exp,
                                        units,
                                        cnn_kernel_shape,
                                        activation=activation,
                                        data_format='channels_last',
                                        name="conv_h1")
            # CNN layer reduced dimension 2 to the size of 1
            # remove this dimension with reshape
            cnn_features = tf.reshape(convolve, shape=(-1, convolve.shape[1], units))
        return cnn_features


    conv_h1 = convolutional_layer(input=words_embedded,
                                  units=h1_dim,
                                  cnn_kernel_shape=kernel_shape,
                                  activation=tf.nn.relu)

    with tf.variable_scope("Non-linearity") as nl:
        # unroll to 2d tensor to apply nn
        token_features_1 = tf.reshape(conv_h1, shape=(-1, h1_dim))
        # apply first non-linearity
        local_h2 = tf.layers.dense(token_features_1,
                                   h2_dim,
                                   activation=tf.nn.tanh,
                                   name="dense_h2")
        # apply second non-linearity
        tag_logits = tf.layers.dense(local_h2,
                                     n_tags,
                                     activation=None,
                                     name="unrolled_logits")
        # reshape back to 3d tensor
        logits = tf.reshape(tag_logits, (-1, seq_len, n_tags))

    with tf.variable_scope('loss') as l:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, output_labels, lengths)
        loss = tf.reduce_mean(-log_likelihood)

    train = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.variable_scope("Accuracy") as acc:
        mask = tf.sequence_mask(lengths, seq_len)
        true_labels = tf.boolean_mask(output_labels, mask)
        argmax = tf.math.argmax(logits, axis=-1)
        estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)
        accuracy = tf.contrib.metrics.accuracy(estimated_labels, true_labels)

    return {
        'input': input_params,
        'labels': output_labels,
        'lengths': lengths,
        'loss': loss,
        'train': train,
        'accuracy': accuracy,
        'argmax': argmax
    }

class Chunker:
    def __init__(self,
                 MAX_SEQ_LEN,
                 N_TOKEN_FEATURES,
                 N_UNIQUE_TAGS):
        self.terminals = assemble_model(seq_len=MAX_SEQ_LEN,
                                   n_params=N_TOKEN_FEATURES,
                                   n_tags=N_UNIQUE_TAGS)
