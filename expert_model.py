# coding=utf-8
# email: interfk@gmail.com
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import abc
import collections
import math
import time
import codecs
import sys
import argparse
import os
import random
import six
import json
import numpy as np
import tensorflow as tf
import opennmt as onmt

from tensorflow.python.ops import lookup_ops
from tensorflow.python.client import device_lib

import re

_DIGIT_RE = re.compile(r"\d")

############################## Vocab Utils ##########################################
BLK = "<blank>"
SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
VOCAB_SIZE_THRESHOLD_CPU = 50000


def print_out(s, f=None, new_line=True):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)
    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab, vocab_size):
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=vocab_size)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_value=vocab_size)
    return src_vocab_table, tgt_vocab_table


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, blk=None, data_file=None, max_vocabulary_size=50000):
    """Check if vocab_file doesn't exist, create from corpus_file."""
    if tf.gfile.Exists(vocab_file):
        print_out("# Vocab file %s exists" % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            if not blk: blk = BLK
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != blk or vocab[1] != sos or vocab[2] != eos:
                print_out("The first 3 vocab words [%s, %s, %s]"
                          " are not [%s, %s, %s]" %
                          (vocab[0], vocab[1], vocab[2], blk, sos, eos))
                vocab = [blk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                    tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = new_vocab_file
    else:
        vocab, vocab_file = create_vocabulary(vocab_file, data_file, max_vocabulary_size)

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, normalize_digits=False):
    print_out("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with tf.gfile.GFile(data_path, mode="rb") as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            line = tf.compat.as_bytes(line)
            tokens = line.split()
            for w in tokens:
                word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = [BLK, SOS, EOS] + sorted(vocab, key=vocab.get, reverse=True) + [UNK]
        if len(vocab_list) > max_vocabulary_size + 1:
            vocab_list = vocab_list[:max_vocabulary_size + 1]
        with tf.gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write("%s\n" % w)
        return vocab_list, vocabulary_path


############################## Iterator #######################################
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer",
                            "qe_iterator"))):
    pass


def get_infer_iterator_exp(src_dataset,
                           tgt_dataset,
                           src_vocab_table,
                           tgt_vocab_table,
                           batch_size,
                           sos,
                           eos,
                           src_max_len=None,
                           tgt_max_len=None):
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    qe_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values,
            tf.string_split([tgt]).values))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # mt
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # mt_len
            padding_values=(
                0,  # src_eos_id,  # src
                0,  # tgt_eos_id,  # tgt_input
                0,  # qe_len -- unused
                0))  # mt_len -- unused

    if src_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt))
    if tgt_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]))

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            src,
            tf.concat(([tgt_sos_id], tgt, [tgt_eos_id]), 0)))

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            src,
            tgt,
            tf.size(src),
            tf.size(tgt)))

    qe_batched_dataset = batching_func(qe_dataset)
    qe_batched_iter = qe_batched_dataset.make_initializable_iterator()
    return BatchedInput(
        initializer=qe_batched_iter.initializer,
        qe_iterator=qe_batched_iter)


def get_iterator_exp(src_dataset,
                     tgt_dataset,
                     src_vocab_table,
                     tgt_vocab_table,
                     batch_size,
                     sos,
                     eos,
                     random_seed,
                     bucket_width,
                     num_gpus,
                     src_max_len=None,
                     tgt_max_len=None,
                     num_parallel_calls=8,
                     output_buffer_size=None,
                     skip_count=None,
                     reshuffle_each_iteration=True,
                     mode=tf.contrib.learn.ModeKeys.TRAIN):
    batch_size = batch_size * num_gpus

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            padding_values=(
                0,  # src_eos_id,  # src
                0,  # tgt_eos_id,  # mt
                0,  # src_len -- unused
                0))  # tgt_len -- unused

    if not output_buffer_size:
        output_buffer_size = 500000

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    qe_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if skip_count is not None:
        qe_dataset = qe_dataset.skip(skip_count)

    qe_dataset = qe_dataset.shuffle(output_buffer_size, random_seed, reshuffle_each_iteration)

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values,
            tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, tgt: (
                src[:src_max_len],
                tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, tgt: (
                src,
                tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src,tgt: (
            src,
            tf.concat(([tgt_sos_id], tgt, [tgt_eos_id]), 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, tgt: (
            src,
            tgt,
            tf.size(src),
            tf.size(tgt)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if bucket_width > 1 and mode == tf.contrib.learn.ModeKeys.TRAIN:

        def key_func(unused_1, unused_2, src_len, tgt_len):
            bucket_id = tf.constant(0, dtype=tf.int32)
            bucket_id = tf.maximum(bucket_id, src_len // bucket_width)
            bucket_id = tf.maximum(bucket_id, tgt_len // bucket_width)
            return tf.to_int64(bucket_id)  # tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        def window_size_func(key):
            if bucket_width > 1:
                key += 1  # For bucket_width == 1, key 0 is unassigned.
            size = batch_size // (key * bucket_width)
            if num_gpus > 1:
                size = size + num_gpus - size % num_gpus
            return tf.to_int64(tf.maximum(size, num_gpus))

        # bucketing for qe data
        qe_batched_dataset = qe_dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func))

    else:
        qe_batched_dataset = batching_func(qe_dataset)

    qe_batched_iter = qe_batched_dataset.make_initializable_iterator()
    return BatchedInput(
        initializer=qe_batched_iter.initializer,
        qe_iterator=qe_batched_iter)


############################## Model helper ###################################
class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
    pass


def create_train_model(model_creator, hparams, scope=None):
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab, hparams.max_vocab_size)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = get_iterator_exp(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            bucket_width=hparams.bucket_width,
            num_gpus=hparams.num_gpus,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            skip_count=skip_count_placeholder)

        model_device_fn = None
        with tf.device(model_device_fn):
            model = model_creator(
                hparams,
                iterator=iterator,
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                scope=scope)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder", "tgt_placeholder",
                            "batch_size_placeholder", "iterator"))):
    pass


def create_infer_model(model_creator, hparams, scope=None):
    graph = tf.Graph()
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    with graph.as_default(), tf.container(scope or "infer"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab, hparams.max_vocab_size)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(tgt_vocab_file, default_value=UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        tgt_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
        tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_placeholder)

        iterator = get_infer_iterator_exp(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.infer_batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)

        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            reverse_target_vocab_table=reverse_tgt_vocab_table,
            scope=scope)

        return InferModel(
            graph=graph,
            model=model,
            src_placeholder=src_placeholder,
            tgt_placeholder=tgt_placeholder,
            batch_size_placeholder=batch_size_placeholder,
            iterator=iterator)


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print_out("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name, hparams):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print_out("  created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))
    global_step = model.global_step.eval(session=session)
    skip_count = model.skip_count.eval(session=session)
    return model, global_step, skip_count


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, gradient_norm_summary, gradient_norm


def avg_checkpoints(model_dir, num_last_checkpoints, global_step,
                    global_step_name):
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if not checkpoint_state:
        print_out("# No checkpoint file found in directory: %s" % model_dir)
        return None

    checkpoints = (checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

    if len(checkpoints) < num_last_checkpoints:
        print_out("# Skipping averaging checkpoints because not enough checkpoints is avaliable.")
        return None

    avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
    if not tf.gfile.Exists(avg_model_dir):
        print_out("# Creating new directory %s for saving averaged checkpoints." % avg_model_dir)
        tf.gfile.MakeDirs(avg_model_dir)

    print_out("# Reading and averaging variables in checkpoints:")
    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if name != global_step_name:
            var_values[name] = np.zeros(shape)

    for checkpoint in checkpoints:
        print_out("    %s" % checkpoint)
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor

    for name in var_values:
        var_values[name] /= len(checkpoints)

    with tf.Graph().as_default():
        tf_vars = [tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name]) for v in var_values]

        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        global_step_var = tf.Variable(global_step, name=global_step_name, trainable=False)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for p, assign_op, (name, value) in zip(placeholders, assign_ops, six.iteritems(var_values)):
                sess.run(assign_op, {p: value})
            saver.save(sess, os.path.join(avg_model_dir, "qe.ckpt"))

    return avg_model_dir


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def approximate_split(x, num_splits, axis=0):
    """Split approximately equally into num_splits parts.
    Args:
        x: a Tensor
        num_splits: an integer
        axis: an integer.
    Returns:
        a list of num_splits Tensors.
    """
    size = shape_list(x)[axis]
    size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
    return tf.split(x, size_splits, axis=axis)


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size."""
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"


def create_emb(embed_name, vocab_size, embed_size, dtype):
    with tf.device(_get_embed_device(vocab_size)):
        embedding = tf.get_variable(
            embed_name, [vocab_size, embed_size], dtype)
    return embedding


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       scope=None):
    with tf.variable_scope(scope or "embeddings", dtype=dtype) as scope:
        # Share embedding
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Share embedding but different src/tgt vocab sizes"
                                 " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
            assert src_embed_size == tgt_embed_size
            print_out("# Use the same embedding for source and target")
            embedding_encoder = create_emb("embedding_share", src_vocab_size, src_embed_size, dtype)
            embedding_decoder = embedding_encoder
        else:
            with tf.variable_scope("encoder"):
                embedding_encoder = create_emb("embedding_encoder", src_vocab_size, src_embed_size, dtype)
            with tf.variable_scope("decoder"):
                embedding_decoder = create_emb("embedding_decoder", tgt_vocab_size, tgt_embed_size, dtype)

    return embedding_encoder, embedding_decoder


def shift_concat(decoder_outputs, fake_tensors, time_major=False):
    """
    suppose the target input is <s> y1, y2, ... <e>
    target output is y1, y2, ... <e>
    Args:
      decoder_outputs: tuple with forward and backward outputs
      ([ft], [bt])

    Return: ft and b1 are disgarded, fake can be the initial state.
      [fake, b2], [f1, b3], ..., [f_{t-2}, b_t], [f_{t-1}, fake]
    """
    (fw_outputs, bw_outputs) = decoder_outputs
    if fake_tensors is not None:
        (fw_ft, bw_ft) = fake_tensors
        fw_ft = tf.expand_dims(fw_ft, 0) if time_major else tf.expand_dims(fw_ft, 1)
        bw_ft = tf.expand_dims(bw_ft, 0) if time_major else tf.expand_dims(bw_ft, 1)
    else:
        fw_ft = tf.zeros_like(fw_outputs[0:1]) if time_major else tf.zeros_like(fw_outputs[:, 0:1])
        bw_ft = tf.zeros_like(bw_outputs[0:1]) if time_major else tf.zeros_like(bw_outputs[:, 0:1])
    if time_major:
        fw_outputs = tf.concat([fw_ft, fw_outputs[:-1]], axis=0)
        bw_outputs = tf.concat([bw_outputs[1:], bw_ft], axis=0)
    else:
        fw_outputs = tf.concat([fw_ft, fw_outputs[:, :-1]], axis=1)
        bw_outputs = tf.concat([bw_outputs[:, 1:], bw_ft], axis=1)
    return tf.concat([fw_outputs, bw_outputs], axis=-1)


############################## Model Class ####################################
class BilingualExpert(object):

    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 reverse_target_vocab_table=None,
                 scope=None):

        self.iterator = iterator
        self.mode = mode

        self.label_smoothing = hparams.label_smoothing

        self.num_encoder_layers = hparams.num_encoder_layers
        self.num_decoder_layers = hparams.num_decoder_layers

        # Initializer
        initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=hparams.random_seed)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        self.embedding_encoder, self.embedding_decoder = create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=hparams.src_vocab_size,
            tgt_vocab_size=hparams.tgt_vocab_size,
            src_embed_size=hparams.embedding_size,
            tgt_embed_size=hparams.embedding_size,
            scope=scope)

        # Expert Model
        # encoder
        self.encoder = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
            hparams.num_encoder_layers,
            num_units=hparams.num_units,
            num_heads=hparams.num_heads,
            ffn_inner_dim=hparams.ffn_inner_dim,
            dropout=hparams.dropout,
            attention_dropout=hparams.dropout,
            relu_dropout=hparams.dropout,
            position_encoder=onmt.layers.position.SinusoidalPositionEncoder())

        #  fw_decoder
        self.fw_decoder = onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
            hparams.num_decoder_layers,
            num_units=hparams.num_units,
            num_heads=hparams.num_heads,
            ffn_inner_dim=hparams.ffn_inner_dim,
            dropout=hparams.dropout,
            attention_dropout=hparams.dropout,
            relu_dropout=hparams.dropout,
            position_encoder=onmt.layers.position.SinusoidalPositionEncoder())

        #  bw_decoder
        self.bw_decoder = onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
            hparams.num_decoder_layers,
            num_units=hparams.num_units,
            num_heads=hparams.num_heads,
            ffn_inner_dim=hparams.ffn_inner_dim,
            dropout=hparams.dropout,
            attention_dropout=hparams.dropout,
            relu_dropout=hparams.dropout,
            position_encoder=onmt.layers.position.SinusoidalPositionEncoder())

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.skip_count = tf.Variable(0, trainable=False, name="skil_count")

        # data used
        self.source, self.target, self.src_sequence_length, self.tgt_sequence_length = self.iterator.qe_iterator.get_next()

        # build graph
        res = self.build_graph(hparams, scope=scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss, _ = res
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            _, sampled_ids = res
            self.sampled_words = reverse_target_vocab_table.lookup(tf.to_int64(sampled_ids))

        self.params = tf.trainable_variables()

        # optimization
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            self.learning_rate = self._get_learning_rate_warmup(hparams)

            self.optimizer = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.998)

            # Gradients
            gradients = tf.gradients(self.train_loss,
                                     self.params,
                                     colocate_gradients_with_ops=True)
            clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients, 5.0)
            self.grad_norm = grad_norm

            self.update = tf.contrib.layers.optimize_loss(self.train_loss,
                                                          self.global_step,
                                                          learning_rate=None,
                                                          optimizer=self.optimizer,
                                                          variables=self.params,
                                                          colocate_gradients_with_ops=True)
            # also update skip count
            tf.assign_add(self.skip_count, tf.size(self.src_sequence_length), use_locking=True)

            # Summary
            self.train_summary = tf.summary.merge([
                                                      tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("train_loss", self.train_loss)
                                                  ] + grad_norm_summary)

        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        if hparams.avg_ckpts:
            self.avg_best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Print trainable variables
        print_out("# Trainable variables")
        for param in self.params:
            print_out("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

    def _get_learning_rate_warmup(self, hparams):
        warmup_steps = hparams.warmup_steps
        print_out("  learning_rate=%g, warmup_steps=%d" % (hparams.learning_rate, warmup_steps))

        step_num = tf.to_float(self.global_step) / 2. + 1
        inv_decay = hparams.num_units ** -0.5 * tf.minimum(step_num * warmup_steps ** -1.5, step_num ** -0.5)
        return inv_decay * self.learning_rate

    def build_graph(self, hparams, scope=None):
        print_out("# creating %s graph ..." % self.mode)
        dtype = tf.float32
        with tf.variable_scope(scope or "transformerpredictor", dtype=dtype):

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                sampled_ids = None
                source_shards = approximate_split(self.source, hparams.num_gpus)
                src_len_shards = approximate_split(self.src_sequence_length, hparams.num_gpus)
                target_shards = approximate_split(self.target, hparams.num_gpus)
                tgt_len_shards = approximate_split(self.tgt_sequence_length, hparams.num_gpus)

                loss_shards = []
                devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
                for i, device in enumerate(devices):
                    with tf.name_scope("parallel_{}".format(i)):
                        with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
                            with tf.device(device):
                                logits, _ = self._build_model(source_shards[i],
                                                              src_len_shards[i],
                                                              target_shards[i],
                                                              tgt_len_shards[i],
                                                              hparams)
                                loss_shards.append(self._compute_loss(logits,
                                                                      target_shards[i],
                                                                      tgt_len_shards[i]))
                _loss = tuple(list(loss_shard) for loss_shard in zip(*loss_shards))
                loss = tf.add_n(_loss[0]) / tf.add_n(_loss[1])

            elif self.mode == tf.contrib.learn.ModeKeys.INFER:
                loss = None
                _, sampled_ids = self._build_model(self.source,
                                                   self.src_sequence_length,
                                                   self.target,
                                                   self.tgt_sequence_length,
                                                   hparams)
            return loss, sampled_ids

    def _build_model(self, src, src_seq_lens, tgt, tgt_seq_lens, hparams):
        with tf.variable_scope("encoder"):
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, src)

            # Encoder_outputs: [max_time, batch_size, num_units]
            encoder_outputs, _, encoder_sequence_length = self.encoder.encode(
                encoder_emb_inp,
                src_seq_lens,
                mode=self.mode)

        fw_target_input = tgt

        fw_decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, fw_target_input)

        bw_decoder_emb_inp = tf.reverse_sequence(
            fw_decoder_emb_inp,
            tgt_seq_lens,
            batch_dim=0,
            seq_dim=1)

        # Decoder
        with tf.variable_scope("decoder"):
            self.output_layer = tf.layers.Dense(
                hparams.tgt_vocab_size, use_bias=False, name="output_projection")
            self.emb_proj_layer = tf.layers.Dense(
                2 * hparams.num_units, use_bias=False, name="emb_proj_layer")

            with tf.variable_scope("fw_decoder"):
                fw_outputs, _, _ = self.fw_decoder.decode(
                    fw_decoder_emb_inp,
                    tgt_seq_lens,
                    vocab_size=hparams.tgt_vocab_size,
                    initial_state=None,  # unused arg for transformer
                    sampling_probability=None,  # unsupported arg for transformer
                    embedding=None,  # unused arg for transformer
                    output_layer=lambda x: x,
                    mode=self.mode,
                    memory=encoder_outputs,
                    memory_sequence_length=encoder_sequence_length)

            with tf.variable_scope("bw_decoder"):
                bw_outputs, _, _ = self.bw_decoder.decode(
                    bw_decoder_emb_inp,
                    tgt_seq_lens,
                    vocab_size=hparams.tgt_vocab_size,
                    initial_state=None,  # unused arg for transformer
                    sampling_probability=None,  # unsupported arg for transformer
                    embedding=None,  # unused arg for transformer
                    output_layer=lambda x: x,
                    mode=self.mode,
                    memory=encoder_outputs,
                    memory_sequence_length=encoder_sequence_length)

            bw_outputs_rev = tf.reverse_sequence(
                bw_outputs,
                tgt_seq_lens,
                batch_dim=0,
                seq_dim=1)

            shift_outputs = shift_concat(
                (fw_outputs, bw_outputs_rev),
                None)

            shift_inputs = shift_concat(
                (fw_decoder_emb_inp, fw_decoder_emb_inp),
                None)

            shift_proj_inputs = self.emb_proj_layer(shift_inputs)
            _pre_qefv = tf.concat([shift_outputs, shift_proj_inputs], axis=-1)
            # _pre_qefv = shift_outputs + shift_proj_inputs
            # Notice, currently <s> is not to predict, but actually in our QE model, we can predict it.
            logits = self.output_layer(_pre_qefv)
            sampled_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        return logits, sampled_ids

    def _compute_loss(self, logits, tgt, tgt_seq_lens):
        crossent, normalizer, _ = onmt.utils.losses.cross_entropy_sequence_loss(
            logits,
            tgt,
            tgt_seq_lens,
            self.label_smoothing,
            average_in_time=True,
            mode=self.mode)
        return crossent, normalizer

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.train_summary,
                         self.global_step,
                         self.grad_norm,
                         self.learning_rate])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run(self.sampled_words)


############################## Training  #################################
def load_data(inference_input_file):
    """Load inference data."""
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(inference_input_file, mode="rb")) as f:
        inference_data = f.read().splitlines()
    return inference_data


def save_hparams(out_dir, hparams):
    hparams_file = os.path.join(out_dir, "hparams")
    print_out("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json(indent="\t"))


def add_summary(summary_writer, global_step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def evaluate(ref_file, pred_file, metric):
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, pred_file)
    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, pred_file)
    else:
        raise ValueError("Unknown metric %s" % metric)
    return evaluation_score


def _clean(sentence, subword_option):
    """Clean and handle BPE or SPM outputs."""
    sentence = sentence.strip()
    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)
    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()
    return sentence


def _bleu(ref_file, trans_file, subword_option=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio, translation_length, reference_length


def _word_accuracy(label_file, pred_file):
    """Compute accuracy on per word basis."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count


def decode_and_evaluate(name,
                        model,
                        sess,
                        pred_file,
                        ref_file,
                        metrics,
                        tgt_eos):
    start_time = time.time()
    num_sentences = 0
    if tgt_eos:
        tgt_eos = tgt_eos.encode("utf-8")
    with codecs.getwriter("utf-8")(tf.gfile.GFile(pred_file, mode="w")) as pred_f:
        pred_f.write("")
        while True:
            try:
                sampled_words = model.infer(sess)
                batch_size = sampled_words.shape[0]
                for sent_id in range(batch_size):
                    output = sampled_words[sent_id].tolist()
                    if tgt_eos and tgt_eos in output:
                        output = output[:output.index(tgt_eos)]
                    pred_f.write((b" ".join(output) + b"\n").decode("utf-8"))
                num_sentences += batch_size
            except tf.errors.OutOfRangeError:
                print_out("  done, num sentences %d, time %ds" % (num_sentences, time.time() - start_time))
                break
    # Evaluation
    evaluation_scores = {}
    if ref_file:
        for metric in metrics:
            score = evaluate(ref_file, pred_file, metric)
            evaluation_scores[metric] = score
            print_out("  %s %s: %.4f" % (metric, name, score))
    return evaluation_scores


def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, ref_file, label, summary_writer,
                   save_on_best, avg_ckpts=False):
    """External evaluation for f1, pearson, etc."""
    out_dir = hparams.out_dir
    if avg_ckpts:
        label = "avg_" + label
    print_out("# External evaluation, global step %d" % global_step)

    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

    output = os.path.join(out_dir, "output_%s" % label)
    scores = decode_and_evaluate(
        label,
        model,
        sess,
        output,
        ref_file,
        metrics=hparams.metrics,
        tgt_eos=hparams.eos)
    # Save on best metrics
    for metric in hparams.metrics:
        if avg_ckpts:
            best_metric_label = "avg_best_" + metric
            if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
                setattr(hparams, best_metric_label, scores[metric])
                model.avg_best_saver.save(
                    sess,
                    os.path.join(
                        getattr(hparams, best_metric_label + "_dir"), str(scores[metric]) + "qe.ckpt"),
                    global_step=model.global_step)
        else:
            best_metric_label = "best_" + metric
            if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
                setattr(hparams, best_metric_label, scores[metric])
                model.best_saver.save(
                    sess,
                    os.path.join(
                        getattr(hparams, best_metric_label + "_dir"), str(scores[metric]) + "qe.ckpt"),
                    global_step=model.global_step)
        add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                    scores[metric])
    save_hparams(out_dir, hparams)
    return scores


def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True, use_test_set=True,
                      avg_ckpts=False):
    """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
    with infer_model.graph.as_default():
        loaded_infer_model, global_step, _ = create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer", hparams)

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_infer_iterator_feed_dict = {
        infer_model.src_placeholder: load_data(dev_src_file),
        infer_model.tgt_placeholder: load_data(dev_tgt_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    dev_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        dev_infer_iterator_feed_dict,
        dev_tgt_file,
        "dev",
        summary_writer,
        save_on_best=save_best_dev,
        avg_ckpts=avg_ckpts)

    test_scores = None
    if use_test_set and hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_infer_iterator_feed_dict = {
            infer_model.src_placeholder: load_data(test_src_file),
            infer_model.tgt_placeholder: load_data(test_tgt_file),
            infer_model.batch_size_placeholder: hparams.infer_batch_size,
        }
        test_scores = _external_eval(
            loaded_infer_model,
            global_step,
            infer_sess,
            hparams,
            infer_model.iterator,
            test_infer_iterator_feed_dict,
            test_tgt_file,
            "test",
            summary_writer,
            save_on_best=False,
            avg_ckpts=avg_ckpts)
    return dev_scores, test_scores, global_step


def run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                          summary_writer, global_step):
    """Creates an averaged checkpoint and run external eval with it."""
    avg_dev_scores, avg_test_scores = None, None
    if hparams.avg_ckpts:
        # Convert VariableName:0 to VariableName.
        global_step_name = infer_model.model.global_step.name.split(":")[0]
        avg_model_dir = avg_checkpoints(model_dir, hparams.num_keep_ckpts, global_step, global_step_name)
        if avg_model_dir:
            avg_dev_scores, avg_test_scores, _ = run_external_eval(
                infer_model,
                infer_sess,
                avg_model_dir,
                hparams,
                summary_writer,
                avg_ckpts=True)
    return avg_dev_scores, avg_test_scores


def init_stats():
    """Initialize statistics that we want to accumulate."""
    return {"step_time": 0.0, "exp_loss": 0.0, "grad_norm": 0.0, }


def update_stats(stats, start_time, step_result):
    """Update stats: write summary and accumulate statistics."""
    (_, step_loss, step_summary, global_step, grad_norm, learning_rate) = step_result
    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["exp_loss"] += step_loss
    stats["grad_norm"] += grad_norm
    return global_step, learning_rate, step_summary


def print_step_info(prefix, global_step, info, result_summary, log_f):
    """Print all info at the current global step."""
    print_out("%sstep %d lr %g step-time %.2fs exp-los %.4f gN %.2f %s, %s" %
              (prefix, global_step, info["learning_rate"], info["avg_step_time"],
               info["avg_train_exp_loss"], info["avg_grad_norm"], result_summary, time.ctime()), log_f)


def process_stats(stats, info, global_step, steps_per_stats, log_f):
    """Update info and check for overflow."""
    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["avg_train_exp_loss"] = stats["exp_loss"] / steps_per_stats
    avg_train_expl = info["avg_train_exp_loss"]
    is_overflow = False
    if math.isnan(avg_train_expl) or math.isinf(avg_train_expl) or avg_train_expl > 1e20:
        print_out("  step %d overflow (expert loss), stop early" % global_step, log_f)
        is_overflow = True
    return is_overflow


def _get_best_results(hparams):
    """Summary of the current best results."""
    tokens = []
    for metric in hparams.metrics:
        tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
    return ", ".join(tokens)


def before_train(loaded_train_model, train_model, train_sess, global_step, skip_count, hparams, log_f):
    """Misc tasks to do before training."""
    stats = init_stats()
    info = {"avg_step_time": 0.0, "avg_grad_norm": 0.0, "avg_train_exp_loss": 0.0,
            "learning_rate": loaded_train_model.learning_rate.eval(
                session=train_sess)}
    start_train_time = time.time()
    print_out("# Start step %d, lr %g, %s" % (global_step, info["learning_rate"], time.ctime()), log_f)

    # Initialize all of the iterators
    print_out("# Init train iterator, skipping %d elements" % skip_count)
    train_sess.run(
        train_model.iterator.initializer,
        feed_dict={train_model.skip_count_placeholder: skip_count})

    return stats, info, start_train_time


def train(hparams, scope=None, target_session=""):
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    steps_per_external_eval = hparams.steps_per_external_eval
    steps_per_eval = 10 * steps_per_stats
    avg_ckpts = hparams.avg_ckpts

    if not steps_per_external_eval:
        steps_per_external_eval = 5 * steps_per_eval

    model_creator = BilingualExpert
    train_model = create_train_model(model_creator, hparams, scope)
    infer_model = create_infer_model(model_creator, hparams, scope)

    summary_name = "train_log"
    model_dir = hparams.out_dir

    # Log and output files
    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    print_out("# log_file=%s" % log_file, log_f)

    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True

    train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model.graph)
    infer_sess = tf.Session(target=target_session, config=config_proto, graph=infer_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step, skip_count = create_or_load_model(
            train_model.model, model_dir, train_sess, "train", hparams)

    # Summary writer
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, summary_name), train_model.graph)

    # First evaluation
    run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer)

    last_stats_step = global_step
    last_eval_step = global_step
    last_external_eval_step = global_step

    # This is the training loop.
    stats, info, start_train_time = before_train(loaded_train_model, train_model, train_sess, global_step, 
                                                 skip_count, hparams, log_f)
    while global_step < num_train_steps:
        # Run a step
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            hparams.epoch_step = 0
            print_out("# Finished an epoch, step %d. " % global_step)

            # # Not necessary
            # run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer)
            # if avg_ckpts:
            #     run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step)

            train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: 0})
            continue

        # Process step_result, accumulate stats, and write summary
        global_step, info["learning_rate"], step_summary = update_stats(stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step
            is_overflow = process_stats(stats, info, global_step, steps_per_stats, log_f)
            print_step_info("  ", global_step, info, _get_best_results(hparams), log_f)
            if is_overflow:
                break

            # Reset statistics
            stats = init_stats()

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step
            print_out("# Save eval, global step %d" % global_step)

            # Save checkpoint
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(out_dir, "exp.ckpt"),
                global_step=global_step)

        if global_step - last_external_eval_step >= steps_per_external_eval:
            last_external_eval_step = global_step

            # Save checkpoint
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(out_dir, "exp.ckpt"),
                global_step=global_step)
            run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer)

            if avg_ckpts:
                run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step)

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(out_dir, "exp.ckpt"),
        global_step=global_step)

    print_out("# Done training!", time.time() - start_train_time)

    summary_writer.close()
    return global_step


############################## HPARAMS ########################################
FLAGS = None


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--embedding_size", type=int, default=64, help="src/rgt word embedding size")
    parser.add_argument("--num_units", type=int, default=512, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Transformer depth.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Num of self-attention head.")
    parser.add_argument("--ffn_inner_dim", type=int, default=512,
                        help="ffn size in self-attention.")

    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--label_smoothing", type=float, default=0.,
                        help="smoothing label for cross entropy")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument(
        "--num_train_steps", type=int, default=100000, help="Num steps to train.")

    # data
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix for QE, e.g., qe.en.")
    parser.add_argument("--tgt", type=str, default=None,
                        help="MT suffix for QE, e.g., qe.de.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    # Vocab
    parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
          Vocab prefix, expect files with src/tgt suffixes.\
          """)
    parser.add_argument("--max_vocab_size", type=int, default=50000, help="""\
          maximum vocab size when creating vocab from corpus.\
          """)
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                        default=False,
                        help="""\
          Whether to use the source vocab and embeddings for both source and
          target.\
          """)
    parser.add_argument("--check_special_token", type="bool", default=True,
                        help="""\
                          Whether check special sos, eos, unk tokens exist in the
                          vocab files.\
                          """)

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=70,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=70,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="""\
          Max length of tgt sequences during inference.  Also use to restrict the
          maximum decoding length.\
          """)
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference mode.")

    # Default settings
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of tokens for each batch.")
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--bucket_width", type=int, default=5,
                        help="Put data into similar-length buckets.")

    # Misc
    parser.add_argument("--metrics", type=str, default="bleu",
                        help=("Comma-separated list of evaluations "
                              "metrics (bleu,word_accuracy)"))
    parser.add_argument("--steps_per_external_eval", type=int, default=None,
                        help="""\
          How many training steps to do per external evaluation.  Automatically set
          based on data if None.\
          """)
    parser.add_argument("--scope", type=str, default=None,
                        help="scope to put variables under")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                        help="Max number of checkpoints to keep.")
    parser.add_argument("--avg_ckpts", type="bool", nargs="?",
                        const=True, default=True, help=("""\
                          Average the last N checkpoints for external evaluation.
                          N can be controlled by setting --num_keep_ckpts.\
                          """))
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of gpus used for training.")


def load_hparams(model_dir):
    hparams_file = os.path.join(model_dir, "hparams")
    if tf.gfile.Exists(hparams_file):
        print_out("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                print_out("  can't load hparams file")
                return None
        return hparams
    else:
        return None


def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        max_vocab_size=flags.max_vocab_size,
        out_dir=flags.out_dir,
        label_smoothing=flags.label_smoothing,

        # Networks
        embedding_size=flags.embedding_size,
        num_units=flags.num_units,
        num_layers=flags.num_layers,  # Compatible
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        num_heads=flags.num_heads,
        ffn_inner_dim=flags.ffn_inner_dim,
        dropout=flags.dropout,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,

        # Data constraints
        bucket_width=flags.bucket_width,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,

        # Vocab
        sos=flags.sos if flags.sos else SOS,
        eos=flags.eos if flags.eos else EOS,
        check_special_token=flags.check_special_token,

        # Misc
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(","),
        random_seed=flags.random_seed,
        num_keep_ckpts=flags.num_keep_ckpts,
        avg_ckpts=flags.avg_ckpts,
        num_gpus=flags.num_gpus,
    )


def extend_hparams(hparams):
    # Get vocab file names first
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    else:
        raise ValueError("hparams.vocab_prefix must be provided.")

    src_vocab_size, src_vocab_file = check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        blk=BLK,
        data_file=hparams.train_prefix + "." + hparams.src,
        max_vocabulary_size=hparams.max_vocab_size)

    if hparams.share_vocab:
        print_out("  using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = check_vocab(
            tgt_vocab_file,
            hparams.out_dir,
            check_special_token=hparams.check_special_token,
            sos=hparams.sos,
            eos=hparams.eos,
            blk=BLK,
            data_file=hparams.train_prefix + "." + hparams.tgt,
            max_vocabulary_size=hparams.max_vocab_size)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

    if not tf.gfile.Exists(hparams.out_dir):
        print_out("# Creating output directory %s ..." % hparams.out_dir)
        tf.gfile.MakeDirs(hparams.out_dir)

    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

        if hparams.avg_ckpts:
            hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
            best_metric_dir = os.path.join(hparams.out_dir, "avg_best_" + metric)
            hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
            tf.gfile.MakeDirs(best_metric_dir)

    return hparams


def run_main(flags, default_hparams, train_fn, target_session=""):
    # Random
    random_seed = flags.random_seed
    if random_seed is not None and random_seed > 0:
        print_out("# Set random seed to %d" % random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Train
    out_dir = flags.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)
    hparams = load_hparams(out_dir)
    if not hparams:
        hparams = extend_hparams(default_hparams)
    save_hparams(out_dir, hparams)
    for metric in hparams.metrics:
        save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams)

    values = hparams.values()
    skip_patterns = None
    for key in sorted(values.keys()):
        if not skip_patterns or all([skip_pattern not in key for skip_pattern in skip_patterns]):
            print_out("  %s=%s" % (key, str(values[key])))

    train_fn(hparams, target_session=target_session)


def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    run_main(FLAGS, default_hparams, train)


if __name__ == "__main__":
    qe_parser = argparse.ArgumentParser()
    add_arguments(qe_parser)
    FLAGS, unparsed = qe_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

