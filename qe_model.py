# coding=utf-8
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
from tensorflow.python.layers import core as layers_core

from sklearn.metrics import f1_score
from scipy.stats.stats import spearmanr

import re
_DIGIT_RE = re.compile(r"\d")

############################## Vocab Utils ##########################################
BLK = "<blank>"
SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
VOCAB_SIZE_THRESHOLD_CPU = 50000
sent_suffix = ".hter"
word_suffix = ".wtags"
gap_suffix = ".gtags"


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


def get_infer_iterator_no_hf(src_dataset,
                             mt_dataset,
                             src_vocab_table,
                             tgt_vocab_table,
                             batch_size,
                             sos,
                             eos,
                             src_max_len=None,
                             tgt_max_len=None):
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    qe_dataset = tf.data.Dataset.zip((src_dataset, mt_dataset))

    qe_dataset = qe_dataset.map(
        lambda src, mt: (
            tf.string_split([src]).values,
            tf.string_split([mt]).values))

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
            lambda src, mt: (src[:src_max_len], mt))
    if tgt_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, mt: (src, mt[:tgt_max_len]))

    qe_dataset = qe_dataset.map(
        lambda src, mt: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(mt), tf.int32)))

    qe_dataset = qe_dataset.map(
        lambda src, mt: (
            src,
            tf.concat(([tgt_sos_id], mt, [tgt_eos_id]), 0)))

    qe_dataset = qe_dataset.map(
        lambda src, mt: (
            src,
            mt,
            tf.size(src),
            tf.size(mt)))

    qe_batched_dataset = batching_func(qe_dataset)
    qe_batched_iter = qe_batched_dataset.make_initializable_iterator()
    return BatchedInput(
        initializer=qe_batched_iter.initializer,
        qe_iterator=qe_batched_iter)


def get_iterator_no_hf(src_dataset,
                       mt_dataset,
                       lab_dataset,  # data of label for each sent or each token
                       src_vocab_table,
                       tgt_vocab_table,
                       qe_batch_size,
                       sos,
                       eos,
                       random_seed,
                       num_buckets,
                       train_level,
                       src_max_len=None,
                       tgt_max_len=None,
                       num_parallel_calls=8,
                       output_buffer_size=None,
                       skip_count=None,
                       reshuffle_each_iteration=True,
                       mode=tf.contrib.learn.ModeKeys.TRAIN):
    if train_level == "sent":

        def batching_func(x):
            return x.padded_batch(
                qe_batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # mt
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # mt_len
                    tf.TensorShape([])),  # fake_lab
                padding_values=(
                    0,  # src_eos_id,  # src
                    0,  # tgt_eos_id,  # mt
                    0,  # src_len -- unused
                    0,  # mt_len -- unused
                    0.))  # label

    elif train_level in ["word", "gap"]:
        def batching_func(x):
            return x.padded_batch(
                qe_batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # mt
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # mt_len
                    tf.TensorShape([None])),  # fake_lab
                padding_values=(
                    0,  # src_eos_id,  # src
                    0,  # tgt_eos_id,  # mt
                    0,  # src_len -- unused
                    0,  # mt_len -- unused
                    0.))  # label

    else:
        raise ValueError("  train level %s is not Supported." % train_level)

    if not output_buffer_size:
        output_buffer_size = 500000

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    qe_dataset = tf.data.Dataset.zip((src_dataset, mt_dataset, lab_dataset))

    if skip_count is not None:
        qe_dataset = qe_dataset.skip(skip_count)

    qe_dataset = qe_dataset.shuffle(output_buffer_size, random_seed, reshuffle_each_iteration)

    if train_level == "sent":
        qe_dataset = qe_dataset.map(
            lambda src, mt, lab: (
                tf.string_split([src]).values,
                tf.string_split([mt]).values,
                tf.string_to_number(lab)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    elif train_level in ["word", "gap"]:
        qe_dataset = qe_dataset.map(
            lambda src, mt, lab: (
                tf.string_split([src]).values,
                tf.string_split([mt]).values,
                tf.map_fn(lambda x: tf.string_to_number(x), tf.string_split([lab]).values, tf.float32)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.filter(
        lambda src, mt, lab: tf.logical_and(tf.size(src) > 0, tf.size(mt) > 0))

    if src_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, mt, lab: (
                src[:src_max_len],
                mt,
                lab),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        if train_level == "sent":
            qe_dataset = qe_dataset.map(
                lambda src, mt, lab: (
                    src,
                    mt[:tgt_max_len],
                    lab),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        elif train_level == "word":
            qe_dataset = qe_dataset.map(
                lambda src, mt, lab: (
                    src,
                    mt[:tgt_max_len],
                    lab[:tgt_max_len]),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        elif train_level == "gap":
            qe_dataset = qe_dataset.map(
                lambda src, mt, lab: (
                    src,
                    mt[:tgt_max_len],
                    lab[:tgt_max_len + 1]),  # gap has one more slot.
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, mt, lab: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(mt), tf.int32),
            lab),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, mt, lab: (
            src,
            tf.concat(([tgt_sos_id], mt, [tgt_eos_id]), 0),
            lab),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, mt, lab: (
            src,
            mt,
            tf.size(src),
            tf.size(mt),
            lab),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    bucket_width = num_buckets
    if bucket_width > 1 and mode == tf.contrib.learn.ModeKeys.TRAIN:

        def key_func(unused_1, unused_2, src_len, tgt_len, unused_5):
            bucket_id = tf.constant(0, dtype=tf.int32)
            bucket_id = tf.maximum(bucket_id, src_len // bucket_width)
            bucket_id = tf.maximum(bucket_id, tgt_len // bucket_width)
            return tf.to_int64(bucket_id)  # tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        # def window_size_func(key):
        #     if bucket_width > 1:
        #         key += 1  # For bucket_width == 1, key 0 is unassigned.
        #     size = batch_size // (key * bucket_width)
        #     return tf.to_int64(size)

        # bucketing for qe data
        qe_batched_dataset = qe_dataset.apply(
            tf.contrib.data.group_by_window(
            #tf.data.experimental.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=qe_batch_size))

    else:
        qe_batched_dataset = batching_func(qe_dataset)

    qe_batched_iter = qe_batched_dataset.make_initializable_iterator()
    return BatchedInput(
        initializer=qe_batched_iter.initializer,
        qe_iterator=qe_batched_iter)



def get_infer_iterator(src_dataset,
                       mt_dataset,
                       hf_dataset,  # real value (fp32) human feature in tfrecord format
                       src_vocab_table,
                       tgt_vocab_table,
                       batch_size,
                       sos,
                       eos,
                       dim_hf,
                       train_level,
                       src_max_len=None,
                       tgt_max_len=None):
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    qe_dataset = tf.data.Dataset.zip((src_dataset, mt_dataset, hf_dataset))

    if train_level == "sent":
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea: (
                tf.string_split([src]).values,
                tf.string_split([mt]).values,
                tf.parse_single_example(fea, features={"shape": tf.FixedLenFeature([2, ], tf.int64),
                                                       "values": tf.FixedLenFeature([dim_hf, ], tf.float32)})[
                    "values"]))

        def batching_func(x):
            return x.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # mt
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # mt_len
                    tf.TensorShape([dim_hf])),  # hm fea
                padding_values=(
                    0,  # src_eos_id,  # src
                    0,  # tgt_eos_id,  # tgt_input
                    0,  # qe_len -- unused
                    0,  # mt_len -- unused
                    0.0))  # hm fea -- unused

    elif train_level in ["word", "gap"]:
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea: (
                tf.string_split([src]).values,
                tf.string_split([mt]).values,
                tf.parse_single_example(fea, features={"shape": tf.FixedLenFeature([2, ], tf.int64),
                                                       "values": tf.VarLenFeature(tf.float32)})))
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea: (
                src,
                mt,
                tf.reshape(fea["values"].values, [tf.cast(fea["shape"][0], tf.int32), dim_hf])))

        def batching_func(x):
            return x.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # mt
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # mt_len
                    tf.TensorShape([None, dim_hf])),  # hm fea
                padding_values=(
                    0,  # src_eos_id,  # src
                    0,  # tgt_eos_id,  # tgt_input
                    0,  # qe_len -- unused
                    0,  # mt_len -- unused
                    0.0))  # hm fea -- unused

    if src_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea: (src[:src_max_len], mt, fea))
    if tgt_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea: (src, mt[:tgt_max_len], fea))

    qe_dataset = qe_dataset.map(
        lambda src, mt, fea: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(mt), tf.int32),
            fea))

    qe_dataset = qe_dataset.map(
        lambda src, mt, fea: (
            src,
            tf.concat(([tgt_sos_id], mt, [tgt_eos_id]), 0),
            fea))

    qe_dataset = qe_dataset.map(
        lambda src, mt, fea: (
            src,
            mt,
            tf.size(src),
            tf.size(mt),
            fea))

    qe_batched_dataset = batching_func(qe_dataset)
    qe_batched_iter = qe_batched_dataset.make_initializable_iterator()
    return BatchedInput(
        initializer=qe_batched_iter.initializer,
        qe_iterator=qe_batched_iter)


def get_iterator(src_dataset,
                 mt_dataset,
                 hf_dataset,  # fp32 human feature in tfrecord format
                 lab_dataset,  # data of label for each sent or each token
                 src_vocab_table,
                 tgt_vocab_table,
                 qe_batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 train_level,
                 dim_hf,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 reshuffle_each_iteration=True,
                 mode=tf.contrib.learn.ModeKeys.TRAIN):
    if train_level == "sent":

        def batching_func(x):
            return x.padded_batch(
                qe_batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # mt
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # mt_len
                    tf.TensorShape([dim_hf]),  # fake_fea
                    tf.TensorShape([])),  # fake_lab
                padding_values=(
                    0,  # src_eos_id,  # src
                    0,  # tgt_eos_id,  # mt
                    0,  # src_len -- unused
                    0,  # mt_len -- unused
                    0.,  # feature
                    0.))  # label

    elif train_level in ["word", "gap"]:
        def batching_func(x):
            return x.padded_batch(
                qe_batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # mt
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # mt_len
                    tf.TensorShape([None, dim_hf]),  # fake_fea
                    tf.TensorShape([None])),  # fake_lab
                padding_values=(
                    0,  # src_eos_id,  # src
                    0,  # tgt_eos_id,  # mt
                    0,  # src_len -- unused
                    0,  # mt_len -- unused
                    0.,  # feature
                    0.))  # label

    else:
        raise ValueError("  train level %s is not Supported." % train_level)

    if not output_buffer_size:
        output_buffer_size = 500000

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    qe_dataset = tf.data.Dataset.zip((src_dataset, mt_dataset, hf_dataset, lab_dataset))

    if skip_count is not None:
        qe_dataset = qe_dataset.skip(skip_count)

    qe_dataset = qe_dataset.shuffle(output_buffer_size, random_seed, reshuffle_each_iteration)

    if train_level == "sent":
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea, lab: (
                tf.string_split([src]).values,
                tf.string_split([mt]).values,
                tf.parse_single_example(fea, features={"shape": tf.FixedLenFeature([2, ], tf.int64),
                                                       "values": tf.FixedLenFeature([dim_hf, ], tf.float32)})["values"],
                tf.string_to_number(lab)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    elif train_level in ["word", "gap"]:
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea, lab: (
                tf.string_split([src]).values,
                tf.string_split([mt]).values,
                tf.parse_single_example(fea, features={"shape": tf.FixedLenFeature([2, ], tf.int64),
                                                       "values": tf.VarLenFeature(tf.float32)}),
                tf.map_fn(lambda x: tf.string_to_number(x), tf.string_split([lab]).values, tf.float32)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        qe_dataset = qe_dataset.map(
            lambda src, mt, fea, lab: (
                src,
                mt,
                tf.reshape(fea["values"].values, [tf.cast(fea["shape"][0], tf.int32), dim_hf]),
                lab),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.filter(
        lambda src, mt, fea, lab: tf.logical_and(tf.size(src) > 0, tf.size(mt) > 0))

    if src_max_len:
        qe_dataset = qe_dataset.map(
            lambda src, mt, fea, lab: (
                src[:src_max_len],
                mt,
                fea,
                lab),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        if train_level == "sent":
            qe_dataset = qe_dataset.map(
                lambda src, mt, fea, lab: (
                    src,
                    mt[:tgt_max_len],
                    fea,
                    lab),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        elif train_level == "word":
            qe_dataset = qe_dataset.map(
                lambda src, mt, fea, lab: (
                    src,
                    mt[:tgt_max_len],
                    fea[:tgt_max_len],
                    lab[:tgt_max_len]),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        elif train_level == "gap":
            qe_dataset = qe_dataset.map(
                lambda src, mt, fea, lab: (
                    src,
                    mt[:tgt_max_len],
                    fea[:tgt_max_len],
                    lab[:tgt_max_len + 1]),  # gap has one more slot.
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, mt, fea, lab: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(mt), tf.int32),
            fea,
            lab),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, mt, fea, lab: (
            src,
            tf.concat(([tgt_sos_id], mt, [tgt_eos_id]), 0),
            fea,
            lab),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    qe_dataset = qe_dataset.map(
        lambda src, mt, fea, lab: (
            src,
            mt,
            tf.size(src),
            tf.size(mt),
            fea,
            lab),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    bucket_width = num_buckets
    if bucket_width > 1 and mode == tf.contrib.learn.ModeKeys.TRAIN:

        def key_func(unused_1, unused_2, src_len, tgt_len, unused_5, unused_6):
            bucket_id = tf.constant(0, dtype=tf.int32)
            bucket_id = tf.maximum(bucket_id, src_len // bucket_width)
            bucket_id = tf.maximum(bucket_id, tgt_len // bucket_width)
            return tf.to_int64(bucket_id)  # tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        # def window_size_func(key):
        #     if bucket_width > 1:
        #         key += 1  # For bucket_width == 1, key 0 is unassigned.
        #     size = batch_size // (key * bucket_width)
        #     return tf.to_int64(size)

        # bucketing for qe data
        qe_batched_dataset = qe_dataset.apply(
            tf.contrib.data.group_by_window(
            #tf.data.experimental.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=qe_batch_size))

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
    mt_file = "%s.%s" % (hparams.train_prefix, hparams.mt)
    fea_file = "%s.%s" % (hparams.train_prefix, hparams.fea)
    lab_file = "%s.%s" % (hparams.train_prefix, hparams.lab)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab, hparams.max_vocab_size)

        src_dataset = tf.data.TextLineDataset(src_file)
        mt_dataset = tf.data.TextLineDataset(mt_file)
        lab_dataset = tf.data.TextLineDataset(lab_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        if hparams.use_hf:
            fea_dataset = tf.data.TFRecordDataset(fea_file)
            iterator = get_iterator(
                src_dataset,
                mt_dataset,
                fea_dataset,
                lab_dataset,
                src_vocab_table,
                tgt_vocab_table,
                qe_batch_size=hparams.qe_batch_size,
                sos=hparams.sos,
                eos=hparams.eos,
                dim_hf=hparams.dim_hf,
                train_level=hparams.train_level,
                random_seed=hparams.random_seed,
                num_buckets=hparams.num_buckets,
                src_max_len=hparams.src_max_len,
                tgt_max_len=hparams.tgt_max_len,
                skip_count=skip_count_placeholder,)
        else:
            iterator = get_iterator_no_hf(
                src_dataset,
                mt_dataset,
                lab_dataset,
                src_vocab_table,
                tgt_vocab_table,
                qe_batch_size=hparams.qe_batch_size,
                sos=hparams.sos,
                eos=hparams.eos,
                train_level=hparams.train_level,
                random_seed=hparams.random_seed,
                num_buckets=hparams.num_buckets,
                src_max_len=hparams.src_max_len,
                tgt_max_len=hparams.tgt_max_len,
                skip_count=skip_count_placeholder,)
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
                           ("graph", "model", "src_placeholder", "mt_placeholder",
                            "fea_file_placeholder",
                            "batch_size_placeholder", "iterator"))):
    pass


def create_infer_model(model_creator, hparams, scope=None):
    graph = tf.Graph()
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    with graph.as_default(), tf.container(scope or "infer"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab, hparams.max_vocab_size)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        mt_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        fea_file_placeholder = tf.placeholder(shape=[], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
        mt_dataset = tf.data.Dataset.from_tensor_slices(mt_placeholder)
        if hparams.use_hf:
            fea_dataset = tf.data.TFRecordDataset(fea_file_placeholder)
            iterator = get_infer_iterator(
                src_dataset,
                mt_dataset,
                fea_dataset,
                src_vocab_table,
                tgt_vocab_table,
                hparams.infer_batch_size,
                sos=hparams.sos,
                eos=hparams.eos,
                dim_hf=hparams.dim_hf,
                train_level=hparams.train_level,
                src_max_len=hparams.src_max_len_infer,
                tgt_max_len=hparams.tgt_max_len_infer)
        else:
            iterator = get_infer_iterator_no_hf(
                src_dataset,
                mt_dataset,
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
            scope=scope)

        return InferModel(
            graph=graph,
            model=model,
            src_placeholder=src_placeholder,
            mt_placeholder=mt_placeholder,
            fea_file_placeholder=fea_file_placeholder,
            batch_size_placeholder=batch_size_placeholder,
            iterator=iterator)


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print_out("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model


def load_expert_weights(exp_model_dir, model, session):
    """
    call this function after session.run(tf.global_variables_initializer())
    """
    checkpoint_state = tf.train.get_checkpoint_state(exp_model_dir)
    if not checkpoint_state:
        print_out("# No checkpoint file found in directory: %s" % exp_model_dir)
        return

    checkpoint = checkpoint_state.all_model_checkpoint_paths[-1]
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)

    var_values = {}
    for name, shape in var_list:
        if name != "global_step" and not "optim" in name and not "words_per_sec" in name:
            if 'inputter_0/w_embs' in name:
                var_values["embeddings/encoder/embedding_encoder:0"] = reader.get_tensor(name)
            elif 'inputter_1/w_embs' in name:
                var_values["embeddings/decoder/embedding_decoder:0"] = reader.get_tensor(name)
            elif 'inputter/w_embs' in name:
                var_values["embeddings/embedding_share:0"] = reader.get_tensor(name)
            else:
                var_values[name+":0"] = reader.get_tensor(name)

    loaded_vars = [var for var in model.exp_params if var.name in var_values]
    placeholders = [tf.placeholder(var.dtype, shape=var.shape) for var in loaded_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(loaded_vars, placeholders)]
    for p, assign_op, var in zip(placeholders, assign_ops, loaded_vars):
        session.run(assign_op, {p: var_values[var.name]})


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
    if global_step == 0:
        if hparams.exp_model_dir:
            start_time = time.time()
            load_expert_weights(hparams.exp_model_dir, model, session)
            print_out("  load pretrained expert weights for %s model, time %.2fs" % (name, time.time() - start_time))
    return model, global_step


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
class QEModel(object):

    def __init__(self,
                 hparams,
                 mode,
                 iterator,
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

        # projection
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(
                    hparams.tgt_vocab_size, use_bias=False, name="output_projection")
                self.emb_proj_layer = layers_core.Dense(
                    2 * hparams.num_units, use_bias=False, name="emb_proj_layer")

        # Estimator Model
        self.estimator = onmt.encoders.BidirectionalRNNEncoder(
            num_layers=hparams.rnn_layers,
            num_units=hparams.rnn_units,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.5,
            residual_connections=False)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # data used
        batch_data_used = self.iterator.qe_iterator.get_next()
        # self.target (in expert) here means mt (in QE).
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            if hparams.use_hf:
                self.source, self.target, self.src_sequence_length, self.tgt_sequence_length, self.feat, self.label = batch_data_used
            else:
                self.source, self.target, self.src_sequence_length, self.tgt_sequence_length, self.label = batch_data_used
        else:
            if hparams.use_hf:
                self.source, self.target, self.src_sequence_length, self.tgt_sequence_length, self.feat = batch_data_used
            else:
                self.source, self.target, self.src_sequence_length, self.tgt_sequence_length = batch_data_used
        self.batch_size = tf.size(self.src_sequence_length)

        # build graph
        res = self.build_graph(hparams, scope=scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[0]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            _, infer_est_logits = res
            self.infer_sent_logits, self.infer_word_logits, self.infer_gap_logits = infer_est_logits
            if self.infer_sent_logits is None:
                self.infer_sent_prob = tf.constant(0, tf.int32)
            else:
                self.infer_sent_prob = tf.sigmoid(self.infer_sent_logits)
            if self.infer_word_logits is None:
                self.infer_word_prob = tf.constant(0, tf.int32)
            else:
                self.infer_word_prob = tf.nn.softmax(self.infer_word_logits, axis=-1)
            if self.infer_gap_logits is None:
                self.infer_gap_prob = tf.constant(0, tf.int32)
            else:
                self.infer_gap_prob = tf.nn.softmax(self.infer_gap_logits, axis=-1)

        self.params = tf.trainable_variables()
        self.exp_params = [var for var in self.params if 'encoder' in var.name or 'decoder' in var.name]
        self.est_params = [var for var in self.params if 'estimator' in var.name]

        # define actual trainable params
        params = self.est_params if hparams.fixed_exp else self.params

        # optimization
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            self.learning_rate = self._get_learning_rate_warmup(hparams)

            self.optimizer = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.998)

            # Gradients
            gradients = tf.gradients(self.train_loss,
                                     params,
                                     colocate_gradients_with_ops=True)
            clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients, 5.0)
            self.grad_norm = grad_norm

            self.update = tf.contrib.layers.optimize_loss(self.train_loss,
                                                          self.global_step,
                                                          learning_rate=None,
                                                          optimizer=self.optimizer,
                                                          variables=params,
                                                          colocate_gradients_with_ops=True)
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
        for param in params:
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
            # Encoder
            encoder_outputs, encoder_sequence_length = self._build_encoder()

            # Decoder
            expert_fea = self._build_decoder(encoder_outputs, encoder_sequence_length, hparams)

            # Estimator
            est_logits_list = self._build_estimator(expert_fea, hparams)

            ## Loss
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                loss = self._compute_est_loss(est_logits_list)
            else:
                loss = None

            return loss, est_logits_list

    def _build_encoder(self):
        with tf.variable_scope("encoder"):
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, self.source)

            # Encoder_outputs: [max_time, batch_size, num_units]
            encoder_outputs, _, encoder_sequence_length = self.encoder.encode(
                encoder_emb_inp,
                self.src_sequence_length,
                mode=self.mode)

        return encoder_outputs, encoder_sequence_length

    def _build_decoder(self, encoder_outputs, encoder_sequence_length, hparams):
        fw_target_input = self.target

        fw_decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, fw_target_input)

        bw_decoder_emb_inp = tf.reverse_sequence(
            fw_decoder_emb_inp,
            self.tgt_sequence_length,
            batch_dim=0,
            seq_dim=1)

        # Decoder
        with tf.variable_scope("decoder"):
            with tf.variable_scope("fw_decoder"):
                fw_outputs, _, _ = self.fw_decoder.decode(
                    fw_decoder_emb_inp,
                    self.tgt_sequence_length,
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
                    self.tgt_sequence_length,
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
                self.tgt_sequence_length,
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
            sample_id = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

            # Extract logits feature for mismatching
            shape = tf.shape(fw_target_input)
            idx0 = tf.expand_dims(tf.range(shape[0]), -1)
            idx0 = tf.tile(idx0, [1, shape[1]])
            idx0 = tf.cast(idx0, fw_target_input.dtype)
            idx1 = tf.expand_dims(tf.range(shape[1]), 0)
            idx1 = tf.tile(idx1, [shape[0], 1])
            idx1 = tf.cast(idx1, fw_target_input.dtype)
            indices_real = tf.stack([idx0, idx1, fw_target_input], axis=-1)
            logits_mt = tf.gather_nd(logits, indices_real)
            logits_max = tf.reduce_max(logits, axis=-1)
            logits_diff = tf.subtract(logits_max, logits_mt)
            logits_same = tf.cast(tf.equal(sample_id, fw_target_input), tf.float32)
            logits_fea = tf.stack([logits_mt, logits_max, logits_diff, logits_same], axis=-1)

            # Extract QEFV
            output_layer_weights = self.output_layer.weights[0]
            _w = tf.nn.embedding_lookup(tf.transpose(output_layer_weights), fw_target_input)
            pre_qefv = _w * _pre_qefv
            post_qefv = tf.concat([fw_outputs, bw_outputs_rev], axis=-1)
            qefv = tf.concat([pre_qefv, post_qefv], axis=-1)

            expert_fea = tf.concat([qefv, logits_fea], axis=-1)

        return expert_fea

    def _build_estimator(self, estimator_inputs, hparams):
        if hparams.train_level != "sent" and hparams.use_hf:
            padded_tensor = tf.zeros_like(self.feat[:, :1])  # [B, 1, D] for <s>, </s>
            self.feat = tf.concat([padded_tensor, self.feat, padded_tensor], axis=1)
            estimator_inputs = tf.concat([estimator_inputs, self.feat], axis=-1)

        with tf.variable_scope("estimator") as scope:
            estimator_outputs, estimator_state, _ = self.estimator.encode(
                estimator_inputs,
                sequence_length=self.tgt_sequence_length,
                mode=self.mode)

            sent_logits, word_logits, gap_logits = None, None, None
            if hparams.train_level == "sent":
                with tf.variable_scope("sent_generator"):
                    sent_fea = estimator_state[-1]
                    if hparams.use_hf:
                        sent_fea = tf.concat([sent_fea, self.feat], axis=-1)
                    sent_logits = tf.layers.dense(sent_fea, 1)
                    sent_logits = tf.squeeze(sent_logits)
            elif hparams.train_level == "word":
                with tf.variable_scope("word_generator"):
                    word_fea = estimator_outputs[:, 1:-1]
                    # word_fea = tf.layers.dropout(word_fea, hparams.dropout,
                    #                              training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
                    word_logits = tf.layers.dense(word_fea, 2)
                    # word_logits = tf.squeeze(word_logits, squeeze_dims=-1)
            elif hparams.train_level == "gap":
                with tf.variable_scope("gap_generator"):
                    gap_fea = tf.concat((estimator_outputs[:, :-1], estimator_outputs[:, 1:]), axis=-1)
                    # gap_fea = tf.layers.dropout(gap_fea, hparams.dropout,
                    #                             training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
                    gap_logits = tf.layers.dense(gap_fea, 2)
                    # gap_logits = tf.squeeze(gap_logits, squeeze_dims=-1)

        return sent_logits, word_logits, gap_logits

    def _compute_est_loss(self, est_logits_list):
        """Compute optimization loss for estimator."""
        sent_logits, word_logits, gap_logits = est_logits_list
        sent_loss, word_loss, gap_loss = 0., 0., 0.
        if sent_logits is not None:
            sent_label = self.label
            sent_pred = tf.sigmoid(sent_logits)
            sent_loss += tf.reduce_mean(tf.square(sent_label - sent_pred))

        if word_logits is not None:
            word_label = tf.cast(self.label, tf.int32)
            word_crossent, word_normalizer, _ = onmt.utils.losses.cross_entropy_sequence_loss(
                word_logits,
                word_label,
                self.tgt_sequence_length - 2,
                self.label_smoothing,
                average_in_time=True,
                mode=self.mode)
            word_loss += word_crossent / word_normalizer

        if gap_logits is not None:
            gap_label = tf.cast(self.label, tf.int32)
            gap_crossent, gap_normalizer, _ = onmt.utils.losses.cross_entropy_sequence_loss(
                gap_logits,
                gap_label,
                self.tgt_sequence_length - 1,
                self.label_smoothing,
                average_in_time=True,
                mode=self.mode)
            gap_loss += gap_crossent / gap_normalizer

        est_loss = sent_loss + word_loss + gap_loss
        return est_loss

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
        return sess.run([self.infer_sent_prob,
                         self.infer_word_prob,
                         self.infer_gap_prob])


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
    if metric.lower() == "pearson":
        evaluation_score = _pearson(ref_file + sent_suffix, pred_file + ".sent")
    elif metric.lower() == "spearman":
        evaluation_score = _spearman(ref_file + sent_suffix, pred_file + ".sent")
    elif metric.lower() == "wordf1":
        evaluation_score = _f1(ref_file + word_suffix, pred_file + ".word")
    elif metric.lower() == "gapf1":
        evaluation_score = _f1(ref_file + word_suffix, pred_file + ".gap")
    else:
        raise ValueError("Unknown metric %s" % metric)
    return evaluation_score


def _pearson(grt_file, pre_file):
    pre = np.loadtxt(pre_file)
    grt = np.loadtxt(grt_file)
    return np.corrcoef(pre, grt)[0, 1]


def _spearman(grt_file, pre_file):
    pre = np.loadtxt(pre_file)
    grt = np.loadtxt(grt_file)
    return spearmanr(pre, grt).correlation


def _f1(grt_file, pre_file, threshold=0.5):
    true_tags, test_tags = [], []
    with open(grt_file, 'r') as fg:
        with open(pre_file, 'r') as fp:
            for lg, lp in zip(fg, fp):
                _lg = [int(x) for x in lg.strip().split()]
                _lp = [int(float(x) > threshold) for x in lp.strip().split()[:len(_lg)]]
                true_tags.append(_lg)
                test_tags.append(_lp)
    return compute_scores(true_tags, test_tags)


def list_of_lists(a_list):
    if isinstance(a_list, (list, tuple, np.ndarray)) and len(a_list) > 0 and all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return True
    return False


def flatten(lofl):
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(true_tags, test_tags):
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print_out("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    f1_mul = f1_bad * f1_good
    return f1_mul


def decode_and_evaluate(name,
                        model,
                        sess,
                        pred_file,
                        ref_file,
                        metrics):
    start_time = time.time()
    num_sentences = 0
    with codecs.getwriter("utf-8")(tf.gfile.GFile(pred_file + ".sent", mode="w")) as sent_f:
        sent_f.write("")
        with codecs.getwriter("utf-8")(tf.gfile.GFile(pred_file + ".word", mode="w")) as word_f:
            word_f.write("")
            with codecs.getwriter("utf-8")(tf.gfile.GFile(pred_file + ".gap", mode="w")) as gap_f:
                gap_f.write("")

                while True:
                    try:
                        sent_prob, word_prob, gap_prob = model.infer(sess)
                        if isinstance(sent_prob, np.ndarray):
                            batch_size = sent_prob.shape[0]
                            for sp in sent_prob.flatten():
                                sent_f.write(str(sp) + "\n")
                        if isinstance(word_prob, np.ndarray):
                            batch_size = word_prob.shape[0]
                            word_prob = word_prob[:, :, 1]
                            for sent_id in range(batch_size):
                                word_f.write(
                                    " ".join([str(_prob) for _prob in word_prob[sent_id].tolist()]) + "\n")
                        if isinstance(gap_prob, np.ndarray):
                            batch_size = gap_prob.shape[0]
                            gap_prob = gap_prob[:, :, 1]
                            for sent_id in range(batch_size):
                                gap_f.write(" ".join([str(_prob) for _prob in gap_prob[sent_id].tolist()]) + "\n")
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
        metrics=hparams.metrics)
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
        loaded_infer_model, global_step = create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer", hparams)

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_mt_file = "%s.%s" % (hparams.dev_prefix, hparams.mt)
    dev_fea_file = "%s.%s" % (hparams.dev_prefix, hparams.fea)
    dev_infer_iterator_feed_dict = {
        infer_model.src_placeholder: load_data(dev_src_file),
        infer_model.mt_placeholder: load_data(dev_mt_file),
        infer_model.fea_file_placeholder: dev_fea_file,
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    dev_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        dev_infer_iterator_feed_dict,
        dev_mt_file,
        "dev",
        summary_writer,
        save_on_best=save_best_dev,
        avg_ckpts=avg_ckpts)

    test_scores = None
    if use_test_set and hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_mt_file = "%s.%s" % (hparams.test_prefix, hparams.mt)
        test_fea_file = "%s.%s" % (hparams.test_prefix, hparams.fea)
        test_infer_iterator_feed_dict = {
            infer_model.src_placeholder: load_data(test_src_file),
            infer_model.mt_placeholder: load_data(test_mt_file),
            infer_model.fea_file_placeholder: test_fea_file,
            infer_model.batch_size_placeholder: hparams.infer_batch_size,
        }
        test_scores = _external_eval(
            loaded_infer_model,
            global_step,
            infer_sess,
            hparams,
            infer_model.iterator,
            test_infer_iterator_feed_dict,
            test_mt_file,
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
    return {"step_time": 0.0, "est_loss": 0.0, "grad_norm": 0.0, }


def update_stats(stats, start_time, step_result):
    """Update stats: write summary and accumulate statistics."""
    (_, step_loss, step_summary, global_step, grad_norm, learning_rate) = step_result
    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["est_loss"] += step_loss
    stats["grad_norm"] += grad_norm
    return global_step, learning_rate, step_summary


def print_step_info(prefix, global_step, info, result_summary, log_f):
    """Print all info at the current global step."""
    print_out("%sstep %d lr %g step-time %.2fs sel %.4f gN %.2f %s, %s" %
              (prefix, global_step, info["learning_rate"], info["avg_step_time"],
               info["avg_train_sel"], info["avg_grad_norm"], result_summary, time.ctime()), log_f)


def process_stats(stats, info, global_step, steps_per_stats, log_f):
    """Update info and check for overflow."""
    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["avg_train_sel"] = stats["est_loss"] / steps_per_stats
    avg_train_sel = info["avg_train_sel"]
    is_overflow = False
    if math.isnan(avg_train_sel) or math.isinf(avg_train_sel) or avg_train_sel > 1e20:
        print_out("  step %d overflow (sent loss), stop early" % global_step, log_f)
        is_overflow = True
    return is_overflow


def _get_best_results(hparams):
    """Summary of the current best results."""
    tokens = []
    for metric in hparams.metrics:
        tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
    return ", ".join(tokens)


def before_train(loaded_train_model, train_model, train_sess, global_step, hparams, log_f):
    """Misc tasks to do before training."""
    stats = init_stats()
    info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
            "avg_grad_norm": 0.0, "avg_train_sel": 0.0,
            "learning_rate": loaded_train_model.learning_rate.eval(
                session=train_sess)}
    start_train_time = time.time()
    print_out("# Start step %d, lr %g, %s" % (global_step, info["learning_rate"], time.ctime()), log_f)

    # Initialize all of the iterators
    skip_count = hparams.qe_batch_size * hparams.epoch_step
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

    model_creator = QEModel
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
        loaded_train_model, global_step = create_or_load_model(
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
    stats, info, start_train_time = before_train(loaded_train_model, train_model, train_sess, global_step, hparams,
                                                 log_f)
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
                os.path.join(out_dir, "qe.ckpt"),
                global_step=global_step)

        if global_step - last_external_eval_step >= steps_per_external_eval:
            last_external_eval_step = global_step

            # Save checkpoint
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(out_dir, "qe.ckpt"),
                global_step=global_step)
            run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer)

            if avg_ckpts:
                run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step)

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(out_dir, "qe.ckpt"),
        global_step=global_step)

    print_out("# Done training!", time.time() - start_train_time)

    summary_writer.close()
    return global_step


def inference(ckpt,
              inference_src_file,
              inference_mt_file,
              inference_fea_file,
              inference_output_file,
              hparams,
              scope=None):
    model_creator = QEModel
    infer_model = create_infer_model(model_creator, hparams, scope)

    """Inference with a single worker."""
    output_infer = inference_output_file
    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True

    with tf.Session(graph=infer_model.graph, config=config_proto) as sess:
        loaded_infer_model = load_model(infer_model.model, ckpt, sess, "infer")
        sess.run(infer_model.iterator.initializer,
                 feed_dict={infer_model.src_placeholder: load_data(inference_src_file),
                            infer_model.mt_placeholder: load_data(inference_mt_file),
                            infer_model.fea_file_placeholder: inference_fea_file,
                            infer_model.batch_size_placeholder: hparams.infer_batch_size})
        # Decode
        print_out("# Start inference")
        decode_and_evaluate(
            "infer",
            loaded_infer_model,
            sess,
            output_infer,
            ref_file=None,
            metrics=hparams.metrics)


############################## HPARAMS ########################################
FLAGS = None


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--embedding_size", type=int, default=64, help="src/rgt word embedding size")
    parser.add_argument("--num_units", type=int, default=512, help="Network size.")
    parser.add_argument("--rnn_units", type=int, default=128, help="RNN Network size.")
    parser.add_argument("--rnn_layers", type=int, default=1, help="RNN Network layers.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Transformer depth.")
    parser.add_argument("--dim_hf", type=int, default=0,
                        help="dimension of human feature for estimator")

    parser.add_argument("--num_heads", type=int, default=8,
                        help="Num of self-attention head.")
    parser.add_argument("--ffn_inner_dim", type=int, default=512,
                        help="ffn size in self-attention.")

    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")

    parser.add_argument("--use_hf", type="bool", nargs="?",
                        const=True, default=False,
                        help="whether to use human feature.")
    parser.add_argument("--fixed_exp", type="bool", nargs="?",
                        const=True, default=False,
                        help="fixed expert weights or not")
    parser.add_argument("--label_smoothing", type=float, default=0.,
                        help="smoothing label for cross entropy")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="How many steps we inverse-decay learning.")

    parser.add_argument("--train_level", type=str, default="sent",
                        help="sent, word or gap")
    parser.add_argument(
        "--num_train_steps", type=int, default=100000, help="Num steps to train.")

    # data
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix for QE, e.g., qe.en.")
    parser.add_argument("--mt", type=str, default=None,
                        help="MT suffix for QE, e.g., qe.de.")
    parser.add_argument("--fea", type=str, default=None,
                        help="Human feature suffix, e.g., qe.fea.")
    parser.add_argument("--lab", type=str, default=None,
                        help="Label suffix, e.g., qe.hter.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")
    parser.add_argument("--exp_model_dir", type=str, default=None,
                        help="Load pretrained expert model.")

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

    # Default settings
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--qe_batch_size", type=int, default=64, help="QE Batch size.")
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")

    # Misc
    parser.add_argument("--metrics", type=str, default="pearson,bleu",
                        help=("Comma-separated list of evaluations "
                              "metrics (bleu,rouge,accuracy)"))
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

    # Inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_src_file", type=str, default=None,
                        help="src data")
    parser.add_argument("--inference_mt_file", type=str, default=None,
                        help="mt data.")
    parser.add_argument("--inference_fea_file", type=str, default=None,
                        help="human feature data")
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file", type=str, default=None,
                        help=("""\
          Reference file to compute evaluation scores (if provided).\
          """))


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
        mt=flags.mt,
        fea=flags.fea,
        lab=flags.lab,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        max_vocab_size=flags.max_vocab_size,
        out_dir=flags.out_dir,
        exp_model_dir=flags.exp_model_dir,
        use_hf=flags.use_hf,
        fixed_exp=flags.fixed_exp,
        dim_hf=flags.dim_hf,
        train_level=flags.train_level,
        label_smoothing=flags.label_smoothing,

        # Networks
        embedding_size=flags.embedding_size,
        num_units=flags.num_units,
        rnn_units=flags.rnn_units,
        num_layers=flags.num_layers,  # Compatible
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        num_heads=flags.num_heads,
        ffn_inner_dim=flags.ffn_inner_dim,
        dropout=flags.dropout,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        qe_batch_size=flags.qe_batch_size,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,

        # Data constraints
        num_buckets=flags.num_buckets,
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
    )


def extend_hparams(hparams):
    # Get vocab file names first
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.mt
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
            data_file=hparams.train_prefix + "." + hparams.mt,
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


def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
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

    if flags.inference_src_file and flags.inference_mt_file and flags.inference_fea_file:
        ckpt = tf.train.latest_checkpoint(flags.ckpt)
        inference_fn(ckpt,
                     flags.inference_src_file,
                     flags.inference_mt_file,
                     flags.inference_fea_file,
                     flags.inference_output_file,
                     hparams)
    else:
        train_fn(hparams, target_session=target_session)


def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    run_main(FLAGS, default_hparams, train, inference)


if __name__ == "__main__":
    qe_parser = argparse.ArgumentParser()
    add_arguments(qe_parser)
    FLAGS, unparsed = qe_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

