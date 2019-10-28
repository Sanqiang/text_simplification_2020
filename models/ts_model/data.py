import os
import numpy as np
import tensorflow as tf
from language_model.gpt2 import encoder
from language_model.bert import tokenization


def _clean_sent_ids(ids, eos_id):
    if eos_id in ids:
        eid = ids.index(eos_id)
        return ids[:eid]
    return ids


def _pad_sent(ids, pad_id, eos_id, length):
    ids.append(eos_id)
    if len(ids) >= length:
        ids = ids[:length]
    else:
        cnt_pad = length - len(ids)
        ids.extend([pad_id] * cnt_pad)
    return ids


class BertVocab:
    def __init__(self, vocab_file):
        self.SYMBOL_GO = '[CLS]'
        self.SYMBOL_EOS = '[SEP]'
        self.SYMBOL_PAD = '[PAD]'
        self.more_tokens  = []
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.pad_id = self.tokenizer.vocab[self.SYMBOL_PAD]
        self.eos_id = self.tokenizer.vocab[self.SYMBOL_EOS]
        self.go_id = self.tokenizer.vocab[self.SYMBOL_GO]

    def encode_token(self, token):
        return self.tokenizer.vocab[token]

    def encode_sent(self, sent):
        return [self.tokenizer.vocab[id] for id in self.tokenizer.tokenize(sent)]

    def decode_token(self, id):
        return self.tokenizer.inv_vocab[id]

    def decode_sent(self, ids):
        sent = []
        for id in _clean_sent_ids(ids, self.eos_id):
            wp = self.tokenizer.inv_vocab[id]
            if wp.startswith("##") and len(sent) > 0:
                sent[-1] += wp[2:]
            else:
                sent.append(wp)
        return ' '.join(sent)

    def size(self):
        return len(self.tokenizer.vocab) - len(self.more_tokens)


class GPT2Vocab:
    def __init__(self, models_dir='', model_name='774M'):
        self.SYMBOL_GO = '<|gooftext|>'
        self.SYMBOL_EOS = '<|endoftext|>'
        self.SYMBOL_PAD = '<|padoftext|>'
        self.more_tokens = [self.SYMBOL_GO, self.SYMBOL_PAD]
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        self.enc = encoder.get_encoder(
            model_name, models_dir,
            more_tokens=self.more_tokens)
        self.pad_id = self.encode_token(self.SYMBOL_PAD)
        self.eos_id = self.encode_token(self.SYMBOL_EOS)
        self.go_id = self.encode_token(self.SYMBOL_GO)

    def encode_token(self, token):
        return self.enc.encoder[token]

    def encode_sent(self, sent):
        return self.enc.encode(sent)

    def decode_token(self, id):
        return self.enc.decoder[id]

    def decode_sent(self, ids):
        return self.enc.decode(_clean_sent_ids(ids, self.eos_id))

    def size(self):
        return len(self.enc.encoder) - len(self.more_tokens)


class Data:
    def __init__(self, flags):
        self.flags = flags
        self.feature_set = {
            'src_wds': tf.FixedLenFeature([], tf.string),
            'trg_wds': tf.FixedLenFeature([], tf.string),
        }
        if 'gpt2_vocab' in self.flags.model_mode:
            self.vocab = GPT2Vocab(flags.models_dir, flags.model_name)
        elif 'bert_vocab' in self.flags.model_mode:
            self.vocab = BertVocab(flags.bert_vocab_file)

    def _parse(self, features):
        def _py_process_line_pair(src_wds, trg_wds):
            src_wds, trg_wds = src_wds.decode(), trg_wds.decode()
            # print('src_wds:%s' % src_wds)
            # print('trg_wds:%s' % trg_wds)
            # print('=====')
            src_ids, trg_ids = (
                _pad_sent(self.vocab.encode_sent(src_wds),
                          self.vocab.pad_id, self.vocab.eos_id, self.flags.max_src_len),
                _pad_sent(self.vocab.encode_sent(trg_wds),
                          self.vocab.pad_id, self.vocab.eos_id, self.flags.max_trg_len))
            return np.array(src_ids, np.int32), np.array(trg_ids, np.int32)

        src_wds, trg_wds = features['src_wds'], features['trg_wds']
        src_ids, trg_ids = tf.py_func(
            _py_process_line_pair,
            [src_wds, trg_wds],
            [tf.int32, tf.int32])
        src_ids.set_shape(
            [self.flags.max_src_len])
        trg_ids.set_shape(
            [self.flags.max_trg_len])
        output = {
            'src_ids': src_ids,
            'trg_ids': trg_ids,
        }
        return output

    def get_input_fn(self, is_training, input_files, num_cpu_threads):

        def input_fn(params):
            batch_size = params['batch_size']
            if is_training:
                files = tf.gfile.Glob(input_files)
                tf.logging.info('Input files: %s' % files)
                d = tf.data.Dataset.from_tensor_slices(tf.constant(files))
                d = d.repeat()
                d = d.shuffle(buffer_size=len(files))
                cycle_length = min(num_cpu_threads, len(files))
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=is_training,
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=100)
            else:
                d = tf.data.TFRecordDataset(input_files)

            d = d.apply(
                tf.data.experimental.map_and_batch(
                    lambda record:  self._parse(tf.parse_single_example(record, self.feature_set)),
                    batch_size=batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=is_training))
            return d

        return input_fn
