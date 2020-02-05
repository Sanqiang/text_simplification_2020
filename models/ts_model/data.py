import os
import numpy as np
import tensorflow as tf
from language_model.gpt2 import encoder
from language_model.bert import tokenization
from models.utils.control_utils import ControlMethod


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
        self.more_tokens = []
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.pad_id = self.tokenizer.vocab[self.SYMBOL_PAD]
        self.eos_id = self.tokenizer.vocab[self.SYMBOL_EOS]
        self.go_id = self.tokenizer.vocab[self.SYMBOL_GO]

    def encode_token(self, token):
        return self.tokenizer.vocab[token]

    def encode_sent(self, sent):
        return [self.tokenizer.vocab[id] for id in self.tokenizer.tokenize(sent)]

    def encode_sent_stack(self, sent):
        return self.tokenizer.tokenize_stack(sent, self.tokenizer.vocab)

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

        if self.flags.control_mode:
            self.feature_set.update(
                {'control_wds': tf.FixedLenFeature([], tf.string),
                 'control_vec': tf.FixedLenFeature([8], tf.float32)})

        if 'syntax_gen' in self.flags.control_mode:
            self.feature_set.update(
                {'template_comp_full': tf.FixedLenFeature([], tf.string),
                 'template_simp_full': tf.FixedLenFeature([], tf.string)})

        if 'syntax_gen' in self.flags.control_mode:
            self.syntax_vocab = BertVocab(flags.syntax_vocab_file)

        if 'gpt2_vocab' in self.flags.model_mode:
            self.vocab = GPT2Vocab(flags.models_dir, flags.model_name)
        elif 'bert_vocab' in self.flags.model_mode:
            self.vocab = BertVocab(flags.bert_vocab_file)

        self.control_vec_len = 0
        if "rel" in self.flags.control_mode:
            self.control_vec_len += 1
        if "sent_length" in self.flags.control_mode:
            self.control_vec_len += 1
        if "word_length" in self.flags.control_mode:
            self.control_vec_len += 1
        if "syntax" in self.flags.control_mode:
            self.control_vec_len += 1
        if "split" in self.flags.control_mode:
            self.control_vec_len += 1
        if "ppdb" in self.flags.control_mode:
            self.control_vec_len += 1

        # self.reserve_dimension = 0
        # self.control_vec_dimension = self.flags.dimension
        # if 'predict' in self.flags.control_mode:
        #     self.reserve_dimension = self.flags.dimension // 4
        #     self.control_vec_dimension = self.flags.dimension - self.reserve_dimension

    def update_data_for_train(self):
        pass

    def update_data_for_eval(self):
        if self.flags.control_mode:
            self.control_obj = ControlMethod(self.flags)

    def _parse(self, features, is_training):
        def _py_process_line_pair(
                src_wds, trg_wds, control_wds, control_vec,
                template_comp, template_simp):
            src_wds, trg_wds = src_wds.decode(), trg_wds.decode()
            extra_vec = [-1]

            if self.flags.control_mode:
                if not is_training:
                    # Update control_vec/control_wds for eval for control mode
                    intput_control_vec, extra_outputs = self.control_obj.get_control_vec_eval(
                        src_wds,
                        rel=float(self.flags.control_mode['rel'] if "rel" in self.flags.control_mode else 0),
                        sent_length=float(self.flags.control_mode['sent_length']
                                          if "sent_length" in self.flags.control_mode else 0),
                        word_length=float(self.flags.control_mode['word_length']
                                          if "word_length" in self.flags.control_mode else 0),
                        syntax=float(self.flags.control_mode['syntax']
                                     if "syntax" in self.flags.control_mode else 0),
                        split=float(self.flags.control_mode['split']
                                    if "split" in self.flags.control_mode else 0),
                        ppdb=float(self.flags.control_mode['ppdb']
                                   if "ppdb" in self.flags.control_mode else 0),
                    )
                else:
                    intput_control_vec = {}
                    if "rel" in self.flags.control_mode:
                        intput_control_vec["rel"] = control_vec[0]
                    if "sent_length" in self.flags.control_mode:
                        intput_control_vec["sent_length"] = control_vec[1]
                    if "word_length" in self.flags.control_mode:
                        intput_control_vec["word_length"] = control_vec[2]
                    if "syntax" in self.flags.control_mode:
                        intput_control_vec["syntax"] = control_vec[3]
                    if "split" in self.flags.control_mode:
                        intput_control_vec["split"] = control_vec[4]
                    if "ppdb" in self.flags.control_mode:
                        intput_control_vec["ppdb"] = control_vec[5]

            output_control_vec = []
            if "rel" in self.flags.control_mode:
                output_control_vec.append(intput_control_vec["rel"])
            if "sent_length" in self.flags.control_mode:
                output_control_vec.append(intput_control_vec["sent_length"])
            if "word_length" in self.flags.control_mode:
                output_control_vec.append(intput_control_vec["word_length"])
            if "syntax" in self.flags.control_mode:
                output_control_vec.append(intput_control_vec["syntax"])
            if "split" in self.flags.control_mode:
                output_control_vec.append(intput_control_vec["split"])
            if "ppdb" in self.flags.control_mode:
                output_control_vec.append(intput_control_vec["ppdb"])


            # if self.flags.control_mode:
            #     if 'predict' in self.flags.control_mode:
            #         length_score = control_vec[0]
            #         extra_vec[0] = length_score
            #
            #         syntax_score = control_vec[1]
            #         extra_vec[1] = syntax_score
            #
            #         split_score = control_vec[2]
            #         extra_vec[2] = split_score
            #     else:
            #         assert len(control_vec) == 4

            control_ids = self.vocab.encode_sent(control_wds)
            if "scatter_ppdb" in self.flags.control_mode and len(control_ids):
                control_ids_unit = control_ids
                while len(control_ids) + len(control_ids_unit) < self.flags.max_ppdb_len:
                    control_ids.extend(control_ids_unit)

            control_ids = _pad_sent(
                control_ids,
                self.vocab.pad_id, self.vocab.eos_id, self.flags.max_ppdb_len)
            #
            # template_comp_ids, template_simp_ids = \
            #     [self.syntax_vocab.pad_id] * self.flags.max_syntax_src_len, \
            #     [self.syntax_vocab.pad_id] * self.flags.max_syntax_trg_len
            template_comps, template_simps = [[] for _ in range(self.flags.syntax_level)], \
                                             [[] for _ in range(self.flags.syntax_level)]
            if 'syntax_gen' in self.flags.control_mode:
                template_comp, template_simp = template_comp.decode(), template_simp.decode()

                for template_comp_tk in template_comp.split():
                    template_comp_tk_stacked_list = template_comp_tk.split('|')
                    for i in range(self.flags.syntax_level):
                        if i < len(template_comp_tk_stacked_list):
                            template_comps[i].append(template_comp_tk_stacked_list[i])
                        else:
                            template_comps[i].append(
                                template_comp_tk_stacked_list[len(template_comp_tk_stacked_list) - 1])

                for template_simp_tk in template_simp.split():
                    template_simp_tk_stacked_list = template_simp_tk.split('|')
                    for i in range(self.flags.syntax_level):
                        if i < len(template_simp_tk_stacked_list):
                            template_simps[i].append(template_simp_tk_stacked_list[i])
                        else:
                            template_simps[i].append(
                                template_simp_tk_stacked_list[len(template_simp_tk_stacked_list) - 1])

                # template_comp_stacked_ids = self.syntax_vocab.encode_sent_stack(template_comp)
                # template_simp_stacked_ids = self.syntax_vocab.encode_sent_stack(template_simp)

                # template_simp_ids = self.syntax_vocab.encode_sent(template_simp)
                # template_comp_ids = _pad_sent(
                #     template_comp_ids,
                #     self.syntax_vocab.pad_id, self.syntax_vocab.eos_id, self.flags.max_syntax_src_len)
                # template_simp_ids = _pad_sent(
                #     template_simp_ids,
                #     self.syntax_vocab.pad_id, self.syntax_vocab.eos_id, self.flags.max_syntax_trg_len)


            src_stacked_ids = self.vocab.encode_sent_stack(src_wds)
            trg_stacked_ids = self.vocab.encode_sent_stack(trg_wds)
            # trg_ids = _pad_sent(self.vocab.encode_sent(trg_wds),
            #               self.vocab.pad_id, self.vocab.eos_id, self.flags.max_trg_len)
            # _pad_sent(,
            #           self.vocab.pad_id, self.vocab.eos_id, self.flags.max_src_len),

            if 'syntax_gen' in self.flags.control_mode:
                # For comp
                template_comp_ids, src_ids = [[] for _ in range(self.flags.syntax_level)], []

                for l_id, template_comp in enumerate(template_comps):
                    for i, template_tk in enumerate(template_comp):
                        if l_id == 0:
                            src_ids.extend(src_stacked_ids[i])
                        template_comp_ids[l_id].append(self.syntax_vocab.encode_token(template_tk))

                for i in range(len(template_comp_ids)):
                    template_comp_ids[i] = _pad_sent(
                        template_comp_ids[i],
                        self.syntax_vocab.pad_id, self.syntax_vocab.eos_id, self.flags.max_syntax_src_len)

                # For simp
                template_simp_ids, trg_ids = [[] for _ in range(self.flags.syntax_level)], []

                for l_id, template_simp in enumerate(template_simps):
                    for i, template_tk in enumerate(template_simp):
                        if l_id == 0:
                            trg_ids.extend(trg_stacked_ids[i])
                        template_simp_ids[l_id].append(self.syntax_vocab.encode_token(template_tk))

                for i in range(len(template_simp_ids)):
                    template_simp_ids[i] = _pad_sent(
                        template_simp_ids[i],
                        self.syntax_vocab.pad_id, self.syntax_vocab.eos_id, self.flags.max_syntax_trg_len)
            else:
                src_ids = [item for sublist in src_stacked_ids for item in sublist]
                template_comp_ids = [] # Not used
                trg_ids = [item for sublist in trg_stacked_ids for item in sublist]
                template_simp_ids = []  # Not used

            src_ids, trg_ids = (
                _pad_sent(
                    src_ids, self.vocab.pad_id, self.vocab.eos_id, self.flags.max_src_len),
                _pad_sent(
                    trg_ids, self.vocab.pad_id, self.vocab.eos_id, self.flags.max_trg_len))

            return (np.array(src_ids, np.int32),
                    np.array(trg_ids, np.int32),
                    np.array(control_ids, np.int32),
                    np.array(output_control_vec, np.float32),
                    np.array(extra_vec, np.float32),
                    np.array(template_comp_ids, np.int32),
                    np.array(template_simp_ids, np.int32))

        src_wds, trg_wds = features['src_wds'], features['trg_wds']
        if self.flags.control_mode:
            control_wds = features['control_wds']
            if is_training:
                control_vec = features['control_vec']
            else:
                control_vec = tf.zeros([1, ])
        else:
            # Just placeholder
            control_wds = src_wds
            control_vec = tf.zeros([1, ])

        if "syntax_gen" in self.flags.control_mode:
            template_comp = features['template_comp_full']
            template_simp = features['template_simp_full']
        else:
            # Just placeholder
            template_comp = src_wds
            template_simp = src_wds

        src_ids, trg_ids, control_ids, control_vec, extra_vec, template_comp_ids, template_simp_ids = tf.py_func(
            _py_process_line_pair,
            [src_wds, trg_wds, control_wds, control_vec, template_comp, template_simp],
            [tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32])
        src_ids.set_shape(
            [self.flags.max_src_len])
        trg_ids.set_shape(
            [self.flags.max_trg_len])
        control_ids.set_shape(
            [self.flags.max_ppdb_len])
        control_vec.set_shape(
            [self.control_vec_len])
        extra_vec.set_shape([1])
        template_comp_ids.set_shape(
            [self.flags.syntax_level, self.flags.max_syntax_src_len])
        template_simp_ids.set_shape(
            [self.flags.syntax_level, self.flags.max_syntax_trg_len])
        output = {
            'src_ids': src_ids,
            'trg_ids': trg_ids,
            'control_ids': control_ids,
            'control_vec': control_vec,
            'extra_vec': extra_vec,
            'template_comp_ids': template_comp_ids,
            'template_simp_ids': template_simp_ids,
        }
        return output

    def get_input_fn(self, is_training, input_files, num_cpu_threads):

        def input_fn(params):
            batch_size = params['batch_size']
            if is_training:
                files = []
                for input_pattern in input_files.split(','):
                    files.extend(tf.gfile.Glob(input_pattern))
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
                    lambda record:  self._parse(tf.parse_single_example(record, self.feature_set), is_training),
                    batch_size=batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=is_training))
            return d

        return input_fn
