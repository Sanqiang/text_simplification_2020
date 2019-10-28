import os
import json
import tensorflow as tf
from tensor2tensor.models import transformer
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import beam_search
from language_model.gpt2 import model
from models.ts_model.seq_loss import sequence_loss


class TsGraph:
    def __init__(self, flags, is_training, data):
        self.flags = flags
        self.is_training = is_training
        self.data = data
        if 'gpt2' in self.flags.model_mode:
            self.hparams = model.default_hparams()
            with open(os.path.join(
                    self.flags.models_dir, self.flags.model_name, 'hparams.json')) as f:
                self.hparams.override_from_dict(json.load(f))
            self.hparams.hidden_size = self.hparams.n_embd
        elif 't2t' in self.flags.model_mode:
            self.hparams = transformer.transformer_base()
            self._setup_hparams()
        self._init_shared_variables()

    def _embedding_fn(self, inputs):
        if type(inputs) == list:
            if not inputs:
                return []
            else:
                return [tf.nn.embedding_lookup(
                    self.shared_tensors['word_embedding_table'], inp) for inp in inputs]
        else:
            return tf.nn.embedding_lookup(
                self.shared_tensors['word_embedding_table'], inputs)

    def _init_shared_variables(self):
        self.shared_tensors = {}
        self.word_embedding_table = tf.get_variable(
            'word_embedding_table',
            shape=[self.data.vocab.size(), self.hparams.hidden_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True, dtype=tf.float32)
        if len(self.data.vocab.more_tokens) > 0:
            self.more_word_embedding_table = tf.get_variable(
                'more_word_embedding_table',
                shape=[len(self.data.vocab.more_tokens), self.hparams.hidden_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True, dtype=tf.float32)
            self.word_embedding_table = tf.concat(
                [self.word_embedding_table, self.more_word_embedding_table], axis=0)
        self.shared_tensors['word_embedding_table'] = self.word_embedding_table

        temp_act_embedding_proj = tf.get_variable(
            'proj_word_w_trans', [self.hparams.hidden_size, self.hparams.hidden_size], tf.float32)
        self.proj_word_w = tf.transpose(
            tf.matmul(self.word_embedding_table, temp_act_embedding_proj))
        self.proj_word_b = tf.get_variable(
            'proj_word_b', [self.data.vocab.size() + len(self.data.vocab.more_tokens)], tf.float32)
        self.shared_tensors['proj_word_w'] = self.proj_word_w
        self.shared_tensors['proj_word_b'] = self.proj_word_b

    def _setup_hparams(self):
        self.hparams.hidden_size = self.flags.dimension

        self.hparams.num_heads = self.flags.num_heads
        self.hparams.num_hidden_layers = self.flags.num_hidden_layers

        if self.is_training:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0

    def _tile_variables(self):
        """
        Tile variables for beam search
        # E.g. [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        :return:
        """
        src_ids = self.shared_tensors['src_ids']
        src_ids = tf.concat(
            [tf.tile(tf.expand_dims(src_ids[o, :], axis=0),
                     [self.flags.beam_search_size, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_ids'] = src_ids

        src_outputs = self.shared_tensors['src_outputs']
        src_outputs = tf.concat(
            [tf.tile(tf.expand_dims(src_outputs[o, :, :], axis=0),
                     [self.flags.beam_search_size, 1, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_outputs'] = src_outputs

        src_bias = self.shared_tensors['src_bias']
        src_bias = tf.concat(
            [tf.tile(tf.expand_dims(src_bias[o, :, :, :], axis=0),
                     [self.flags.beam_search_size, 1, 1, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_bias'] = src_bias

        batch_go = self.shared_tensors['batch_go']
        batch_go = tf.concat(
            [tf.tile(tf.expand_dims(batch_go[o, :], axis=0),
                     [self.flags.beam_search_size, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['batch_go'] = batch_go

    def decode_srcs_to_trgs(self, trg_emb):
        trg_emb = common_attention.add_timing_signal_1d(trg_emb)
        trg_length = tf.shape(trg_emb)[1]
        if 'gpt2' in self.flags.model_mode:
            trg_outputs = model.gpt2_decoder(
                self.hparams, trg_emb,
                encoder_outputs=self.shared_tensors['src_outputs'],
                encoder_bias=self.shared_tensors['src_bias'])
        elif 't2t' in self.flags.model_mode:
            trg_self_attention_bias = common_attention.attention_bias_lower_triangle(
                trg_length)
            trg_outputs = transformer.transformer_decoder(
                decoder_input=trg_emb,
                decoder_self_attention_bias=trg_self_attention_bias,
                encoder_output=self.shared_tensors['src_outputs'],
                encoder_decoder_attention_bias=self.shared_tensors['src_bias'],
                hparams=self.hparams,
                name='trg_decoder')
        else:
            raise ValueError('model_mode not known')
        return trg_outputs

    def build(self, features):
        src_ids = features['src_ids']
        trg_ids = None
        self.batch_size = tf.shape(src_ids)[0]
        if self.is_training:
            trg_ids = features['trg_ids']

        with tf.variable_scope('src_encoder'):
            self.shared_tensors['src_ids'] = src_ids
            src_mask = tf.cast(
                tf.equal(src_ids, self.data.vocab.pad_id), tf.float32)
            src_bias = common_attention.attention_bias_ignore_padding(src_mask)
            self.shared_tensors['src_bias'] = src_bias

            src_embs = self._embedding_fn(src_ids)
            src_embs = common_attention.add_timing_signal_1d(src_embs)
            if 'gpt2' in self.flags.model_mode:
                src_outputs = model.gpt2_encoder(self.hparams, src_embs, encoder_bias=src_bias)
            elif 't2t' in self.flags.model_mode:
                src_outputs = transformer.transformer_encoder(src_embs, src_bias, self.hparams)
            else:
                raise ValueError('model_mode not known.')
            self.shared_tensors['src_outputs'] = src_outputs

        batch_go = tf.tile(
            tf.expand_dims(
                self._embedding_fn(
                    self.data.vocab.go_id),
                axis=0),
            [self.batch_size, 1])
        self.shared_tensors['batch_go'] = batch_go

        outputs = {}
        with tf.variable_scope("trg_decoder"):
            if self.is_training:
                trg_ids_list = tf.unstack(trg_ids, axis=1)
                trg_emb_list = [self.shared_tensors['batch_go']] + self._embedding_fn(
                    trg_ids_list[:-1])
                trg_emb = tf.stack(trg_emb_list, axis=1)

                decoder_outputs = self.decode_srcs_to_trgs(
                    trg_emb=trg_emb)
                word_logits = tf.nn.conv1d(
                    decoder_outputs,
                    tf.expand_dims(self.shared_tensors['proj_word_w'], axis=0),
                    1, 'SAME') + tf.expand_dims(
                    tf.expand_dims(
                        self.shared_tensors['proj_word_b'], axis=0), axis=0)
                word_gen = tf.argmax(word_logits, axis=-1)
                outputs['gen'] = word_gen
                outputs['logits'] = word_logits

                weight = tf.cast(
                    tf.not_equal(trg_ids, self.data.vocab.pad_id), tf.float32)
                loss = sequence_loss(logits=word_logits, targets=trg_ids, weights=weight)
                outputs['loss'] = loss
                outputs['perplexity'] = tf.exp(loss)
            else:
                self._tile_variables()

                def symbol_to_logits_fn(gen_ids):
                    cur_ids = gen_ids[:, 1:]
                    cur_embs = tf.nn.embedding_lookup(
                        self.shared_tensors['word_embedding_table'], cur_ids)
                    cur_embs = tf.concat(
                        [tf.expand_dims(self.shared_tensors['batch_go'], axis=1), cur_embs],
                        axis=1)
                    cur_outputs = self.decode_srcs_to_trgs(
                        trg_emb=cur_embs)
                    cur_logit = tf.matmul(
                        cur_outputs[:, -1, :], self.shared_tensors['proj_word_w']) + self.shared_tensors['proj_word_b']
                    return cur_logit

                beam_ids, beam_score = beam_search.beam_search(
                    symbols_to_logits_fn=symbol_to_logits_fn,
                    initial_ids=tf.ones(
                        [self.flags.eval_batch_size], tf.int32) * self.data.vocab.go_id,
                    beam_size=self.flags.beam_search_size,
                    decode_length=self.flags.max_trg_len,
                    vocab_size=self.data.vocab.size() + len(self.data.vocab.more_tokens),
                    alpha=0.6,
                    eos_id=self.data.vocab.eos_id
                )
                top_beam_ids = beam_ids[:, 0, 1:]
                top_beam_ids = tf.pad(top_beam_ids,
                                      [[0, 0],
                                       [0, self.flags.max_trg_len - tf.shape(top_beam_ids)[1]]])
                confident_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])
                outputs['gen_trg_ids'] = top_beam_ids
                outputs['gen_trg_scores'] = confident_score

        return outputs
