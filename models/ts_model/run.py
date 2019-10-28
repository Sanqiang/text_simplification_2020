import os
import numpy as np
import glob
import tensorflow as tf
from models.ts_model import data as data_static
from models.ts_model.data import Data
from models.ts_model.graph import TsGraph
from models.utils import ckpt_utils, sari_utils, mteval_bleu, hook_utils, restore_utils
from nltk.translate.bleu_score import sentence_bleu


flags = tf.flags

flags.DEFINE_string(
    "name", "d6",
    "Name of experiment")

flags.DEFINE_string(
    "mode", "train",
    "choice of train/infer")

flags.DEFINE_string(
    "model_mode", "gpt2_vocab:gpt2",
    "mode of model, e.g. gpt2:t2t:bert")

flags.DEFINE_string(
    "init_ckpt_path", None,
    "The Checkpoint for warm start.")

flags.DEFINE_string(
    "exp_dir", "/Users/sanqiang/git/ts/exp/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "train_tfexample",
    "/Users/sanqiang/git/ts/ts_2020_data/wikilarge/test/*.example",
    "The path pattern of train tf.Example files.")

flags.DEFINE_integer(
    "num_train_steps", 1000000,
    "Number of training step."
)

flags.DEFINE_integer(
    "num_warmup_steps", 10000,
    "Number of training step."
)

flags.DEFINE_integer(
    "train_batch_size", 3,
    "Size of minibatch."
)

flags.DEFINE_integer(
    "max_src_len", 30,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_trg_len", 25,
    "Maximum length of sentence."
)

flags.DEFINE_float(
    "lr", 0.1, "Learning rate.")

flags.DEFINE_integer(
    "beam_search_size", 1,
    "The size of beam search."
)

# For t2t
flags.DEFINE_integer(
    "dimension", 32,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "num_heads", 8,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "num_hidden_layers", 6,
    "Maximum length of sentence."
)

# For Inference
flags.DEFINE_integer(
    "num_ref", 3,
    "Number of reference files.")

flags.DEFINE_integer(
    "eval_batch_size", 1,
    "Size of minibatch."
)

# For BERT
flags.DEFINE_string(
    "bert_vocab_file", "/Users/sanqiang/git/ts/text_simplification_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt",
    "The file path of bert vocab")

# For GPT2
flags.DEFINE_string(
    "gpt2_ckpt_path", None,
    "The Checkpoint for warm start of GPT2.")

flags.DEFINE_string(
    'model_name', '117M',
    'The name of model, e.g. 774M')

flags.DEFINE_string(
    'models_dir', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/models',
    'the folder of model.')

# Won't change a lot
flags.DEFINE_integer(
    "num_cpu", 5,
    "Number of CPUs used for processing data."
)

flags.DEFINE_string(
    "infer_tfexample",
    "/Users/Shared/Previously Relocated Items/Security/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/eval.tfexample",
    "The path pattern of train tf.Example files.")

flags.DEFINE_string(
    "infer_src_file", "/Users/sanqiang/git/ts/text_simplification_edit/data/dummy_data/eval_src.txt",
    "The path of reference files.")

flags.DEFINE_string(
    "infer_ref_file", "/Users/sanqiang/git/ts/text_simplification_edit/data/dummy_data/eval_trg",
    "The path of reference files.")

FLAGS = flags.FLAGS


def model_fn_builder(data, init_ckpt_path=None):

    def model_fn(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        model = TsGraph(FLAGS, is_training, data)
        outputs = model.build(features)
        global_step = tf.train.get_or_create_global_step()
        tvars = tf.trainable_variables()

        if is_training:
            if init_ckpt_path:
                (assignment_map, initialized_variable_names
                 ) = restore_utils.get_assignment_map_from_checkpoint(tvars, init_ckpt_path)
                tf.train.init_from_checkpoint(init_ckpt_path, assignment_map)
                tf.logging.info('Init from %s' % init_ckpt_path)

            if 'gpt2' in FLAGS.model_mode and FLAGS.gpt2_ckpt_path:
                assignment_map = restore_utils.get_gpt2_assignment_map_from_checkpoint(tvars, FLAGS.gpt2_ckpt_path)
                tf.train.init_from_checkpoint(FLAGS.gpt2_ckpt_path, assignment_map)
                tf.logging.info('Init GPT2 from %s' % FLAGS.gpt2_ckpt_path)

        if mode == tf.estimator.ModeKeys.TRAIN:
            grads = tf.gradients(outputs['loss'], tvars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            learning_rate = tf.constant(
                value=FLAGS.lr, shape=[], dtype=tf.float32)
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                FLAGS.num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)
            if FLAGS.num_warmup_steps:
                global_steps_int = tf.cast(global_step, tf.int32)
                warmup_steps_int = tf.constant(FLAGS.num_warmup_steps, dtype=tf.int32)

                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

                warmup_percent_done = global_steps_float / warmup_steps_float
                warmup_learning_rate = FLAGS.lr * warmup_percent_done

                is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
                learning_rate = (
                        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
            # optimizer = tf.contrib.opt.LARSOptimizer(learning_rate)
            optimizer = tf.train.AdagradOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step)
            new_global_step = global_step + 1
            train_op = tf.group(train_op, [global_step.assign(new_global_step)])
            vars_to_hook = {
                "global_step": global_step,
                "learning_rate": learning_rate,
                "perplexity": outputs['perplexity'],
                "total_loss": outputs['loss'],}

            logging_hook = hook_utils.LoggingTensorHook(
                vars_to_hook,
                every_n_iter=100)

            step_hook = hook_utils.StepCounterHook(
                batch_size=FLAGS.train_batch_size)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=outputs['loss'],
                train_op=train_op,
                training_hooks=[logging_hook, step_hook])
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    'gen_trg_ids': outputs['gen_trg_ids'],
                    'gen_trg_scores': outputs['gen_trg_scores'],
                    'src_ids': features['src_ids']
                })

        return output_spec

    return model_fn


def train(data, estimator):
    train_input_fn = data.get_input_fn(
        input_files=FLAGS.train_tfexample,
        num_cpu_threads=FLAGS.num_cpu,
        is_training=True)
    estimator.train(input_fn=train_input_fn)


def repeat_infer(data, estimator, log_dir, model_dir, result_dir):
    src_lines = [line.strip() for line in open(FLAGS.infer_src_file).readlines()]
    ref_lines_tmp = [[] for _ in range(FLAGS.num_ref)]
    for rid in range(FLAGS.num_ref):
        ref_lines_tmp[rid] = [line.strip() for line
                              in open(FLAGS.infer_ref_file + str(rid)).readlines()]
    ref_lines = []
    for inst_id in range(len(ref_lines_tmp[0])):
        ref_lines.append(
            [ref_lines_tmp[rid][inst_id] for rid in range(FLAGS.num_ref)])

    best_score = ckpt_utils.get_best_score(result_dir)
    while True:
        ckpt = ckpt_utils.get_ckpt(model_dir, log_dir)

        eval_input_fn = data.get_input_fn(
            input_files=FLAGS.infer_tfexample,
            num_cpu_threads=FLAGS.num_cpu,
            is_training=False)
        results = estimator.predict(
            input_fn=eval_input_fn,
            checkpoint_path=ckpt)
        global_step = int(ckpt[ckpt.rindex('-') + 1:])
        reports = []
        gen_trg_scores = []
        all_generated_sents = []

        for inst_id, result in enumerate(results):
            gen_trg_sent = data.vocab.decode_sent(list(result['gen_trg_ids']))
            all_generated_sents.append(gen_trg_sent)
            gen_trg_score = result['gen_trg_scores']
            # src_sent = data_static._clean_sent(
            #     result['src_ids'], data.vocab.SYMBOL_PAD)

            gen_trg_scores.append(gen_trg_score)
            try:
                sari = sari_utils.SARIsent(src_lines[inst_id],
                                           gen_trg_sent,
                                           ref_lines[inst_id])
            except:
                sari = 0.0

            try:
                bleu = sentence_bleu(
                    [sent.split() for sent in ref_lines[inst_id]],
                    gen_trg_sent)
            except:
                bleu = 0.0

            report = list()
            report.append('gen:\t' + gen_trg_sent)
            report.append('srcs:\t' + src_lines[inst_id])
            for rid in range(FLAGS.num_ref):
                report.append('ref%s:\t' % rid + ref_lines[inst_id][rid])
            report.append('sari:\t' + str(sari))
            report.append('bleu:\t' + str(bleu))
            report.append('score:\t' + str(gen_trg_score))
            report.append('==============================')
            report.append('')
            reports.append('\n'.join(report))

        all_reports = '\n'.join(reports)
        gen_trg_score = np.mean(gen_trg_scores)

        bleu_joshua = mteval_bleu.get_bleu_from_joshua(
            global_step, None, FLAGS.infer_ref_file, all_generated_sents, result_dir, FLAGS.num_ref
        )

        sari_joshua = sari_utils.get_sari_from_joshua(
            global_step, None, FLAGS.infer_ref_file, FLAGS.infer_src_file, all_generated_sents, result_dir,
            FLAGS.num_ref
        )

        format = "%.4f"
        bleu_joshua_format = format % bleu_joshua
        sari_joshua_format = format % sari_joshua
        gen_trg_score_format = format % gen_trg_score

        filename = 'step%s-score%s-sari%s-bleu%s.txt' % (
            global_step, gen_trg_score_format, sari_joshua_format, bleu_joshua_format)
        open(os.path.join(result_dir, filename), 'w').write(all_reports)

        if gen_trg_score < best_score:
            for fl in glob.glob(ckpt + '*'):
                os.remove(fl)
                tf.logging.info('remove ckpt file:%s' % fl)
        else:
            for file in os.listdir(model_dir):
                step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                if step not in file:
                    os.remove(os.path.join(model_dir,file))
            tf.logging.info('Get Best Model, remove ckpt except:%s.' % ckpt)
            best_score = sari_joshua


def main(_):
    log_dir = os.path.join(FLAGS.exp_dir, FLAGS.name, 'log')
    model_dir = os.path.join(FLAGS.exp_dir, FLAGS.name, 'model')
    result_dir = os.path.join(FLAGS.exp_dir, FLAGS.name, 'result')
    tf.gfile.MakeDirs(log_dir)
    tf.gfile.MakeDirs(model_dir)
    tf.gfile.MakeDirs(result_dir)
    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS.model_mode = FLAGS.model_mode.split(':')

    data = Data(FLAGS)

    run_config = tf.contrib.tpu.RunConfig(
            model_dir=log_dir,
            save_checkpoints_steps=1000)

    model_fn = model_fn_builder(
        data=data,
        init_ckpt_path=FLAGS.init_ckpt_path)

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size,
        use_tpu=False,
    )

    tf.logging.info('Current FLAGS:')
    tf.logging.info(tf.app.flags.FLAGS.flag_values_dict())

    if FLAGS.mode == 'train':
        train(data, estimator)
    elif FLAGS.mode == 'infer':
        repeat_infer(
            data, estimator, log_dir, model_dir, result_dir)


if __name__ == '__main__':
    tf.app.run()
