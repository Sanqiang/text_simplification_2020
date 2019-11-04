import os
import glob
import json
import spacy
import tensorflow as tf
from models.utils.control_utils import ControlMethod

flags = tf.flags

flags.DEFINE_string(
    "prefixs",
    "wikilarge,wikisplit",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "json_file",
    "/zfs1/hdaqing/saz31/dataset/tmp_wikisplit_8192/lm_score/,/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/lm_score/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    'example_output_path',
    '/zfs1/hdaqing/saz31/dataset/example_v1/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    'text_output_path',
    '/zfs1/hdaqing/saz31/dataset/text_v1/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    'rule_output_path',
    '/zfs1/hdaqing/saz31/dataset/rule_v1/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    "ppdb_file", "/zfs1/hdaqing/saz31/dataset/ppdb.txt",
    "The file path of ppdb")

flags.DEFINE_string(
    "ppdb_vocab", "/zfs1/hdaqing/saz31/dataset/rule_v1/vocab",
    "The file path of ppdb vocab generated from train")

FLAGS = flags.FLAGS


nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _split_first_token(sent):
    tokens = sent.split()
    token = tokens[0]
    sent = tokens[1:]
    assert token.startswith('[[[') and token.endswith(']]]')
    return ' '.join(sent), token


def _get_reorder_sim_score(sent1, sent2):
    sign_src = set()
    for tok in nlp(sent1):
        sign_src.add('%s-%s' % (tok.text, tok.dep_))
    sign_dst = set()
    for tok in nlp(sent2):
        sign_dst.add('%s-%s' % (tok.text, tok.dep_))
    return len(sign_src & sign_dst) / len(sign_src | sign_dst)


def _validate(sent1, sent2):
    s1 = set(sent1.split())
    s2 = set(sent2.split())
    return len(s1 & s2) / len(s1 | s2) >= 0.5


def process_line(line):
    comps, simps = [], []
    obj = json.loads(line)

    # Get comp sentence
    comp, comp_score = None, None
    for sent in obj:
        nsent, token = _split_first_token(sent)
        score = obj[sent]
        if token == '[[[COMP]]]':
            comp, comp_score = nsent.lower(), score
            break

    # Loop other sentences
    fluent_sent, fluent_score = None, 99999
    largest_reorder_sent, largest_reorder_score = None, 99999
    for sent in obj:
        nsent, token = _split_first_token(sent)
        nsent = nsent.lower()
        score = obj[sent]
        if token != '[[[COMP]]]':
            if _validate(comp, nsent):
                if score < fluent_score and score < comp_score:
                    fluent_sent, fluent_score = nsent, score

                reorder_score = _get_reorder_sim_score(comp, nsent)
                if reorder_score < largest_reorder_score and score < comp_score:
                    largest_reorder_score, largest_reorder_sent = reorder_score, nsent

    if fluent_sent is not None and fluent_sent != comp:
        comps.append(comp)
        simps.append(fluent_sent)
    if largest_reorder_sent is not None and fluent_sent != largest_reorder_sent and largest_reorder_sent != comp:
        comps.append(comp)
        simps.append(largest_reorder_sent)

    return comps, simps


def process(idx, json_file, prefix, control_obj):
    json_file = json_file + 'shard%s.txt' % idx
    if not os.path.exists(json_file):
        return

    os.makedirs(FLAGS.example_output_path, exist_ok=True)
    example_file = FLAGS.example_output_path + 'shard_%s_%s.example' % (prefix, idx)
    if os.path.exists(example_file):
        return
    os.makedirs(FLAGS.text_output_path, exist_ok=True)
    text_file = FLAGS.text_output_path + 'shard_%s_%s.txt' % (prefix, idx)

    os.makedirs(FLAGS.rule_output_path, exist_ok=True)
    rule_file = FLAGS.rule_output_path + 'shard_%s_%s.txt' % (prefix, idx)

    writer = tf.python_io.TFRecordWriter(example_file)
    comps, simps = [], []
    for line in open(json_file):
        tmp_comps, tmp_simps = process_line(line)
        comps.extend(tmp_comps)
        simps.extend(tmp_simps)
    texts, rules = [], []
    for comp, simp in zip(comps, simps):
        comp = comp.strip()
        simp = simp.strip()
        control_vec, control_inputs, rule = control_obj.get_control_vec(
            comp, simp)

        feature = {'src_wds': _bytes_feature([str.encode(comp)]),
                   'trg_wds': _bytes_feature([str.encode(simp)]),
                   'control_wds': _bytes_feature([str.encode(control_inputs[0])]),
                   'control_vec': _float_feature(control_vec)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        texts.append('%s\n%s\n%s\n%s\n\n\n' % (comp, simp, control_inputs[0], control_vec))
        rules.append('\t'.join(rule))
    writer.close()
    open(text_file, 'w').write('\n'.join(texts))
    open(rule_file, 'w').write('\n'.join(rules))


if __name__ == '__main__':
    json_files = FLAGS.json_file.split(',')
    prefixs = FLAGS.prefixs.split(',')
    assert len(prefixs) == len(json_files)

    control_obj = ControlMethod(FLAGS)

    for json_file, prefix in zip(json_files, prefixs):
        print('start process %s' % json_file)
        for i in range(9000):
            process(i, json_file, prefix, control_obj)


