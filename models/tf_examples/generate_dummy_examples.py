import tensorflow as tf

flags = tf.flags

flags.DEFINE_string(
    'comp_path',
    '/Users/sanqiang/git/ts/text_simplification_data/wikilarge/data-simplification/wikilarge/wiki.full.aner.ori.train.src',
    'The path for comp file.')
flags.DEFINE_string(
    'simp_path',
    '/Users/sanqiang/git/ts/text_simplification_data/wikilarge/data-simplification/wikilarge/wiki.full.aner.ori.train.dst',
    'The path for comp file.')
flags.DEFINE_string(
    'example_output_path',
    '/Users/Shared/Previously Relocated Items/Security/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/wiki.tfexample',
    'The path for ppdb outputs.')

FLAGS = flags.FLAGS


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


writer = tf.python_io.TFRecordWriter(FLAGS.example_output_path)

# max_l = 0
# from language_model.gpt2 import encoder
# import os
#
# models_dir = os.path.expanduser(
#     os.path.expandvars('/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/models'))
# enc = encoder.get_encoder('774M', models_dir)
for line_src, line_trg in zip(open(FLAGS.comp_path), open(FLAGS.simp_path)):

    # l = len(enc.encode(line))
    # max_l = max(l, max_l)

    feature = {
        'src_wds': _bytes_feature([str.encode(line_src.strip())]),
        'trg_wds': _bytes_feature([str.encode(line_trg.strip())])}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

print('Done')