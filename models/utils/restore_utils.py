"""
Utils for ckpt
"""
import collections
import re
import tensorflow as tf

def get_bert_assignment_map_from_checkpont(tvars, bert_init_checkpoint):
  initialized_variable_names = {
    'bert/embeddings/word_embeddings': 'word_embedding_table'}
  initialized_variable_names['word_embedding_table'] = 1
  initialized_variable_names['word_embedding_table:0'] = 1

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(bert_init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name_ckpt, var) = (x[0], x[1])
    for prefix in ['src_encoder/', 'trg_decoder/']:
      name_model = prefix + name_ckpt
      if name_model not in name_to_variable:
        continue
      assignment_map[name_ckpt] = name_model
      initialized_variable_names[name_model] = 1
      initialized_variable_names[name_model + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def get_gpt2_assignment_map_from_checkpoint(tvars, gpt2_init_checkpoint):
  assignment_map = {}
  assignment_map['model/wte'] = 'word_embedding_table'

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  ckpt_init_vars = tf.train.list_variables(gpt2_init_checkpoint)
  for x in ckpt_init_vars:
    (name_ckpt, var) = (x[0], x[1])
    if name_ckpt == 'model/wpe' or name_ckpt == 'model/wte':
      continue

    # For Encoder
    name_model = 'src_encoder/' + name_ckpt
    assignment_map[name_ckpt] = name_model
    # del name_to_variable[name_model]
    # For Decoder
    if (name_ckpt.startswith('model/ln_f/') or
            'mlp' in name_ckpt or
            'ln_2' in name_ckpt):
      name_model = 'trg_decoder/' + name_ckpt
      assignment_map[name_ckpt] = name_model
      # del name_to_variable[name_model]
    elif 'attn' in name_ckpt or 'ln_1' in name_ckpt:
      if 'attn' in name_ckpt:
        idx = name_ckpt.find('attn')
      elif 'ln_1' in name_ckpt:
        idx = name_ckpt.find('ln_1')
      else:
        idx = -1
      name_model = 'trg_decoder/' + name_ckpt[:idx] + 'decoder_selfattn/' + name_ckpt[idx:]
      assignment_map[name_ckpt] = name_model
      # del name_to_variable[name_model]
      # name_model = 'trg_decoder/' + name_ckpt[:idx] + 'decoder_encattn/' + name_ckpt[idx:]
      # assignment_map[name_ckpt] = name_model
      # del name_to_variable[name_model]

  return assignment_map


def check_shape_match(shape_ckpt, shape_model):
  if len(shape_ckpt) != len(shape_model):
    return False
  if len(shape_ckpt) == 1:
    return shape_ckpt[0] == shape_model[0].value
  elif len(shape_ckpt) == 2:
    return shape_ckpt[0] == shape_model[0].value and shape_ckpt[1] == shape_model[1].value
  else:
    return False


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, shape_ckpt) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    if not check_shape_match(shape_ckpt, name_to_variable[name].get_shape()):
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)