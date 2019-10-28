import os
import time
import shutil

import tensorflow as tf

ckpt_prefix = 'model.ckpt-'

def get_best_score(resultdir):
    """
    Get current best performance
    :param resultdir:
    :return:
    """
    best_sari = 0.0
    if os.path.exists(resultdir):
        results = os.listdir(resultdir)
        for result in results:
            if result.startswith('step') and result.endswith('.result'):
                sari = float(result[(result.index('score')+len('score')):result.rindex('-sari')])
                best_sari = max(sari, best_sari)
    return best_sari


def get_ckpt(modeldir, logdir, wait_second=10):
    """
    Get latest checkpoint from logdir and copy it to modeldir
    :param modeldir:
    :param logdir:
    :param wait_second:
    :return:
    """
    while True:
        try:
            ckpt = _copy_ckpt_to_modeldir(modeldir, logdir)
            return ckpt
        except FileNotFoundError as exp:
            if wait_second:
                tf.logging.info(str(exp) + '\nWait for 1 minutes.')
                time.sleep(wait_second)
            else:
                return None


def _find_train_ckptfiles(path, is_delete):
    """Find checkpoint files based on its max steps.
       is_outdir indicates whether find from outdir or modeldir.
       note that outdir generated from train and eval copy them to modeldir.
    """
    # if not exists(path):
    #     return None, -1
    steps = [int(f[len(ckpt_prefix):-5]) for f in os.listdir(path)
             if f[:len(ckpt_prefix)] == ckpt_prefix and f[-5:] == '.meta']
    if len(steps) == 0:
        if is_delete:
            raise FileNotFoundError('No Available ckpt.')
        else:
            return None, -1
    max_step = max(steps)
    if len(steps) > 5 and is_delete:
        del_model_files = _get_model_files(sorted(steps)[:-5], path)
        for del_model_file in del_model_files:
            os.remove(path + del_model_file)

    model_files = _get_model_files(max_step, path)
    return model_files, max_step


def _get_model_files(steps, path):
    """
    Get model files based on steps
    :param steps:
    :param path:
    :return:
    """
    if not isinstance(steps, list):
        steps = [steps]
    model_files = []
    for step in steps:
        model_pref = ckpt_prefix + str(step)
        model_files.extend([f for f in os.listdir(path)
                            if os.path.isfile(os.path.join(path, f)) and f[:len(model_pref)] == model_pref])
    return model_files


def _copy_ckpt_to_modeldir(modeldir, logdir):
    """
    Copy checkpoints from logdir to modeldir
    :param modeldir:
    :param logdir:
    :return:
    """
    files, max_step = _find_train_ckptfiles(logdir, False)
    _, cur_max_step = _find_train_ckptfiles(modeldir, False)
    if cur_max_step == max_step:
        raise FileNotFoundError('No new ckpt. cur_max_step: %s, max_step: %s.'
                                % (cur_max_step, max_step))

    for file in files:
        source = os.path.join(logdir, file)
        target = os.path.join(modeldir, file)
        shutil.copy2(source, target)
        print('Copy Ckpt from %s \t to \t %s.' % (source, target))
    return os.path.join(modeldir, ckpt_prefix + str(max_step))


def is_fresh_run(logdir):
    _, max_step = _find_train_ckptfiles(logdir, False)
    print('current max step %s is fresh:%s' % (str(max_step), str(max_step<=0)))
    return max_step <= 0


def find_latest_ckpt(logdir):
    _, max_step = _find_train_ckptfiles(logdir, False)
    return ''.join([logdir, 'model.ckpt-', str(max_step)])
