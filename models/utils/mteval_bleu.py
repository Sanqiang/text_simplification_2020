import copy as cp
import subprocess
import re
import os

joshua_script = '/zfs1/hdaqing/saz31/dataset/ts_script/ppdb-simplification-release-joshua5.0/joshua/bin/bleu'
joshua_class = '/zfs1/hdaqing/saz31/dataset/ts_script/ppdb-simplification-release-joshua5.0/joshua/class'


def result2txt(sents, lowercase=False, join_split=' '):
    nsents = []
    for sent in sents:
        if lowercase:
            sent = sent.lower()
        sent = sent.strip()
        nsents.append(sent)

    nsents = '\n'.join(nsents)
    return nsents


def get_bleu_from_joshua(step, path_dst, path_ref, targets, resultdir, num_ref):
    path_tar = resultdir + '/joshua_target_%s.txt' % step
    if not os.path.exists(path_tar):
        f = open(path_tar, 'w', encoding='utf-8')
        # joshua require lower case
        f.write(result2txt(targets, lowercase=True))
        f.close()

    if num_ref > 0:
        return get_result_joshua(path_ref, path_tar, num_ref)
    else:
        return get_result_joshua_nonref(path_dst, path_tar)


def get_result_joshua(path_ref, path_tar, num_ref):
    args = ' '.join([joshua_script, path_tar, path_ref,
                     str(num_ref), joshua_class])

    pipe = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    mteval_result = pipe.communicate()

    m = re.search(b'BLEU = ([\d+\.]+)', mteval_result[0])

    try:
        result = float(m.group(1))
    except AttributeError:
        result = 0
    return result

def get_result_joshua_nonref(path_ref, path_tar):
    args = ' '.join([joshua_script, path_tar, path_ref,
                     '1', joshua_class])

    pipe = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    mteval_result = pipe.communicate()

    m = re.search(b'BLEU = ([\d+\.]+)', mteval_result[0])

    try:
        result = float(m.group(1))
    except AttributeError:
        result = 0
    return result
