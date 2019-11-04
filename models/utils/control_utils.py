import re
import os
import spacy
import operator
from collections import defaultdict
from nltk.corpus import stopwords


stopwords_set = set(stopwords.words('english'))

def _wd_valid(ori_wd):
    for wd in ori_wd.split():
        if wd not in stopwords_set:
            return True
    return False


class ControlMethod:

    def __init__(self, flags):
        self.flags = flags

        self.spliter = re.compile('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        self.nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])
        self.mapper = self._get_mapper()
        self.eval_mapper = self._get_eval_mapper()

    def _get_eval_mapper(self):
        if not os.path.exists(self.flags.ppdb_vocab):
            print('not found %s . it may be ok if it is not training/eval but generate example'
                  % self.flags.ppdb_vocab)
            return
        mapper = defaultdict(list)
        for line in open(self.flags.ppdb_vocab):
            items = line.split('\t')
            pair = items[0].split('=>')
            freq = int(items[1])
            ori_wd = pair[0]
            tar_wd = pair[1]
            mapper[ori_wd].append((tar_wd, freq))

        # Sort each rule
        for ori_wd in mapper:
            mapper[ori_wd] = sorted(mapper[ori_wd], key=operator.itemgetter(1))

        return mapper

    def _get_mapper(self):
        mapper = defaultdict(set)
        for line in open(self.flags.ppdb_file):
            items = line.split('\t')
            if items:
                weight = float(items[2])
                ori_wd = items[0]
                tar_wd = items[1]

                if (tar_wd in stopwords_set or ori_wd in stopwords_set
                        or ',' in ori_wd or '.' in ori_wd
                        or '\'' in ori_wd or '"' in ori_wd
                        or ',' in tar_wd or '.' in tar_wd
                        or '\'' in tar_wd or '"' in tar_wd or
                        not _wd_valid(ori_wd) or not _wd_valid(tar_wd)):
                    continue

                mapper[ori_wd].add((tar_wd, weight))
        return mapper

    def sequence_contain_(self, seq, targets):
        if len(targets) == 0:
            print('%s_%s' % (seq, targets))
            return False
        if len(targets) > len(seq):
            return False
        for s_i, s in enumerate(seq):
            t_i = 0
            s_loop = s_i
            if s == targets[t_i]:
                while t_i < len(targets) and s_loop < len(seq) and seq[s_loop] == targets[t_i]:
                    t_i += 1
                    s_loop += 1
                if t_i == len(targets):
                    return s_loop - 1
        return -1

    def get_best_targets_(self, oriwords, line_dst, line_src, context_window_size=3):
        ress = []
        for tar in self.mapper[oriwords]:
            tar_words, weight = tar
            pos_dst = self.sequence_contain_(line_dst, tar_words.split())
            pos_src = self.sequence_contain_(line_src, tar_words.split())
            if pos_dst > 0 and pos_src == -1:
                # Check context
                pos_src = self.sequence_contain_(line_src, oriwords.split())

                left_win_src, left_win_dst = set(), set()
                w = 1
                while w <= (1 + context_window_size):
                    if pos_src - w >= 0:
                        try:
                            left_win_src.add(line_src[pos_src - w])
                        except:
                            pass
                        left_win_src.add(line_src[pos_src - w])
                    if pos_dst - w >= 0:
                        try:
                            left_win_dst.add(line_dst[pos_dst - w])
                        except:
                            pass
                        left_win_dst.add(line_dst[pos_dst - w])
                    w += 1

                right_win_src, right_win_dst = set(), set()
                w = 0
                while w <= context_window_size:
                    if pos_src + len(oriwords.split()) + w < len(line_src):
                        try:
                            right_win_src.add(line_src[pos_src + len(oriwords.split()) + w])
                        except:
                            pass
                        right_win_src.add(line_src[pos_src + len(oriwords.split()) + w])
                    if pos_dst + len(tar_words.split()) + w < len(line_dst):
                        try:
                            right_win_dst.add(line_dst[pos_dst + len(tar_words.split()) + w])
                        except:
                            pass
                        right_win_dst.add(line_dst[pos_dst + len(tar_words.split()) + w])
                    w += 1
                if len(left_win_src & left_win_dst) == 0 and len(right_win_src & right_win_dst) == 0:
                    continue

                res = ('%s=>%s=>%s' % (oriwords, tar_words, weight), weight)
                ress.append(res)
            else:
                continue
        if ress:
            ress.sort(key=operator.itemgetter(1), reverse=True)
            return ress[0]
        else:
            return None

    def length_score(self, comp, simp):
        return float(len(comp) / len(simp))

    def syntax_score(self, comp, simp):
        sign_src = set()
        for tok in self.nlp(comp):
            sign_src.add('%s-%s' % (tok.text, tok.dep_))
        sign_dst = set()
        for tok in self.nlp(simp):
            sign_dst.add('%s-%s' % (tok.text, tok.dep_))
        return len(sign_src & sign_dst) / len(sign_src | sign_dst)

    def split_score(self, comp, simp):
        return float(len(self.spliter.split(simp)) > len(self.spliter.split(comp)))

    def get_rules(self, line_src, line_dst):
        rule = set()
        line_src = line_src.split()
        line_dst = line_dst.split()
        # print('bleu:%s' % bleu)
        for wid in range(len(line_src)):
            # For unigram
            unigram = line_src[wid]
            if unigram in self.mapper and unigram not in line_dst:
                res = self.get_best_targets_(unigram, line_dst, line_src)
                if res:
                    rule.add(res[0])

            # For bigram
            if wid + 1 < len(line_src):
                bigram = line_src[wid] + ' ' + line_src[wid + 1]
                if bigram in self.mapper and self.sequence_contain_(line_dst, (line_src[wid], line_src[wid + 1])) == -1:
                    res = self.get_best_targets_(bigram, line_dst, line_src)
                    if res:
                        rule.add(res[0])

            # For trigram
            if wid + 2 < len(line_src):
                trigram = line_src[wid] + ' ' + line_src[wid + 1] + ' ' + line_src[wid + 2]
                if trigram in self.mapper and self.sequence_contain_(line_dst, (
                line_src[wid], line_src[wid + 1], line_src[wid + 2])) == -1:
                    res = self.get_best_targets_(trigram, line_dst, line_src)
                    if res:
                        rule.add(res[0])
        tmp = [(r.split('=>')[1], float(
            r.split('=>')[2]) + float(len(r.split('=>')[0].split()))) for r in rule if r]
        tmp.sort(key=operator.itemgetter(1), reverse=True)
        tmp = [t[0] for t in tmp]
        return ' '.join(tmp), rule

    def ppdb_score(self, comp, simp):
        rule_tars, rules = self.get_rules(comp, simp)
        score = float(len(rule_tars) / len(simp))
        return score, rule_tars, rules

    def get_rules_eval(self, comp):
        ori_wds = comp.split()
        tar_wds = []
        for i in range(len(ori_wds)):
            unigram = ori_wds[i]
            if unigram in self.eval_mapper:
                tar_wds.extend(self.eval_mapper[unigram])

            if i + 1 < len(ori_wds):
                bigram = ori_wds[0] + ' ' + ori_wds[1]
                if bigram in self.eval_mapper:
                    tar_wds.extend(self.eval_mapper[bigram])

            if i + 2 < len(ori_wds):
                trigram = ori_wds[0] + ' ' + ori_wds[1] + ' ' + ori_wds[2]
                if trigram in self.eval_mapper:
                    tar_wds.extend(self.eval_mapper[trigram])
        tar_wds = sorted(tar_wds, key=operator.itemgetter(1))
        return ' '.join([w[0] for w in tar_wds])

    def get_control_vec_eval(self,
                             comp,
                             length=0.5,
                             syntax=0.0,
                             split=1.0,
                             ppdb=1.0,
                             ):
        vec = []
        dim_per_factor = 1
        vec.extend([length] * dim_per_factor)
        vec.extend([syntax] * dim_per_factor)
        vec.extend([split] * dim_per_factor)
        val = [ppdb] * dim_per_factor
        ppdb_tars = self.get_rules_eval(comp)
        vec.extend(val)

        return vec, ppdb_tars

    def get_control_vec(self, comp, simp):
        vec = []
        external_inputs = []
        dim_per_factor = 1
        # if 'length' in self.flags.control_mode:
        vec.extend([self.length_score(comp, simp)] * dim_per_factor)
        # if 'syntax' in self.flags.control_mode:
        vec.extend([self.syntax_score(comp, simp)] * dim_per_factor)
        # if 'split' in self.flags.control_mode:
        vec.extend([self.split_score(comp, simp)] * dim_per_factor)
        # if 'ppdb' in self.flags.control_mode:
        val, ppdb_tars, rules = self.ppdb_score(comp, simp)
        vec.extend([val] * dim_per_factor)
        external_inputs.append(ppdb_tars)

        return vec, external_inputs, rules

