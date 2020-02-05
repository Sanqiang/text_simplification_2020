import re
import os
import math
import spacy
import nltk
import textstat
import operator
from collections import defaultdict
from nltk.corpus import stopwords


stopwords_set = set(["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats", "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"])
stemmer = nltk.PorterStemmer()
removed_wds = set(['a', 'an', 'the', ' '])


def _wd_valid(ori_wd):
    for ch in ori_wd:
        if str.isnumeric(ch):
            return False

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
            freq = float(items[1])
            ori_wd = pair[0]
            tar_wd = pair[1]
            if freq >= 0:
                mapper[ori_wd].append((tar_wd, freq))

        # Sort each rule
        for ori_wd in mapper:
            mapper[ori_wd] = sorted(mapper[ori_wd], key=operator.itemgetter(1))

        return mapper

    def _get_mapper(self):
        mapper = defaultdict(set)
        if not os.path.exists(self.flags.ppdb_file):
            print("Cannot find ppdb_file so train mapper is diabled.")
            return
        for line in open(self.flags.ppdb_file):
            items = line.split('\t')
            if items:
                weight = float(items[2])
                ori_wd = ' '.join([wd for wd in items[0].split() if wd not in removed_wds])
                tar_wd = ' '.join([wd for wd in items[1].split() if wd not in removed_wds])
                ori_wd_stem = ' '.join([stemmer.stem(wd) for wd in ori_wd.split()])
                tar_wd_stem = ' '.join([stemmer.stem(wd) for wd in tar_wd.split()])

                if (tar_wd in stopwords_set or ori_wd in stopwords_set
                        or ',' in ori_wd or '.' in ori_wd
                        or '\'' in ori_wd or '"' in ori_wd
                        or ',' in tar_wd or '.' in tar_wd
                        or '\'' in tar_wd or '"' in tar_wd
                        or '(' in ori_wd or ')' in ori_wd
                        or '(' in tar_wd or ')' in tar_wd or
                        tar_wd_stem in ori_wd_stem or ori_wd_stem in tar_wd_stem or
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

    def syntax_length_score(self, comp, simp):
        if "val" in self.flags.control_mode:
            # return textstat.flesch_kincaid_grade(simp)
            return len(simp)
        else:
            # return float(textstat.flesch_kincaid_grade(comp) / textstat.flesch_kincaid_grade(simp))
            return float(len(comp) / len(simp))

    def syntax_rel_score(self, comp, simp):
        wd_comp = set(comp)
        wd_simp = set(simp)
        return float(len(wd_comp & wd_simp) / len(wd_comp | wd_simp))

    def sent_length_score(self, comp, simp):
        if "val" in self.flags.control_mode:
            # return textstat.flesch_kincaid_grade(simp)
            return len(simp)
        else:
            # return float(textstat.flesch_kincaid_grade(comp) / textstat.flesch_kincaid_grade(simp))
            return float(len(comp) / len(simp))

    def word_length_score(self, comp, simp):
        if "val" in self.flags.control_mode:
            return float(len(simp) / len(simp.split()))
        else:
            return float(len(simp) / len(simp.split())) / float(len(comp) / len(comp.split()))

    def rel_score(self, comp, simp):
        wd_comp = set(comp)
        wd_simp = set(simp)
        return float(len(wd_comp & wd_simp) / len(wd_comp | wd_simp))

    # def _get_depth_spacytree_old(self, sent):
    #     doc = self.nlp(sent)
    #
    #     template = []
    #     for token in doc:
    #         if token.head.dep_ == "ROOT":
    #             template.append(token.dep_)
    #     template = " ".join(template)
    #
    #     level = 1
    #     for token in doc:
    #         if token.dep_ == "ROOT":
    #             q = [token]
    #             visited = {token}
    #             while q:
    #                 for _ in range(len(q)):
    #                     node = q.pop()
    #                     for sub in node.subtree:
    #                         if sub not in visited:
    #                             q.append(sub)
    #                             visited.add(sub)
    #                 level += 1
    #     return level, template

    def _get_depth_spacytree(self, sent):
        doc = self.nlp(sent)

        template = []
        for token in doc:
            if token.head.dep_ == "ROOT":
                template.append(token.dep_)
        template = " ".join(template)

        level = 1
        for token in doc:
            if token.dep_ == "ROOT":
                q = [token]
                visited = {token}
                while q:
                    for _ in range(len(q)):
                        node = q.pop()
                        for sub in node.subtree:
                            if sub not in visited:
                                q.append(sub)
                                visited.add(sub)
                    level += 1
        return level, template

    def _get_spacytree_templatefull(self, sent):
        def rec(node, cur_output, outputs):
            if len(list(node.children)) == 0:
                return

            for child in node.children:
                cur_output.append(child.dep_)
                rec(child, cur_output, outputs)
                outputs[child.i] = (child.text, list(cur_output))
                del cur_output[-1]

        doc = self.nlp(sent)
        outputs = ['#UNASSIGNED#' for _ in range(len(doc))]
        for token in doc:
            if token.head.dep_ == "ROOT":
                outputs[token.i] = (token.text, ['root'])
                rec(token, ['root'], outputs)
        assert all([tag != '#UNASSIGNED#' for tag in outputs])

        # Merge different tokenizers
        sent_split = sent.split()
        if len(outputs) != len(sent_split):
            foutputs = []
            i, j = 0, 0
            while i < len(outputs) and j < len(sent_split):
                if outputs[i][0] == sent_split[j]:
                    foutputs.append(outputs[i])
                    i += 1
                    j += 1
                elif sent_split[j].startswith(outputs[i][0]):
                    shortest_syntax = ['#UNASSIGNED#'] * 100
                    tmp = sent_split[j]
                    while i < len(outputs) and tmp.startswith(outputs[i][0]):
                        if len(outputs[i][1]) < len(shortest_syntax):
                            shortest_syntax = outputs[i][1]
                        tmp = tmp[len(outputs[i][0]):]
                        i += 1
                    assert shortest_syntax[0] != '#UNASSIGNED#'
                    foutputs.append((sent_split[j], shortest_syntax))
                    j += 1
                else:
                    raise ValueError("Unexpected Token: %s:%s" % (outputs, sent))
            outputs = foutputs

        template_full = []
        for output in outputs:
            template_full.append('|'.join(output[1]))

        assert len(template_full) == len(sent_split)

        return ' '.join(template_full)

    def syntax_score(self, comp, simp):
        sign_src, template_comp = self._get_depth_spacytree(comp)
        sign_dst, template_simp = self._get_depth_spacytree(simp)
        template_comp_full = self._get_spacytree_templatefull(comp)
        template_simp_full = self._get_spacytree_templatefull(simp)

        if "val" in self.flags.control_mode:
            return sign_dst, template_comp, template_simp, template_comp_full, template_simp_full
        else:
            return float(sign_dst) / float(sign_src), template_comp, template_simp, template_comp_full, template_simp_full

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
        return ' | '.join(tmp), rule

    def ppdb_score(self, comp, simp):
        rule_tars, rules = self.get_rules(comp, simp)
        score = float(len(rule_tars) / len(simp))
        return score, rule_tars, rules

    def get_rules_eval(self, comp):
        ori_wds = comp.split()
        tar_wds = []
        counter = defaultdict(int)
        for i in range(len(ori_wds)):
            unigram = ori_wds[i]
            if unigram in self.eval_mapper:
                for tar in sorted(self.eval_mapper[unigram],
                                  key=lambda wd: float(1000 * wd[1] + len(wd[0])), reverse=True):
                    counter[unigram] += 1
                    weight = tar[1] * (1 / counter[unigram]) ** 0.3
                    # tar_wds.append(('%s=>%s=>%s' % (unigram, tar[0], weight), weight))
                    tar_wds.append(('%s' % (tar[0]), weight))

            if i + 1 < len(ori_wds):
                bigram = ori_wds[i] + ' ' + ori_wds[1 + i]
                if bigram in sorted(self.eval_mapper[bigram],
                                    key=lambda wd: float(1000 * wd[1] + len(wd[0])), reverse=True):
                    for tar in self.eval_mapper[bigram]:
                        counter[bigram] += 1
                        weight = tar[1] * (1 / counter[bigram]) ** 0.3
                        # tar_wds.append(('%s=>%s=>%s' % (bigram, tar[0], weight), weight))
                        tar_wds.append(('%s' % (tar[0]), weight))

            if i + 2 < len(ori_wds):
                trigram = ori_wds[i] + ' ' + ori_wds[1 + i] + ' ' + ori_wds[2 + i]
                if trigram in sorted(self.eval_mapper[trigram],
                                     key=lambda wd: float(1000 * wd[1] + len(wd[0])), reverse=True):
                    for tar in self.eval_mapper[trigram]:
                        counter[trigram] += 1
                        weight = tar[1] * (1 / counter[trigram]) ** 0.3
                        # tar_wds.append(('%s=>%s=>%s' % (trigram, tar[0], weight), weight))
                        tar_wds.append(('%s' % (tar[0]), weight))

        tar_wds = list(set([wd for wd in tar_wds if wd[0] not in comp]))
        tar_wds.sort(key=lambda wd: float(1000 * wd[1] + len(wd[0])), reverse=True)
        output = ''
        for tar_wd in tar_wds:
            if tar_wd[0] not in output:
                output += " | " + tar_wd[0]
        return output
        # return '|'.join([w[0] for w in tar_wds])
        # return tar_wds

    def get_control_vec_eval(self,
                             comp,
                             rel=1.0,
                             sent_length=1.0,
                             word_length=1.0,
                             syn_length=1.0,
                             syntax=0.0,
                             split=1.0,
                             ppdb=1.0):
        # Disable length syntax split when predict is enabled
        vec = {}
        extra_outputs = {}

        if "val" in self.flags.control_mode:
            if "rel" in self.flags.control_mode:
                vec["rel"] = rel * self.rel_score(comp, comp)
            if "sent_length" in self.flags.control_mode:
                vec["sent_length"] = sent_length * self.sent_length_score(comp, comp)
            if "word_length" in self.flags.control_mode:
                vec["word_length"] = word_length * self.word_length_score(comp, comp)
            if "syntax" in self.flags.control_mode:
                syn_score, template_comp, template_simp, template_comp_full, template_simp_full = self.syntax_score(comp, comp)
                vec["syntax"] = syntax * syn_score
                extra_outputs["template_simp"] = template_simp
                extra_outputs["template_comp"] = template_comp
                if "syn_length" in self.flags.control_mode:
                    vec["syn_length"] = syn_length * self.syntax_length_score(template_comp, template_comp)
                if "syn_rel" in self.flags.control_mode:
                    vec["syn_rel"] = syn_length * self.syntax_rel_score(template_comp, template_comp)
        else:
            if "rel" in self.flags.control_mode:
                vec["rel"] = rel
            if "sent_length" in self.flags.control_mode:
                vec["sent_length"] = sent_length
            if "word_length" in self.flags.control_mode:
                vec["word_length"] = word_length
            if "syntax" in self.flags.control_mode:
                # syn_score, template_comp, template_simp = self.syntax_score(comp, comp)
                vec["syntax"] = syntax

        if "split" in self.flags.control_mode:
            vec["split"] = split
        if "ppdb" in self.flags.control_mode:
            vec["ppdb"] = ppdb
            ppdb_tars = self.get_rules_eval(comp)
            extra_outputs["external_inputs"] = ppdb_tars
            extra_outputs["rules"] = ppdb_tars  # Never used

        return vec, extra_outputs

    def get_control_vec(self, comp, simp):
        """Used for generate examples."""
        vec = []
        external_inputs = []
        extra_outputs = {}
        dim_per_factor = 1
        if "rel" in self.flags.control_mode:
            vec.extend([self.rel_score(comp, simp)] * dim_per_factor)
        if "sent_length" in self.flags.control_mode:
            vec.extend([self.sent_length_score(comp, simp)] * dim_per_factor)
        if "word_length" in self.flags.control_mode:
            vec.extend([self.word_length_score(comp, simp)] * dim_per_factor)
        if "syntax" in self.flags.control_mode:
            syn_score, template_comp, template_simp, template_comp_full, template_simp_full = self.syntax_score(comp, simp)
            vec.extend([syn_score] * dim_per_factor)
            extra_outputs["template_simp"] = template_simp
            extra_outputs["template_comp"] = template_comp
            extra_outputs["template_simp_full"] = template_simp_full
            extra_outputs["template_comp_full"] = template_comp_full
            if "syn_length" in self.flags.control_mode:
                vec.extend([self.syntax_length_score(template_simp, template_comp)] * dim_per_factor)
            if "syn_rel" in self.flags.control_mode:
                vec.extend([self.syntax_rel_score(template_simp, template_comp)] * dim_per_factor)
        if "split" in self.flags.control_mode:
            vec.extend([self.split_score(comp, simp)] * dim_per_factor)
        if "ppdb" in self.flags.control_mode:
            val, ppdb_tars, rules = self.ppdb_score(comp, simp)
            vec.extend([val] * dim_per_factor)
            external_inputs.append(ppdb_tars)
            extra_outputs["external_inputs"] = external_inputs
            extra_outputs["rules"] = rules

        return vec, extra_outputs

