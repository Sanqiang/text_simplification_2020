import spacy

sent = "CrÃ py , Pas-de-Calais"

nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])
doc = nlp(sent)

def rec(node, cur_output, outputs):
    if len(list(node.children)) == 0:
        return

    for child in node.children:
        cur_output.append(child.dep_)
        rec(child, cur_output, outputs)
        outputs[child.i] = (child.text, list(cur_output))
        # print((child.text, list(cur_output)))
        del cur_output[-1]

outputs = ['xxx' for _ in range(len(doc))]
for token in doc:
    if token.dep_ == "ROOT":
        outputs[token.i] = (token.text, ['root'])
        rec(token, ['root'], outputs)

# print('\noutputs:\n')
# for output in outputs:
#     print(output)


sent = sent.split()
if len(outputs) != len(sent):
    foutputs = []
    i, j = 0, 0
    while i < len(outputs) and j < len(sent):
        if outputs[i][0] == sent[j]:
            foutputs.append(outputs[i])
            i += 1
            j += 1
        elif sent[j].startswith(outputs[i][0]):
            shortest_syntax = ['root'] * 100
            tmp = sent[j]
            while i < len(outputs) and tmp.startswith(outputs[i][0]):
                if len(outputs[i][1]) < len(shortest_syntax):
                    shortest_syntax = outputs[i][1]

                tmp = tmp[len(outputs[i][0]):]
                i += 1

            foutputs.append((sent[j], shortest_syntax))
            j += 1
        else:
            raise ValueError("Unexpected Token: %s:%s" % (outputs, sent))

    print('\nfoutputs:\n')
    for output in foutputs:
        print(output)

    # template_full = []
    # for output in foutputs:
    #     template_full.append('|'.join(output[1]))
    # print(template_full)