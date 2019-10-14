
from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate

def Dataprocess(filename,vocab = False):
    # print("data process...")

    wordlists = []
    taglists = []
    symbol = [' ','\n','']
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            lists = line.replace(' ', '').split("{{")
            wordlist = []
            taglist = []
            if lists[0] != '':
                l = lists[0].strip()
                for word in l:
                    taglist.append('o')
                    wordlist.append(word)
            for i in range(1,len(lists)):
                s = lists[i].split("}}")
                if len(s) < 2:
                    continue
                tag, words = s[0].split(':',1)
                for j in range(len(words)):
                    if j == 0:
                        taglist.append('B_' + tag)
                    elif j == len(words) - 1:
                        taglist.append('E_' + tag)
                    else:
                        taglist.append('M_' + tag)
                    wordlist.append(words[j])
                for k in s[1]:
                    if k not in symbol:
                        taglist.append('o')
                        wordlist.append(k)
            wordlists.append(wordlist)
            taglists.append(taglist)
    if vocab:
        word2id = Token2id(wordlists)
        tag2id = Token2id(taglists)
        return wordlists, taglists,word2id,tag2id
    else:
        return wordlists,taglists

def Token2id(lists):
    dic = {}
    for list in lists:
        for token in list:
            if token not in dic:
                dic[token] = len(dic)
    return dic

def Makecorpus(wordlists,taglists,pencent=0.7):
    assert len(wordlists) == len(taglists)

    length = len(wordlists)
    trainlength = int(length*pencent)
    train_wordlist,train_taglists = wordlists[0:trainlength],taglists[0:trainlength]
    test_wordlists,test_taglists = wordlists[trainlength:length],taglists[trainlength:length]

    return train_wordlist,train_taglists,test_wordlists,test_taglists
def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    # train_word_lists, train_tag_lists, word2id, tag2id = \
    #     build_corpus("train")
    # dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    # test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    file = 'BosonNLP_NER_6C.txt'
    wordlists, taglists, word2id, tag2id = Dataprocess(file, vocab=True)
    train_word_lists, train_tag_lists, test_word_lists, test_tag_lists = Makecorpus(wordlists, taglists, pencent=0.8)

    # # 训练评估ｈｍｍ模型
    # print("正在训练评估HMM模型...")
    # hmm_pred = hmm_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists),
    #     word2id,
    #     tag2id
    # )
    #
    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )

    testlen = len(test_word_lists)
    half_testlen = int(testlen/2)
    dev_word_lists, dev_tag_lists = test_word_lists[0:half_testlen], test_word_lists[0:half_testlen]
    train_word_lists, train_tag_lists = test_word_lists[half_testlen:-1], test_word_lists[half_testlen:-1]

    # 训练评估BI-LSTM模型
    print("正在训练评估双向LSTM模型...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        crf=False
    )

    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id
    )

    # ensemble_evaluate(
    #     [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
    #     test_tag_lists
    # )


if __name__ == "__main__":
    main()
