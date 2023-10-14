import numpy as np
import re
import shutil
import os
import random
from tqdm import tqdm

from fairseq.tokenizer import tokenize_line

list1 = "qwertyuiopasdfghjklzxcvbnmư"
list2 = "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./"
dict1 = {'à':'aw', 'á':'as', 'ả':'ar', 'ã':'ax', 'ạ':'aj', 'ă':'aw', 'ằ':'afw', 'ắ':'asw', 'ẳ':'arw', 'ẵ':'axw', 'ặ':'ajw', 'â':'aa', 'ầ':'aaf', 'ấ':'aas', 'ẩ':'aar', 'ẫ':'aax', 'ậ':'aaj', 'đ':'dd', 'è':'ef', 'é':'es', 'ẻ':'er', 'ẽ':'ex', 'ẹ':'ej', 'ê':'ee', 'ề':'eef', 'ế':'ees', 'ể':'eer', 'ễ':'eex', 'ệ':'eej', 'ì':'if', 'í':'is', 'ỉ':'ir', 'ĩ':'ix', 'ị':'ij', 'ò':'of', 'ó':'os', 'ỏ':'or', 'õ':'ox', 'ọ':'oj', 'ô':'oo', 'ồ':'oof', 'ố':'oos', 'ổ':'oor', 'ỗ':'oox', 'ộ':'ooj', 'ơ':'ow', 'ờ':'owf', 'ớ':'ows', 'ở':'owr', 'ỡ':'owx', 'ợ':'owj', 'ù':'uf', 'ú':'us', 'ủ':'ur', 'ũ':'ux', 'ụ':'uj', 'ư':'uw', 'ừ':'uwf', 'ứ':'uws', 'ử':'uwr', 'ữ':'uwx', 'ự':'uwj', 'ỳ':'yf', 'ý':'ys', 'ỷ':'yr', 'ỹ':'yx', 'ỵ':'yj'}

class NoiseInjector(object):

    def __init__(self, corpus,
                 error_rate):

        # READ-ONLY, do not modify
        self.corpus = corpus
        self.error_rate = error_rate

    #chuyển word về dạng telex của nó, nếu nó ko có từ nào chuyển được thì giữ nguyên
    def _replace_func(self, i, p):
        # replace here
        vn_letters = [char for char in p if char in dict1.keys()]
        if vn_letters:
            # Randomly select one of the Vietnamese letters
            modified_letter = random.choice(vn_letters)
            rnd_word = p.replace(modified_letter, dict1[modified_letter])
            # append -1 with the replaced word
            return (-1, rnd_word)
        else:
            return (i, p)
    
    # thay 1 char bằng 1 char khác
    def _replace_1_char_func(self, i, p):
        modified_word = list(p)
        word_length = len(modified_word)
        modified_index = random.randint(0, word_length - 1)
        modified_word[modified_index] = random.choice(list2)
        rnd_word = ''.join(modified_word)

        # replace here
        return (-1, rnd_word)           

    # xóa 1 char khỏi word
    def _delete_func(self, i, p):
        # delete here
        modified_word = list(p)
        word_length = len(modified_word)
        modified_index = random.randint(0, word_length - 1)
        new_word =  ''.join(modified_word[:modified_index] + modified_word[modified_index + 1:])
        return (-1, new_word)


    # thêm 1 char vào word
    def _add_func(self, i, p):
        # add 1  here
        modified_word = list(p)
        word_length = len(modified_word)
        modified_index = random.randint(0, word_length - 1)
        y = random.randrange(len(list1))
        rnd_word = ''.join(modified_word[:modified_index]) + list1[y] + ''.join(modified_word[modified_index:])
        # append -1 with the added word
        return (-1, rnd_word)

    
    # đổi chỗ vị trí 2 char
    def _swap_func(self, i, p):
        modified_word = list(p)
        word_length = len(modified_word)
        if word_length - 2 <= 0:
            return (i, p)

        modified_index = random.randint(0, word_length - 2)
        modified_word[modified_index], modified_word[modified_index + 1] = modified_word[modified_index + 1], modified_word[modified_index]
        rnd_word = ''.join(modified_word)

        # append -1 with the swaped word
        return (-1, rnd_word)

    def _parse(self, pairs):
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][0]
            w = pairs[si][1]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def inject_noise(self, tokens):
        self.rnd = np.random.random(len(tokens))

        # tgt is a vector of integers
        funcs = [self._add_func, self._replace_func, self._delete_func, self._replace_1_char_func, self._swap_func]
        np.random.shuffle(funcs)        
        pairs = [(i, w) for (i, w) in enumerate(tokens)]
        # for each token, if random of token < error rate so we made a error
        for i in range(len(pairs)):
            f = funcs[i % len(funcs)]
            if self.rnd[i] < self.error_rate:
                pairs[i] = f(pairs[i][0], pairs[i][1])

        return self._parse(pairs)

def save_file_by_append(filename, contents):
    with open(filename, 'a') as ofile:
        for content in contents:
            ofile.write(' '.join(content) + '\n')

# make noise from filename
def noise_and_save_file(lines, ofile_suffix, batch_size, error_rate = 0.2):
    tgts = [tokenize_line(line.strip()) for line in lines]            
    noise_injector = NoiseInjector(tgts, error_rate=error_rate)
    
    srcs = []
    aligns = []
    for tgt in tgts:
        src, align = noise_injector.inject_noise(tgt)
        srcs.append(src)
        aligns.append(align)
    
    # save batch in file
    save_file_by_append('{}.src'.format(ofile_suffix), srcs)
    save_file_by_append('{}.tgt'.format(ofile_suffix), tgts)
    save_file_by_append('{}.forward'.format(ofile_suffix), aligns)        

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-bs', '--batch-size', type=int, default=100)


args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
if __name__ == '__main__':
    print("epoch={}, seed={}, batch_size={}".format(args.epoch, args.seed, args.batch_size))

    path = '/home/lenovo/Documents/1. FPT University/4.Project/Data/news-corpus-categorys-20181217/corpus'
    save_path = '/home/lenovo/Documents/1. FPT University/4.Project/Data/news-corpus-categorys-20181217/noised_corpus'
    for file_name in os.listdir(path):
        if not file_name.endswith('.txt'):
            continue
        ofile_suffix = 'train_1b_{}'.format(os.path.basename(file_name))
        # make dir 
        os.mkdir(os.path.join(save_path, os.path.splitext(file_name)[0]))
        # create saved file
        ofile_suffix = os.path.join(save_path, os.path.splitext(file_name)[0], ofile_suffix)
        print("Loading file:", file_name)
        file_name = os.path.join(path, file_name)
        i = 0
        lines = []
        with open(file_name, encoding = 'utf-8') as txt_file:
            for line in tqdm(txt_file):
                lines.append(line)
                i = i + 1
                if i % args.batch_size == 0:
                    noise_and_save_file(lines, ofile_suffix, args.batch_size)
                    # break
                    lines = []

            # check if there are any sent in the last lines
            if len(lines) > 0:
                noise_and_save_file(lines, ofile_suffix, args.batch_size)

# Hom nay Andy di hoc.
# An# #dy ? 
# Andy -> Andy (UNK)
# moi cau 1 loai loi, 1 loi la 20%.
# Hien tai: 5 loai loi, moi loai loi la 10%
# 20% loi, check tung tu trong cau, neu co the co loi thi goi ham.