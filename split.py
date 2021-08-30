import random


def split_set(file):
    lines_lst = file.readlines() # will return a list of reviews.
    train_set = []
    test_set = []
    l = [i for i in range(1000)]
    random.shuffle(l)
    for i in l[:700]:
        train_set.append(lines_lst[i])
    for i in l[700:]:
        test_set.append(lines_lst[i])
    return train_set,test_set