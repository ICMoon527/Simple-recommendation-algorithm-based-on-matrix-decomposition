import numpy as np
import pandas as pd
import jieba
from jieba import analyse as analyse
# (209user, 566post)


def user_no_post():
    post = pd.read_csv('data_clean/particle_post.csv')
    x = [i for i in range(1, 210)]  # represent all the users
    y = np.array(post['poster_id'].values)
    difference = np.setdiff1d(x, y)
    return difference


def find_new_user(input):
    input = np.abs(input)
    new_user_id = []
    for i in range(input.shape[0]):
        if input[i][input[i].argmax()] == 0:  # user who doesn't like anything
            new_user_id.append(i+1)  # we want user_id not index, so we do +1
    no_post = user_no_post()  # user who didn't post
    new_user_id = np.intersect1d(new_user_id, no_post)  # user who is totally new
    print('new users are: ', new_user_id)
    return new_user_id


def get_key_words_with_weight(input):  # usually use this
    key_word = []
    weight = []

    for x, w in jieba.analyse.extract_tags(input, withWeight=True):
        key_word.append(x)
        weight.append(str(w))

    print('output: {}\n----------------------------------------------'.format(key_word))
    return ','.join(key_word), ','.join(weight)


def get_key_words_list(input):
    print('input: {}'.format(input))
    key_word = []
    x = jieba.analyse.extract_tags(input, withWeight=False)
    key_word = ','.join(x)
    print('output: {}\n----------------------------------------------'.format(key_word))
    return key_word


def load_embedding(path):
    embedding_index = {}
    f = open(path,encoding='utf8')
    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    return embedding_index


def topn_post(user):
    # assume n equals to 5
    doc = pd.read_csv('data_clean/test.csv')
    if user > doc.shape[0]:
        print("userID bigger than {}".format(doc.shape[0]))
        raise IndexError
    post_id = []
    for i in range(5):
        post_id.append(doc.loc[user - 1, 'top' + str(i + 1)].astype(int))
    print(post_id)
    return post_id


def word2vec_mean(words, weights):
    word2vec = pd.read_csv('data_clean/my_dictionary.csv')

    i = 0
    while i < len(words):
        if not words[i] in word2vec.keys():
            words.remove(words[i])
            weights.remove(weights[i])
            i -= 1
        i += 1

    if len(words) == 0:
        print('this vectors are illegal')
        return 0
    weights = np.asarray(weights, dtype=np.float)
    vectors = word2vec.loc[:, words] * weights
    vectors['mean'] = vectors.apply(lambda x: x.sum(), axis=1)  # calculate the mean of vectors
    vectors['mean'] = vectors['mean'] / len(words)
    return np.asarray(vectors['mean'])


def calculate_similarity(user_vector, post_vectors):
    similarity = []

    for i, item in enumerate(post_vectors):
        similarity.append(vector_similarity(user_vector, item))

    return similarity


def vector_similarity(x, y):
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    if not len(x) == len(y):
        print('dims of vectors are not equal!')
        raise MemoryError
    if y[0] == 0 and len(set(y)) == 1:
        # print('this post is empty')
        return 0
    if x[0] == 0 and len(set(x)) == 1:
        # print('this user is empty')
        return 0

    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i]**2  # sum(X*X)
        result3 += y[i]**2  # sum(Y*Y)

    result = result1 / ((result2 * result3)**0.5)
    return result


def get_stupid_rank_and_score():
    post = pd.read_csv('data_clean/particle_post.csv', usecols=['n_likes', 'n_comments'])
    post['sum'] = post.apply(lambda x: x.sum(), axis=1)
    temp = np.asarray(post['sum'].argsort())
    id = []
    score = []
    [id.append(temp[-i] + 1) for i in range(1, 6)]  # post_id
    [score.append(post.loc[temp[-i], 'sum']) for i in range(1, 6)]  # the score
    return id, score


def get_bias(like_or_not):
    u = like_or_not.mean()
    bias_user = [like_or_not[i].mean() for i in range(like_or_not.shape[0])] - u
    bias_post = [like_or_not[:, i].mean() for i in range(like_or_not.shape[1])] - u

    bias_user_broadcast = np.zeros((like_or_not.shape[0], like_or_not.shape[1]), dtype=np.float32)
    for i in range(like_or_not.shape[1]):
        bias_user_broadcast[:, i] = bias_user

    bias_post_broadcast = np.zeros((like_or_not.shape[0], like_or_not.shape[1]), dtype=np.float32)
    for i in range(like_or_not.shape[0]):
        bias_post_broadcast[i, :] = bias_post

    bias = u + bias_post_broadcast + bias_user_broadcast
    return bias, u, bias_post_broadcast, bias_user_broadcast


if __name__ == '__main__':
    user_like = pd.read_csv('data_clean/particle_like.csv')
    like_or_not = np.zeros((566, 209))
    # loop for filling the like_or_not up
    for index, row in user_like.iterrows():
        # if user cancel the 'like', describe it as 'dislike'
        like_or_not[row['post_id'] - 1, row['user_id'] - 1] = 1 if row['valid'] is 1 else -1
    bias, u, bias_post, bias_user = get_bias(like_or_not.transpose())
