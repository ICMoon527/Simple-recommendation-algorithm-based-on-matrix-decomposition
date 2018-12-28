import pandas as pd
import numpy as np
import tensorflow as tf
import utils

np.set_printoptions(threshold=1e6)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def DF_show(input):
    print(input.head(), '\n', input.tail())
    print('-' * 50)
    print(input.describe())
    print('-' * 50)
    print(input.isnull().describe())
    print('-' * 50)
    print(input.dtypes)


def generate_particle_like():
    print('reading particle_like.csv...')
    name = ['like_id', 'post_id', 'user_id', 'time', 'valid']
    like = pd.read_csv('data_raw/particle_like.csv', header=0, names=name)
    DF_show(like)
    like.to_csv('data_clean/particle_like.csv', index=False)
    return like


def generate_particle_userinfo():  # 209 userID
    print('reading particle_userinfo.csv...')
    userinfo = pd.read_csv('data_raw/particle_userinfo.csv', header=0)
    # extract key words in the description
    descriptions = userinfo['des']

    for i, description in enumerate(descriptions):  # deal with content
        if not pd.isnull(description):
            key_words, weight = utils.get_key_words_with_weight(description)
            userinfo.loc[i, 'des'] = key_words
            userinfo.loc[i, 'des_weight'] = weight

    features = userinfo['feature']

    for i, feature in enumerate(features):  # deal with content
        if not pd.isnull(feature):
            key_words, weight = utils.get_key_words_with_weight(feature)
            userinfo.loc[i, 'feature'] = key_words
            userinfo.loc[i, 'feature_weight'] = weight

    major = userinfo['major']

    for i, item in enumerate(major):  # deal with content
        if not pd.isnull(item):
            key_words, weight = utils.get_key_words_with_weight(item)
            userinfo.loc[i, 'major'] = key_words
            userinfo.loc[i, 'major_weight'] = weight

    DF_show(userinfo)
    userinfo.to_csv('data_clean/particle_userinfo.csv', index=False)
    return userinfo


def generate_particle_post():  # 566 postID
    print('reading particle_post.csv...')
    name = ['post_id', 'poster_id', 'content', 'n_likes', 'n_comments', 'tags']
    post = pd.read_csv('data_raw/particle_post.csv', header=0, names=name)
    # extract key words in the content and tags
    contents = post['content']

    for i, content in enumerate(contents):  # deal with content
        if not pd.isnull(content):
            key_words, weight = utils.get_key_words_with_weight(content)
            post.loc[i, 'content'] = key_words
            post.loc[i, 'content_weight'] = weight

    tags = post['tags']

    for i, tag in enumerate(tags):
        if not pd.isnull(tag):
            key_words, weight = utils.get_key_words_with_weight(tag)
            post.loc[i, 'tags'] = key_words
            post.loc[i, 'tags_weight'] = weight

    DF_show(post)
    post.to_csv('data_clean/particle_post.csv', index=False)
    return post


def generate_particle_comment():
    print('reading particle_comment.csv...')
    comment = pd.read_csv('data_raw/particle_comment.csv')
    # extract key words in the content
    contents = comment['content']

    for i, content in enumerate(contents):  # deal with content
        if not pd.isnull(content):
            key_words, weight = utils.get_key_words_with_weight(content)
            comment.loc[i, 'content'] = key_words
            comment.loc[i, 'content_weight'] = weight

    DF_show(comment)
    comment.to_csv('data_clean/particle_comment.csv', index=False)
    # todo: if the length is not enough(less than 5 for example), just cut the comment off without extracting key words
    return comment


def generate_dictionary():  # path = "/datanew/DATASET/word2vec/Tencent_AILab_ChineseEmbedding.txt"
    # gathering all the key words
    userinfo_des = pd.read_csv('data_clean/particle_userinfo.csv')['des']
    post_content = pd.read_csv('data_clean/particle_post.csv')['content']
    post_tags = pd.read_csv('data_clean/particle_post.csv')['tags']
    comment_content = pd.read_csv('data_clean/particle_comment.csv')['content']
    userinfo_feature = pd.read_csv('data_clean/particle_userinfo.csv')['feature']
    userinfo_major = pd.read_csv('data_clean/particle_userinfo.csv')['major']

    word_set = set()
    userinfo_des = pd.concat([userinfo_des, post_content, post_tags, comment_content, userinfo_feature, userinfo_major])

    for item in userinfo_des:
        if not pd.isnull(item):
            words = item.split(',')
            [word_set.add(word) for word in words]

    print('loading dictionary...')
    dictionary = utils.load_embedding('/datanew/DATASET/word2vec/Tencent_AILab_ChineseEmbedding.txt')
    print('loading dictionary done')
    new_dic = {}
    exist_count = 0

    for word in word_set:
        if word in dictionary.keys():
            new_dic[word] = dictionary[word]
            exist_count += 1

    print("In total: {} the number that words not exist: {}".format(len(word_set), len(word_set) - exist_count))
    # In total: 4378 the number that words not exist: 157
    print('-' * 50)
    new_dic_df = pd.DataFrame(new_dic)
    new_dic_df.to_csv('data_clean/my_dictionary.csv')


def get_test():
    print('reading test.csv...')
    test = pd.read_csv('data_clean/test.csv')
    test[['top1', 'top2']] = test[['top1', 'top2']].astype(int)
    DF_show(test)
    return test


if __name__ == '__main__':
    generate_particle_post()