import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import utils

np.set_printoptions(threshold=1e6)
num_features = 20  # hyper-parameter when construct the model

# analyse the text information in the particle_userinfo.csv and particle_post.csv, the content of comment is absent
# because it both has sth. to do with user and post
user_information = pd.read_csv('data_clean/particle_userinfo.csv', usecols=['feature', 'des', 'major',
                                                                            'feature_weight', 'des_weight', 'major_weight'])
post_information = pd.read_csv('data_clean/particle_post.csv', usecols=['content', 'tags',
                                                                        'content_weight', 'tags_weight'])
user_post_matching_score = []
post_text_information_vector = []
content = []
tags = []
for i in range(post_information.shape[0]):
    print('dealing with post: {}'.format(i + 1))
    if not pd.isnull(post_information.loc[i, 'content']):
        content = utils.word2vec_mean(post_information.loc[i, 'content'].split(','),
                                      post_information.loc[i, 'content_weight'].split(','))
    else:
        content = np.zeros(200)
    if not pd.isnull(post_information.loc[i, 'tags']):
        tags = utils.word2vec_mean(post_information.loc[i, 'tags'].split(','),
                                   post_information.loc[i, 'tags_weight'].split(','))
    else:
        tags = np.zeros(200)

    post_text_information_vector.append(0.5 * content + 0.5 * tags)  # part of posts is done

for i in range(user_information.shape[0]):  # get similarity matrix between users and posts (209user, 566post)
    print('dealing with user: {}'.format(i + 1))
    feature = []
    des = []
    major = []
    if not pd.isnull(user_information.loc[i, 'feature']):
        feature = utils.word2vec_mean(user_information.loc[i, 'feature'].split(','),
                                      user_information.loc[i, 'feature_weight'].split(','))
    else:
        feature = np.zeros(200)
    if not pd.isnull(user_information.loc[i, 'des']):
        des = utils.word2vec_mean(user_information.loc[i, 'des'].split(','),
                                  user_information.loc[i, 'des_weight'].split(','))
    else:
        des = np.zeros(200)
    if not pd.isnull(user_information.loc[i, 'major']):
        major = utils.word2vec_mean(user_information.loc[i, 'major'].split(','),
                                    user_information.loc[i, 'major_weight'].split(','))
    else:
        major = np.zeros(200)

    user_text_information_vector = 0.7 * feature + 0.3 * des + 0.3 * major  # one user's part is done (200, 1)
    # calculate the similarity of user_text and post_text
    user_post_matching_score.append(utils.calculate_similarity(user_text_information_vector, post_text_information_vector))  # finally: (user_num, post_num)


# analyse the information in the particle_like.csv
user_like = pd.read_csv('data_clean/particle_like.csv')
like_num = user_like.shape[0]
# from preprocessing we know the N of users and posts
user_num = user_information.shape[0]
post_num = post_information.shape[0]
like_or_not = np.zeros((post_num, user_num))
# loop for filling the like_or_not up
for index, row in user_like.iterrows():
    # if user cancel the 'like', describe it as 'dislike'
    like_or_not[row['post_id'] - 1, row['user_id'] - 1] = 1 if row['valid'] is 1 else -1
print('finished like-filling work')
# create record matrix to record whether one post is read by one user (or did something instead)
record = (like_or_not != 0)
record = record.astype(int)
record = record.transpose()  # (209user, 566post)

# analyse the information in the particle_comment.csv
user_comment = pd.read_csv('data_clean/particle_comment.csv')
comment_or_not = np.zeros((post_num, user_num))
for index, row in user_comment.iterrows():
    comment_or_not[row['post_id'] - 1, row['comment_user_id'] - 1] = 1
print('finished comment-filling work')

# get like_or_not and comment_or_not together into like_or_not
for i, item_i in enumerate(like_or_not):
    for j, item_j in enumerate(item_i):
        if item_j == 1 and comment_or_not[i][j] == 1:
            like_or_not[i][j] += 2
        elif comment_or_not[i][j] == 1 and item_j == 0:
            like_or_not[i][j] += 1

# construct the model and normalize the data
# normalize qualitative information
like_or_not = like_or_not.transpose()  # (209user, 566post)
scalar_like_or_not = preprocessing.StandardScaler().fit(like_or_not)
like_or_not = scalar_like_or_not.transform(like_or_not)
# normalize quantitative information and get them together
scalar_user_post_matching_score = preprocessing.StandardScaler().fit(user_post_matching_score)
like_or_not += scalar_user_post_matching_score.transform(user_post_matching_score)

# calculate the bias
bias, u, bias_post, bias_user = utils.get_bias(like_or_not)

# find new user who did nothing(like and comment)
new_user_id = utils.find_new_user(record)

P = tf.get_variable(name='P',
                    shape=[user_num, num_features],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.2))
Q = tf.get_variable(name='Q',
                    shape=[num_features, post_num],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.2))
loss = 1/2 * tf.reduce_sum(((tf.matmul(P, Q) + bias - like_or_not) * record)**2) + \
       1/2 * (tf.reduce_sum(P**2) + tf.reduce_sum(Q**2) + tf.reduce_sum(bias_user**2) + tf.reduce_sum(bias_post**2))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# begin to train
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for e in range(20000):
    sess.run(train_op)
    if (e + 1) % 100 is 0:
        loss_train = sess.run(loss)
        print("step: {}\tloss_sum: {}\tloss_mean: {}".format(e, loss_train, loss_train / (user_num * post_num)))

# predict
P_last, Q_last = sess.run([P, Q])
preditct = np.dot(P_last, Q_last) + bias
preditct = preditct + (record * (- preditct.max()))  # decline the score if one has read the post

# write test.csv
result = pd.read_csv('data_raw/test.csv')
for i in range(preditct.shape[0]):
    temp = preditct[i].argsort()  # sorted score for the i+1 user
    result.loc[i, 'top1'] = (temp[-1] + 1)
    result.loc[i, 'top2'] = (temp[-2] + 1)
    result.loc[i, 'top3'] = (temp[-3] + 1)
    result.loc[i, 'top4'] = (temp[-4] + 1)
    result.loc[i, 'top5'] = (temp[-5] + 1)
    result.loc[i, 'score1'] = preditct[i][temp[-1]]
    result.loc[i, 'score2'] = preditct[i][temp[-2]]
    result.loc[i, 'score3'] = preditct[i][temp[-3]]
    result.loc[i, 'score4'] = preditct[i][temp[-4]]
    result.loc[i, 'score5'] = preditct[i][temp[-5]]

# begin to deal with the special situation of new users
new_user_id = new_user_id

for id in new_user_id:
    i = id - 1
    temp = np.argsort(user_post_matching_score[i])  # recommendation of the i+1 user who is new
    result.loc[i, 'top1'] = (temp[-1] + 1)
    result.loc[i, 'top2'] = (temp[-2] + 1)
    result.loc[i, 'top3'] = (temp[-3] + 1)
    result.loc[i, 'top4'] = (temp[-4] + 1)
    result.loc[i, 'top5'] = (temp[-5] + 1)
    result.loc[i, 'score1'] = user_post_matching_score[i][temp[-1]]
    result.loc[i, 'score2'] = user_post_matching_score[i][temp[-2]]
    result.loc[i, 'score3'] = user_post_matching_score[i][temp[-3]]
    result.loc[i, 'score4'] = user_post_matching_score[i][temp[-4]]
    result.loc[i, 'score5'] = user_post_matching_score[i][temp[-5]]

stupid_id, stupid_score = utils.get_stupid_rank_and_score()  # for users who exactly did nothing

for i in range(result.shape[0]):
    if result.loc[i, 'score1'] == 0:
        for j, item in enumerate(stupid_id):
            result.loc[i, 'top' + str(j + 1)] = item
            result.loc[i, 'score' + str(j + 1)] = stupid_score[j]

result[['top1', 'top2', 'top3', 'top4', 'top5']] = result[['top1', 'top2', 'top3', 'top4', 'top5']].astype(int)
result.to_csv('data_clean/test.csv', index=False)