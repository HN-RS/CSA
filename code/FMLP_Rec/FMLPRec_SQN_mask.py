import numpy as np
import pandas as pd
import os
import argparse

import tf_slim
import trfl
from utility import *
from Modules import *

import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised SASRec.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='../../data/RC15/',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument('--cuda', type=int, default=1,
                        help='cuda device.')
    parser.add_argument('--q_w', type=float, default=0.1,
                        help='weight of qloss.')

    parser.add_argument("--no_filters", action="store_true",
                        help="if no filters, filter layers transform to self-attention")
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')

    parser.add_argument('--cond_len', type=int, default=3,
                        help='length of masked state.')
    parser.add_argument('--num_aug', type=int, default=2,
                        help='Number of state augmentations.')
    return parser.parse_args()


class FMLPRecModel:
    def __init__(self, hidden_size, learning_rate, item_num, state_size, args, name='FMLPRecModel'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.name = name
        with tf.variable_scope(self.name):
            all_embeddings = self.initialize_embeddings()

            self.inputs = tf.placeholder(tf.int32, [None, state_size], name='inputs')
            self.len_state = tf.placeholder(tf.int32, [None], name='len_state')
            self.mask = tf.placeholder(tf.float32, [args.num_aug, None, state_size])
            # Embedding Layer
            self.input_emb = tf.nn.embedding_lookup(all_embeddings['state_embeddings'], self.inputs)
            # Positional Encoding
            self.pos_emb = tf.nn.embedding_lookup(all_embeddings['pos_embeddings'],
                                                  tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0),
                                                          [tf.shape(self.inputs)[0], 1]))
            self.sequence_emb = self.input_emb + self.pos_emb
            # Normalize & DropOut
            self.sequence_emb = normalize(self.sequence_emb)
            self.sequence_emb = tf.layers.dropout(self.sequence_emb, rate=args.hidden_dropout_prob)

            attention_mask = tf.cast((self.inputs < item_num), dtype=tf.float32)
            extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, dim=1), dim=2)
            max_len = attention_mask.shape[-1]  # 64
            attn_shape = (1, max_len, max_len)  # (1,64,64)
            subsequent_mask = tf.expand_dims(tf.linalg.band_part(tf.ones(attn_shape), -1, 0), dim=1)
            extended_attention_mask = extended_attention_mask * subsequent_mask
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            self.complex_weight = tf.Variable(tf.random_normal([1, self.state_size // 2 + 1, args.hidden_factor, 2],
                                                               0.0, 1.0), name='complex_weight')
            self.states_hidden_aug2 = [[]] * (args.num_aug + 1)
            self.sequence_emb_aug = self.sequence_emb

            for na in range(args.num_aug + 1):
                item_encoded_layers = item_encoder(args, self.sequence_emb_aug, extended_attention_mask, self.complex_weight,
                                                   output_all_encoded_layers=True, reuse=tf.AUTO_REUSE)
                sequence_output = item_encoded_layers[-1]

                self.state_hidden = extract_axis_1(sequence_output, self.len_state - 1)

                self.states_hidden_aug2[na] = self.state_hidden
                if na < args.num_aug:
                    self.sequence_emb_aug = tf.expand_dims(self.mask[na], axis=-1) * self.sequence_emb

            self.output1_ = tf_slim.fully_connected(self.states_hidden_aug2, self.item_num,
                                                    activation_fn=None, scope="q-value")  # all q-values
            self.output2 = tf_slim.fully_connected(self.states_hidden_aug2[0], self.item_num,
                                                   activation_fn=None, scope="ce-logits")  # all logits
            self.output1 = tf.reshape(self.output1_, [args.num_aug + 1, -1, self.item_num])


            # TRFL way
            self.actions = tf.placeholder(tf.int32, [None])
            self.targetQs_ = tf.placeholder(tf.float32, [args.num_aug + 1, None, item_num])
            self.targetQs_selector = tf.placeholder(tf.float32, [args.num_aug + 1, None, item_num])
            self.reward = tf.placeholder(tf.float32, [None])
            self.discount = tf.placeholder(tf.float32, [None])

            # TRFL double qlearning
            self.qloss = 0
            for i in range(args.num_aug + 1):
                qloss, q_learning = trfl.double_qlearning(self.output1[i], self.actions, self.reward, self.discount,
                                                          self.targetQs_[i], self.targetQs_selector[i])
                self.qloss += qloss

            celoss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)
            self.loss = tf.reduce_mean(args.q_w * celoss + self.qloss)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings = tf.Variable(tf.random_normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                       name='state_embeddings')
        pos_embeddings = tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                     name='pos_embeddings')
        all_embeddings['state_embeddings'] = state_embeddings
        all_embeddings['pos_embeddings'] = pos_embeddings
        return all_embeddings


def evaluate(sess):
    eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val_sessions.df'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated += 1
            if evaluated >= len(eval_ids):
                break
        mask = np.ones([args.num_aug, len(actions), state_size])
        prediction = sess.run(FMLPRec1.output2, feed_dict={FMLPRec1.inputs: states, FMLPRec1.len_state: len_states,
                                                         FMLPRec1.is_training: False, FMLPRec1.mask: mask})
        sorted_list = np.argsort(prediction)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks,
                      hit_purchase, ndcg_purchase)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#############################################################')


if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, including state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk = [5, 10, 15, 20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    tf.reset_default_graph()

    FMLPRec1 = FMLPRecModel(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                            state_size=state_size, args=args, name='FMLPRec1')

    FMLPRec2 = FMLPRecModel(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                            state_size=state_size, args=args, name='FMLPRec2')

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    saver = tf.train.Saver()

    total_step = 0
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        evaluate(sess)
        num_rows = replay_buffer.shape[0]
        num_batches = int(num_rows / args.batch_size)
        for i in range(args.epoch):
            for j in range(num_batches):
                batch = replay_buffer.sample(n=args.batch_size).to_dict()

                next_state = list(batch['next_state'].values())
                len_next_state = list(batch['len_next_states'].values())

                if_mask_nstate = np.array(len_next_state) > args.cond_len
                mask = np.ones([args.batch_size, state_size])
                mask_nstate = np.tile(np.expand_dims(mask, axis=0), (args.num_aug, 1, 1))
                for na in range(args.num_aug):
                    for k in range(args.batch_size):
                        if if_mask_nstate[k]:
                            ind_mask = mask_nstate[na][k]
                            ind = np.random.randint(0, state_size)
                            ind_mask[ind] = 0.0
                            mask_nstate[na][k] = ind_mask
                # double q learning, pointer is for selecting which network  is target and which is main
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = FMLPRec1
                    target_QN = FMLPRec2
                else:
                    mainQN = FMLPRec2
                    target_QN = FMLPRec1
                target_Qs = sess.run(target_QN.output1,
                                     feed_dict={target_QN.inputs: next_state,
                                                target_QN.len_state: len_next_state,
                                                target_QN.is_training: True,
                                                target_QN.mask: mask_nstate})
                target_Qs_selector = sess.run(mainQN.output1,
                                              feed_dict={mainQN.inputs: next_state,
                                                         mainQN.len_state: len_next_state,
                                                         mainQN.is_training: True,
                                                         mainQN.mask: mask_nstate})

                is_done = list(batch['is_done'].values())
                for index in range(target_Qs.shape[1]):
                    if is_done[index]:
                        target_Qs[:, index, :] = np.zeros([item_num])

                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                action = list(batch['action'].values())
                is_buy = list(batch['is_buy'].values())
                reward = []
                for k in range(len(is_buy)):
                    reward.append(reward_buy if is_buy[k] == 1 else reward_click)
                discount = [args.discount] * len(action)

                if_mask_state = np.array(len_state) > args.cond_len
                mask_state = np.tile(np.expand_dims(mask, axis=0), (args.num_aug, 1, 1))
                for na in range(args.num_aug):
                    for k in range(args.batch_size):
                        if if_mask_state[k]:
                            ind_mask = mask_state[na][k]
                            ind = np.random.randint(0, state_size)
                            ind_mask[ind] = 0.0
                            mask_state[na][k] = ind_mask

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs: state,
                                              mainQN.len_state: len_state,
                                              mainQN.targetQs_: target_Qs,
                                              mainQN.reward: reward,
                                              mainQN.discount: discount,
                                              mainQN.actions: action,
                                              mainQN.targetQs_selector: target_Qs_selector,
                                              mainQN.is_training: True,
                                              mainQN.mask: mask_state})
                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 2000 == 0:
                    evaluate(sess)
        # saver.save(sess, save_file)
