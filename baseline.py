# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow import feature_column as fc
from comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST
from evaluation import uAUC, compute_weighted_score
import config as self_config
import mlflow


tf.logging.set_verbosity(tf.logging.INFO)


class WideAndDeep(object):

    def __init__(self, linear_feature_columns, dnn_feature_columns, stage, action):
        """
            :param linear_feature_columns: List of tensorflow feature_column
            :param dnn_feature_columns: List of tensorflow feature_column
            :param stage: String. Including
                                - "online_train"
                                - "offline_train"
                                - "evaluate"
                                - "submit"
            :param action: String. Including
                                - "read_comment"
                                - "like"
                                - "click_avatar"
                                - "favorite"
                                - "forward"
                                - "comment"
                                - "follow"
        """
        super(WideAndDeep, self).__init__()
        self.num_epochs_dict = self_config.num_epochs_dict
        self.estimator = None
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.stage = stage
        self.action = action

    def build_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"

        # 构造模型目录
        model_checkpoint_stage_dir = os.path.join(self_config.model_checkpoint_dir, stage, self.action)
        if not os.path.exists(model_checkpoint_stage_dir):
            os.makedirs(model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            del_file(model_checkpoint_stage_dir)

        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=self_config.learning_rate,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1)

        config = tf.estimator.RunConfig(model_dir=model_checkpoint_stage_dir, tf_random_seed=self_config.SEED)
        self.estimator = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_checkpoint_stage_dir,
            linear_feature_columns=self.linear_feature_columns,
            dnn_feature_columns=self.dnn_feature_columns,
            dnn_hidden_units=self_config.dnn_hidden_units,
            dnn_optimizer=optimizer,
            config=config
        )

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        """
            把DataFrame转为tensorflow dataset
            :param df: pandas dataframe.
            :param stage: String.
            :param action: String.
            :param shuffle: Boolean.
            :param batch_size: Int.
            :param num_epochs: Int.
            :return: tf.data.Dataset object.
        """
        if stage != "submit":
            label = df[action]
            ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=self_config.SEED)

        ds = ds.batch(batch_size)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs)
        return ds

    def input_fn_train(self, df, stage, action, num_epochs):
        return self.df_to_dataset(df, stage, action, shuffle=True, batch_size=self_config.batch_size,
                                  num_epochs=num_epochs)

    def input_fn_predict(self, df, stage, action):
        return self.df_to_dataset(df, stage, action, shuffle=False, batch_size=len(df), num_epochs=1)

    def train(self):
        """
        训练单个行为的模型
        """
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=self.action,
                                                                       day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(self_config.root_path, self.stage, file_name)
        df = pd.read_csv(stage_dir)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(df, self.stage, self.action, self.num_epochs_dict[self.action])
        )

    def evaluate(self):
        """
            评估单个行为的uAUC值
        """
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        else:
            # 测试集，所有action在同一个文件
            action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                       day=STAGE_END_DAY[self.stage])
        evaluate_dir = os.path.join(self_config.root_path, self.stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        labels = df[self.action].values

        # 评估uAUC
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    def predict(self):
        """
            预测单个行为的发生概率
        """
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action="all",
                                                                       day=STAGE_END_DAY[self.stage])
        submit_dir = os.path.join(self_config.root_path, self.stage, file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time() - t) * 1000.0 / len(df) * 2000.0
        return df[["userid", "feedid"]], logits, ts


def del_file(path):
    """
        删除path目录下的所有内容
    """
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def get_feature_columns():
    """
        获取特征列
    """
    dnn_feature_columns = list()
    linear_feature_columns = list()

    # DNN features
    user_cate = fc.categorical_column_with_hash_bucket("userid", 40000, tf.int64)
    feed_cate = fc.categorical_column_with_hash_bucket("feedid", 240000, tf.int64)
    author_cate = fc.categorical_column_with_hash_bucket("authorid", 40000, tf.int64)
    bgm_singer_cate = fc.categorical_column_with_hash_bucket("bgm_singer_id", 40000, tf.int64)
    bgm_song_cate = fc.categorical_column_with_hash_bucket("bgm_song_id", 60000, tf.int64)

    user_embedding = fc.embedding_column(user_cate, self_config.embed_dim, max_norm=self_config.embed_l2)
    feed_embedding = fc.embedding_column(feed_cate, self_config.embed_dim, max_norm=self_config.embed_l2)
    author_embedding = fc.embedding_column(author_cate, self_config.embed_dim, max_norm=self_config.embed_l2)
    bgm_singer_embedding = fc.embedding_column(bgm_singer_cate, self_config.embed_dim)
    bgm_song_embedding = fc.embedding_column(bgm_song_cate, self_config.embed_dim)

    dnn_feature_columns.append(user_embedding)
    dnn_feature_columns.append(feed_embedding)
    dnn_feature_columns.append(author_embedding)
    dnn_feature_columns.append(bgm_singer_embedding)
    dnn_feature_columns.append(bgm_song_embedding)

    # Linear features
    video_seconds = fc.numeric_column("videoplayseconds", default_value=0.0)
    device = fc.numeric_column("device", default_value=0.0)
    linear_feature_columns.append(video_seconds)
    linear_feature_columns.append(device)

    # 行为统计特征
    for b in FEA_COLUMN_LIST:
        feed_b = fc.numeric_column(b + "sum", default_value=0.0)
        linear_feature_columns.append(feed_b)

        user_b = fc.numeric_column(b + "sum_user", default_value=0.0)
        linear_feature_columns.append(user_b)
    return dnn_feature_columns, linear_feature_columns


def main(stage):
    t = time.time()

    dnn_feature_columns, linear_feature_columns = get_feature_columns()
    print('Stage: %s' % stage)

    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None

    for action in ACTION_LIST:
        print("Action:", action)
        model = WideAndDeep(linear_feature_columns, dnn_feature_columns, stage, action)
        model.build_estimator()

        if stage in ["online_train", "offline_train"]:
            # 训练 并评估
            model.train()
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            if stage == "offline_train":
                mlflow.log_artifact(os.path.join(self_config.model_checkpoint_dir, stage, action))

        if stage == "evaluate":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        if stage == "submit":
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits

    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {
            "read_comment": 4,
            "like": 3,
            "click_avatar": 2,
            "favorite": 1,
            "forward": 1,
            "comment": 1,
            "follow": 1
        }
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)
        mlflow.log_dict({
            stage + "_uAUC": eval_dict
        })
        mlflow.log_metric(stage + "_weight_auc", weight_auc)

    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit.csv"
        submit_file = os.path.join(self_config.root_path, stage, file_name)
        print('Save to: %s' % submit_file)
        res.to_csv(submit_file, index=False)
        if stage == "submit":
            mlflow.log_artifact(submit_file)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s' % (time.time() - t))
    mlflow.log_metric(stage + "_speed_time", round((time.time() - t), 0))


if __name__ == "__main__":
    main("online_train")
