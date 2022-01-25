import mlflow


model_checkpoint_dir = "./data/model"
root_path = "./data/"
SEED = 666


batch_size = 512
embed_dim = 10
learning_rate = 0.1
# If not None, embedding values are l2-normalized to this value.
embed_l2 = None

dnn_hidden_units = [32, 8]


base_epochs = 1
num_epochs_dict = {
    "read_comment": base_epochs,
    "like": base_epochs,
    "click_avatar": base_epochs,
    "favorite": base_epochs * 2,
    "forward": base_epochs,
    "comment": base_epochs,
    "follow": base_epochs
}


# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10, "favorite": 10}
# 各个行为构造训练数据的天数(负样本下采样限制)
ACTION_DAY_NUM = {"read_comment": 10, "like": 10, "click_avatar": 10, "forward": 10, "comment": 10, "follow": 10, "favorite": 10}


mlflow.log_params({
    "batch_size": batch_size,
    "embed_dim": embed_dim,
    "learning_rate": learning_rate,
    "embed_l2": learning_rate,
    "dnn_hidden_units": dnn_hidden_units,
    "action_sample_rate": ACTION_SAMPLE_RATE,
    "action_day_num": ACTION_DAY_NUM,
    "num_epochs_dict": num_epochs_dict
})

mlflow.log_artifact("./config.py")
mlflow.log_artifact("./baseline.py")
mlflow.log_artifact("./comm.py")
mlflow.log_artifact("./evaluation.py")