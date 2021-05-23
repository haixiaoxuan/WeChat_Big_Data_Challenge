import mlflow
mlflow.set_experiment('rec_sys')
mlflow.start_run(nested=True)


import comm
import baseline


if __name__ == "__main__":

    comm.main()
    baseline.main("offline_train")
    baseline.main("evaluate")
    # baseline.main("online_train")
    # baseline.main("submit")
