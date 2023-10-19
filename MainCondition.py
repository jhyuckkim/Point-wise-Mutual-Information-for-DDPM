from DiffusionFreeGuidence.TrainCondition import train, eval
import os

def main(model_config=None):
    base_model_config = {
        "state": "train", # or eval
        "epoch": 100,
        "batch_size": 80,
        "T": 4000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.0,
        "save_dir": "./CheckpointsCondition4000/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_99_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        base_model_config.update(model_config)

    for i in range(10):
        # Update save directory for each model
        base_model_config["save_dir"] = f"./CheckpointsCondition4000/Model_{i}/"
        print(base_model_config["save_dir"])
        if not os.path.exists(base_model_config["save_dir"]):
            os.makedirs(base_model_config["save_dir"])

        # Train the model
        if base_model_config["state"] == "train":
            train(base_model_config)
        else:
            eval(base_model_config)


if __name__ == '__main__':
    main()
