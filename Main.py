from Diffusion.Train import train, eval
from pathlib import Path

def find_latest_checkpoint(checkpoint_dir: str):
    checkpoint_files = list(Path(checkpoint_dir).glob('ckpt_*.pt'))
    if not checkpoint_files:
        return None, 0
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
    return str(latest_checkpoint), int(latest_checkpoint.stem.split('_')[1])

def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,

        # Newly added
        "latest_checkpoint": None,
        "latest_epoch": 0
        }
    if model_config is not None:
        modelConfig = model_config

    if modelConfig["state"] == "train":
        latest_checkpoint, latest_epoch = find_latest_checkpoint(modelConfig["save_weight_dir"])
        if latest_checkpoint:
            # modelConfig["training_load_weight"] = latest_checkpoint.split('/')[-1]
            # modelConfig["latest_epoch"] = latest_epoch
            modelConfig["training_load_weight"] = "ckpt_150_.pt"
            modelConfig["latest_epoch"] = 150
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
