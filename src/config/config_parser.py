import yaml
from easydict import EasyDict as edict
import os


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return edict(cfg_dict)

if __name__ == "__main__":
    config_path = os.path.join("configs", "English_BRUSH_HW_GENERATOR.yml")
    cfg = load_config(config_path)

    print("Base LR:", cfg.HYPERPARAMETER.BASE_LR)
    print("Image Size:", cfg.ENV.IMG_H, cfg.ENV.IMG_W)
    print("Encoder Type:", cfg.MODEL.ENCODER_TYPE)
