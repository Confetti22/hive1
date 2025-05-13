from yacs.config import CfgNode as CN
def load_cfg(cfg_path):
    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_path)
    return cfg

