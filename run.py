import hydra
import os
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="configs/", config_name="run.yaml", version_base="1.1")
def main(config: DictConfig):
    from utils import experiments as exp
    #return exp.evaluate_threshold(config)
    return exp.compare_with_ln(config)

if __name__ == "__main__":
    main()
