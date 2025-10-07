"""Tools for computing FID after training"""

import logging
import time

from absl import app, flags
from ml_collections.config_flags import config_flags

from prox_diff.utils.img_exp.engine_eval import compute_fid_for_ckpt

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")
flags.DEFINE_string("mode", "compute_fid_for_ckpt", "Mode of operation.")
flags.DEFINE_string("ckpt_dir", None, "Path to save model checkpoints.")
flags.DEFINE_integer("ckpt_num", None, "Checkpoint number to use.")
flags.DEFINE_string("ckpt_path", None, "Path to the checkpoint file.")
flags.DEFINE_string("output_root", None, "Path to the output root directory.")
flags.DEFINE_bool("overwrite", False, "Overwrite existing results.")
flags.mark_flags_as_required(["config"])


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main(argv):
    start_time = time.time()

    if FLAGS.mode == "compute_fid_for_ckpt":
        compute_fid_for_ckpt(
            FLAGS.config,
            FLAGS.ckpt_path,
            overwrite=FLAGS.overwrite,
            output_root=FLAGS.output_root,
        )
    else:
        raise ValueError(f"Unknown mode: {FLAGS.mode}")

    logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    app.run(main)
