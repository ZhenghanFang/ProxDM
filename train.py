"""Main training script."""

from absl import app, flags
from ml_collections.config_flags import config_flags

from prox_diff.utils.img_exp.engine_train import train
from prox_diff.utils.misc import setup_logging

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")
flags.DEFINE_string("ckpt_dir", None, "Path to save model checkpoints.")
flags.DEFINE_bool("debug", False, "Enable debug mode.")
flags.mark_flags_as_required(["config", "ckpt_dir"])


def main(argv):
    setup_logging(FLAGS.debug)
    config = FLAGS.config
    train(config, FLAGS.ckpt_dir, FLAGS.debug)


if __name__ == "__main__":
    app.run(main)
