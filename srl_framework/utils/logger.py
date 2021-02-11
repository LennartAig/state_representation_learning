from torch.utils.tensorboard import SummaryWriter
from srl_framework.utils.video import VideoRecorder
from srl_framework.utils.mpi_tools import mpi_statistics_scalar
from srl_framework.utils.utilities import make_dir
from datetime import datetime
import time

import torch
import joblib
import shutil
import warnings
import json

import numpy as np
import os.path as osp, atexit, os

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    """
    A general-purpose logger adapted from SpinningUP. It extends it with the possibilities
    of video logging and live analysis using tensorboard.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    
    Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

    """

    def __init__(
        self,
        log_dir=None,
        param_path="../experiment.yaml",
        output_fname="progress.txt",
        exp_name=None,
        normalized_image=False,
        args=dict(),
    ):
        """
        Initialize a Logger.

        Args
        -------
            - log_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            - output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            - exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """

        if osp.exists(log_dir):
            print(
                "Warning: Log dir %s already exists! Storing info there anyway."
                % log_dir
            )
            self.log_dir = log_dir
        else:
            self.log_dir = make_dir(log_dir)
        time.sleep(0.5)
        self.video_dir = make_dir(os.path.join(log_dir, "video"))
        time.sleep(0.5)
        self.fig_dir = make_dir(os.path.join(log_dir, "fig"))
        time.sleep(0.5)
        self.model_dir = make_dir(os.path.join(log_dir, "model"))
        time.sleep(0.5)
        self.buffer_dir = make_dir(os.path.join(log_dir, "buffer"))
        time.sleep(0.5)

        self.writer = SummaryWriter(self.log_dir)
        self.video_rec = VideoRecorder(self.video_dir)

        shutil.copy2(param_path, self.log_dir + "/experiment.yaml")
        with open(self.log_dir + "/commandline_args.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

        self.output_file = open(osp.join(self.log_dir, output_fname), "w")
        atexit.register(self.output_file.close)
        print(
            colorize("Logging data to %s" % self.output_file.name, "green", bold=True)
        )
        print(
            colorize("Tensorboard is logged to %s" % self.log_dir, "green", bold=True)
        )

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val, tensorboard=False, epoch=-1):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if tensorboard:
            self.writer.add_scalar(key, val, epoch)
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that you didn't include in the first iteration"
                % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
            % key
        )
        self.log_current_row[key] = val

    def log_tensorboard(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tensorboard`` to store values for each diagnostic,
        values are stored in tensorboard
        """
        self.writer.add_scalar(key, val)

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(
                config_json, separators=(",", ":\t"), indent=4, sort_keys=True
            )
            print(colorize("Saving config:\n", color="cyan", bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), "w") as out:
                out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        if proc_id() == 0:
            fname = "vars.pkl" if itr is None else "vars%d.pkl" % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log("Warning: could not pickle state_dict.", color="red")
            if hasattr(self, "tf_saver_elements"):
                self._tf_simple_save(itr)
            if hasattr(self, "pytorch_saver_elements"):
                self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        assert hasattr(
            self, "pytorch_saver_elements"
        ), "First have to setup saving with self.setup_pytorch_saver"
        fpath = "pyt_save"
        fpath = osp.join(self.model_dir, fpath)
        fname = "model" + ("%d" % itr if itr is not None else "") + ".pt"
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We are using a non-recommended way of saving PyTorch models,
            # by pickling whole objects (which are dependent on the exact
            # directory structure at the time of saving) as opposed to
            # just saving network weights. This works sufficiently well
            # for the purposes of Spinning Up, but you may want to do
            # something different for your personal PyTorch project.
            # We use a catch_warnings() context to avoid the warnings about
            # not being able to save the source code.
            torch.save(self.pytorch_saver_elements.state_dict(), fpath)
            torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """

        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def _try_sw_log(self, key, value, step):
        if self.writer is not None:
            self.writer.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self.writer is not None:
            # assert image.dim() == 3
            # if image.shape[0] > 3:
            images = torch.zeros(
                [int(image.shape[0] / 3), 3, image.shape[1], image.shape[2]],
                dtype=torch.float32,
            )
            for i in range(int(image.shape[0] / 3)):
                j = i * 3
                images[i] = image[j : j + 3]
            # grid = torchvision.make_grid(image.unsqueeze(1))
            self.writer.add_images(key, images, step)

    def _try_sw_log_image_seq(self, key, image, step):
        if self.writer is not None:
            self.writer.add_images(key, image, step)

    def _try_sw_log_video(self, key, frames, step):
        if self.writer is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self.writer.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self.writer is not None:
            self.writer.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + "_w", param.weight.data, step)
        if hasattr(param.weight, "grad") and param.weight.grad is not None:
            self.log_histogram(key + "_w_g", param.weight.grad.data, step)
        if hasattr(param, "bias"):
            self.log_histogram(key + "_b", param.bias.data, step)
            if hasattr(param.bias, "grad") and param.bias.grad is not None:
                self.log_histogram(key + "_b_g", param.bias.grad.data, step)

    def log_image(self, key, image, step):
        # assert key.startswith('train') or key.startswith('eval')
        if image.ndim == 4:
            self._try_sw_log_image_seq(key, image, step)
        else:
            self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step):
        assert key.startswith("train") or key.startswith("eval")
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith("train") or key.startswith("eval")
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, "train")
        self._eval_mg.dump(step, "eval")

    def log_grads_to_histogram(self, mo):
        _limits = np.array([float(i) for i in range(len(gradmean))])
        _num = len(gradmean)
        self.writer.add_histogram_raw(
            tag=netname + "/abs_mean",
            min=0.0,
            max=0.3,
            num=_num,
            sum=gradmean.sum(),
            sum_squares=np.power(gradmean, 2).sum(),
            bucket_limits=_limits,
            bucket_counts=gradmean,
            global_step=global_step,
        )
        # where gradmean is np.abs(p.grad.clone().detach().cpu().numpy()).mean()
        # # _limits is the x axis, the layers and
        # _mean = {}
        for i, name in enumerate(layers):
            _mean[name] = gradmean[i]
        self.writer.add_scalars(netname + "/abs_mean", _mean, global_step=global_step)

    def plot_grad_flow(self, named_parameters, epoch):
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        fig = (plt.figure(),)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )
        fig.savefig(
            self.fig_dir + "gradient_flow_epoch{}.png".format(epoch),
            bbox_inches="tight",
            dpi=150,
        )


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(
        self,
        key,
        val=None,
        with_min_and_max=False,
        average_only=False,
        tensorboard=True,
        epoch=0,
    ):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val, tensorboard, epoch)
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(
                key if average_only else "Average" + key,
                stats[0],
                tensorboard=tensorboard,
                epoch=epoch,
            )
            if not (average_only):
                super().log_tabular(
                    "Std" + key, stats[1], tensorboard=tensorboard, epoch=epoch
                )
            if with_min_and_max:
                super().log_tabular(
                    "Max" + key, stats[3], tensorboard=tensorboard, epoch=epoch
                )
                super().log_tabular(
                    "Min" + key, stats[2], tensorboard=tensorboard, epoch=epoch
                )
        self.epoch_dict[key] = []

    def log_tensorboard(
        self, key, val=None, with_min_and_max=False, average_only=False
    ):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic to tensorboard only.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tensorboard(key, val, tensorboard)
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tensorboard(key if average_only else "Average" + key, stats[0])
            if not (average_only):
                super().log_tensorboard("Std" + key, stats[1])
            if with_min_and_max:
                super().log_tensorboard("Max" + key, stats[3])
                super().log_tensorboard("Min" + key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = (
            np.concatenate(v)
            if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
            else v
        )
        return mpi_statistics_scalar(vals)


def create_epoch_logger(param, parameter_path, args):
    env_type = args.env_type
    task_name = args.task_name if env_type == "dmc" else ""
    env_name = args.domain_name + task_name
    obs_type = args.obs_type
    rl_name = param.RL.NAME
    if param.SRL.USE:
        srl_name = ""
        for model in param.SRL.MODELS:
            if model == "LATENT":
                srl_name = srl_name + model + "_" + param.SRL.LATENT.TYPE + "_"
            else:
                srl_name = srl_name + model + "_"
        for loss in param.SRL.LOSSES:
            srl_name = srl_name + loss + "_"
        if param.SRL.ONLY_PRETRAINING:
            srl_name = srl_name + "pre"
        else:
            if param.SRL.JOINT_TRAINING:
                srl_name = srl_name + "joint"
            else:
                srl_name = srl_name + "alt"
    else:
        srl_name = "NoSRL"
    now = datetime.now()
    timestamp = now.strftime("%m_%d_%Y_%H_%M")
    fileDir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(
        fileDir, "../../log", obs_type, env_type, env_name, rl_name, srl_name, timestamp
    )
    return EpochLogger(
        log_dir=log_dir,
        param_path=parameter_path,
        normalized_image=args.normalize_obs,
        args=args,
    )
