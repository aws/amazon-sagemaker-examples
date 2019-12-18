import subprocess
import os
import logging
import numpy as np
import json


class VWError(Exception):
    """ Class for errors """

    def __init__(self, message):
        super(VWError, self).__init__()
        self.message = message


class VWModelDown(Exception):
    """ When the model is down """

    def __init__(self):
        super(VWModelDown, self).__init__("The model is down")


class VWAgent:
    def __init__(self,
                 cli_args="",
                 model_path=None,
                 test_only=False,
                 quiet_mode=False,
                 adf_mode=True,
                 num_actions=None,
                 top_k=1,
                 output_dir=None):
        """
        Args:
            model_path: location of the model weights
            cli_args: additional args to pass to VW
        """
        # Save args for model serialization
        self._save_args(locals())

        self.logger = logging.getLogger("vw_model.VWModel")
        self.logger.info("creating an instance of VWModel")

        # if a model does not have a current proc it is uninitialized
        self.closed = False
        self.current_proc = None
        self.test_mode = test_only
        self.adf_mode = adf_mode
        self.top_k = top_k

        if not self.adf_mode:
            assert num_actions, "Please specify num_actions."
        self.num_actions = num_actions

        if len(cli_args) == 0:
            raise VWError("No arguments specified to create/load a VW model.")

        # command arguments for shell process
        # we redirect the score to /dev/stdout to capture it
        self.cmd = ["vw", *cli_args.split(), "-p", "/dev/stdout"]

        if quiet_mode:
            self.cmd.append("--quiet")

        if self.test_mode:
            self.cmd.extend(["--testonly"])

        if model_path:
            self.model_file = os.path.expanduser(os.path.expandvars(model_path))
            self.cmd.extend(["-i", self.model_file])
        else:
            # Apply interactions if training for the first time only. VW saves the q parameter.
            self.cmd.extend(["-q", "sd"])

        if output_dir:
            self.output_dir = output_dir
            self.model_file = os.path.join(output_dir, "vw.model")
            self.cmd.extend(["-f", self.model_file, "--save_resume"])

        self.logger.info("successfully created VWModel")
        self.logger.info("command: %s", self.cmd)

    def _save_args(self, args):
        args_to_pop = ["self", "model_path", "output_dir"]
        for arg in args_to_pop:
            args.pop(arg)
        self.args = args

    def start(self):
        """
        Starts the VW C++ process
        """
        if self.closed:
            raise VWError("Cannot start a closed model")
        if self.current_proc is not None:
            raise VWError("Cannot start a model with an active current_proc")

        # note bufsize=1 will make sure we immediately flush each output
        # line so that we can keep scoring the model.
        # bufsize=1 means line buffered.
        self.current_proc = subprocess.Popen(self.cmd, bufsize=1,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=False)

        self.logger.info("Started VW process!")

        # TODO: Check for errors in CLI args by polling the process

    def learn(self, shared_features, candidate_arms_features, action_index, action_prob,
              reward, user_id=None, candidate_ids=None, cost_fn=None):

        if not cost_fn:
            cost_fn = lambda x: 1 - x

        cost = cost_fn(reward)

        shared_vw_features = None
        item_vw_features = None

        shared_feature_dim = 0
        if shared_features is not None:
            shared_vw_features = self.transform_context(shared_features)
            shared_feature_dim = len(shared_features)

        if user_id is not None:
            if shared_vw_features:
                shared_vw_features += " user_%s" % user_id
            else:
                shared_vw_features = "user_%s" % user_id

        start_index = shared_feature_dim + 1
        num_actions = self.num_actions
        if candidate_arms_features is not None:
            item_vw_features = [self.transform_context(i, start_index) for i in candidate_arms_features]
            num_actions = len(item_vw_features)
        else:
            item_vw_features = [f"a{i}" for i in range(num_actions)]

        parsed_example = self.generate_experience_string_multiline(shared_vw_features,
                                                                   item_vw_features,
                                                                   action_index,
                                                                   cost,
                                                                   action_prob)
        # TODO: Error handling in parsing the given example
        if self.current_proc is None:
            raise VWError("trying to learn model when current_proc is None")

        if self.current_proc.returncode is not None:
            raise VWModelDown()

        self.current_proc.stdin.write(parsed_example.encode())
        self.current_proc.stdin.flush()
        self.current_proc.stdout.flush()

        # VW will make a prediction on each training instance too.
        self.current_proc.stdout.readline()
        self.current_proc.stdout.readline()

    def _choose_action(self, vw_prediction_str):
        # TODO: Error handling in parsing the given example
        if self.current_proc is None:
            raise VWError("trying to score model when current_proc is None")

        if self.current_proc.returncode is not None:
            raise VWModelDown()

        # TODO: Write to stdin in chunks so that PIPE buffer never overflows
        self.current_proc.stdin.write(vw_prediction_str.encode())

        # we need to flush to score & collect the score
        # otherwise one needs to wait for the process to end
        self.current_proc.stdin.flush()
        self.current_proc.stdout.flush()

        vw_scores_string = self.current_proc.stdout.readline().decode()
        if len(vw_scores_string.strip()) == 0:
            print("No output gotten from VW")
        else:
            self.last_scores_string = vw_scores_string

        # Need an extra readline as VW returns a blank line
        self.current_proc.stdout.readline().decode()

        action, prob = vw_scores_string.strip().split(",")[0].split(":")
        action = int(action)
        prob = float(prob)
        if prob == 0:
            scores_dict = eval("{" + vw_scores_string + "}")
            item_scores = np.array([scores_dict[i] for i in range(len(scores_dict))])
            item_probs = (item_scores / item_scores.sum())
            action = np.argmax(item_probs)
            prob = item_probs[action]
        return action, prob

    def choose_actions(self, shared_features, candidate_arms_features,
                       user_id=None, candidate_ids=None, top_k=None):
        shared_vw_features = None
        item_vw_features = None
        action_ids = None
        top_k_actions = top_k if top_k else self.top_k

        shared_feature_dim = 0
        if shared_features is not None:
            shared_vw_features = self.transform_context(shared_features)
            shared_feature_dim = len(shared_features)

        if user_id is not None:
            if shared_vw_features:
                shared_vw_features += " user_%s" % user_id
            else:
                shared_vw_features = "user_%s" % user_id

        start_index = shared_feature_dim + 1
        num_actions = self.num_actions

        if candidate_arms_features is not None:
            item_vw_features = [self.transform_context(i, start_index) for i in candidate_arms_features]
            num_actions = len(item_vw_features)
        else:
            item_vw_features = [f"a{i}" for i in range(num_actions)]

        action_ids = list(range(num_actions))

        actions, probs = [], []
        best_action_index, action_prob = None, None
        for k in range(top_k_actions):
            if k > 0:
                item_vw_features.pop(best_action_index)
                action_ids.pop(best_action_index)
            parsed_example = self.generate_prediction_string_multiline(shared_features=shared_vw_features,
                                                                       item_features=item_vw_features)
            best_action_index, action_prob = self._choose_action(parsed_example)
            actions.append(action_ids[best_action_index])
            probs.append(action_prob)

        return actions, probs

    @staticmethod
    def generate_prediction_string_multiline(shared_features, item_features):
        string = ""
        if shared_features is not None:
            string = f"shared |s {shared_features}\n"

        for i in range(len(item_features)):
            string += f" |d {item_features[i]}\n"
        return string + "\n"

    @staticmethod
    def generate_experience_string_multiline(shared_features, item_features,
                                             action, cost, probability):
        string = ""
        if shared_features is not None:
            string = f"shared |s {shared_features}\n"

        string += f"0:{cost}:{probability} |d {item_features[action]}\n"

        # Sending all items is not required using if using multi-task regression
        # for i in range(len(item_features)):
        #     if action == i:
        #         string += f"0:{cost}:{probability} |d {item_features[i]}\n"
        #     else:
        #         string += f" |d {item_features[i]}\n"

        return string + "\n"

    @staticmethod
    def transform_context(feature_vector, start_index=1):
        out_string = " ".join(["%s:%s" % (i + start_index, j) for i, j in enumerate(feature_vector)])
        return out_string

    @staticmethod
    def load_model(metadata_loc, weights_loc, test_only=True, quiet_mode=True, output_dir=None):
        """
        Initialize vw model with given metadata and weights locations
        """
        with open(metadata_loc) as f:
            metadata = f.read().strip()
        metadata = json.loads(metadata)

        metadata["quiet_mode"] = quiet_mode
        metadata["test_only"] = test_only
        metadata["model_path"] = weights_loc
        metadata["output_dir"] = output_dir
        metadata["cli_args"] = metadata.get("cli_args", "")

        return VWAgent(**metadata)

    def save_model(self, close=False):
        """Call to save metadata. close=True closes the VW process """
        metadata_file = os.path.join(self.output_dir, "vw.metadata")
        with open(metadata_file, "w") as f:
            f.write(json.dumps(self.args))
            f.write("\n")
        if close:
            return self.close()

    def close(self):
        """
        Closes the model.
        """
        training_info = ""
        if self.current_proc is not None:
            self.current_proc.stdin.close()
            self.current_proc.stdout.close()
            training_info = self.current_proc.stderr.read()
            self.current_proc.stderr.close()

            # putting wait after terminate will
            # make sure the process is terminated
            # before proceeding to the next line
            self.current_proc.terminate()
            self.current_proc.wait()

            self.current_proc = None

        self.closed = True
        return training_info
