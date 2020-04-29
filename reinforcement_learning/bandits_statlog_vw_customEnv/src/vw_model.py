import subprocess
import os
import logging
import numpy as np

logger = logging.getLogger("vw_model.VWModel")
logger.setLevel(logging.INFO)


class VWError(Exception):
    """ Class for errors """

    def __init__(self, message):
        super(VWError, self).__init__()
        self.message = message


class VWModelDown(Exception):
    """ When the model is down """

    def __init__(self):
        super(VWModelDown, self).__init__("The model is down")


class VWModel:
    def __init__(self, model_path=None, cli_args="", test_only=True, quiet_mode=True):
        """ VWModel object starts a VW CLI process and communicates with it using pipes
        Args:
            model_path (str): location of the model weights
            cli_args (str): additional args to pass to VW
            test_only (boolean): If True, no learning is done.
            test_only (boolean): If True, VW does not print any training progress or loss
        """
        self.logger = logging.getLogger("VW CLI")
        self.logger.info("creating an instance of VWModel")

        # if a model does not have a current proc it is
        # uninitialized
        self.closed = False
        self.current_proc = None
        self.test_mode = test_only

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

        self.logger.info("successfully created VWModel")
        self.logger.info("command: %s", self.cmd)

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

        # Check if process didn't close with some error
        if self.test_mode:
            try:
                self.predict([])
            except Exception as e:
                self.logger.exception("Unable to load VW model. Please check the arguments.")
                raise VWError("Cannot load the model with the provided arguments: %s" % e)

    def learn(self, context_vector, action, cost, probability):
        """ Learn on a given experience
        Args:
            context_vector (list or np.array): A vector of context features
            action (int): The action ID that was taken (starts with 1)
            cost (float): Cost of taking the action. Can be -reward or 1-reward
            probability (float): Probability of taking the action
        """
        # parsed_example is a string that looks like:
        # "| 1:0.1 2:0.2\n"
        parsed_example = self.parse_example(context_vector) + "\n"

        parsed_example = f"{action}:{cost}:{probability} {parsed_example}"
            
        if self.current_proc is None:
            raise VWError("trying to learn model when current_proc is None")

        if self.current_proc.returncode is not None:
            raise VWModelDown()

        self.current_proc.stdin.write(parsed_example.encode())
        self.current_proc.stdin.flush()
        self.current_proc.stdout.flush()

        # VW will make a prediction on each training instance too. Read a line to keep the stdout PIPE empty
        self.current_proc.stdout.readline()

    def predict(self, context_vector):
        """
        Scores an example using the shell process
        Args:
            context_vector (list or np.array): A vector of context features
        Returns:
            np.array: A numpy array of action probabilities
        """
        parsed_example = self.parse_example(context_vector) + "\n"

        if self.current_proc.returncode is not None:
            raise VWModelDown()

        self.current_proc.stdin.write(parsed_example.encode())

        self.current_proc.stdin.flush()
        self.current_proc.stdout.flush()

        scores = np.array(list(map(float, self.current_proc.stdout.readline().split())))
        scores = (scores / scores.sum())
        return scores
    
    def evaluate(self, context_vector, action, cost, probability, label):
        """ Used when evaluating a policy offline using logged bandits dataset 
        Args:
            context_vector (list or np.array): A vector of context features
            action (int): The action ID that was taken (starts with 1) by the old policy
            cost (float): Cost of taking the action. Can be -reward or 1-reward
            probability (float): Probability of taking the action
            label (int): Action that the current policy (to be evaluated) predicts, given the context
        """
        parsed_example = self.parse_example(context_vector) + "\n"
        parsed_example = f"{label} {action}:{cost}:{probability} {parsed_example}"
        
        # TODO: Error handling in parsing the given example
        if self.current_proc is None:
            raise VWError("trying to score model when current_proc is None")

        if self.current_proc.returncode is not None:
            raise VWModelDown()

        self.current_proc.stdin.write(parsed_example.encode())
            
        # we need to flush to score & collect the score
        # otherwise one needs to wait for the process to end
        self.current_proc.stdin.flush()
        self.current_proc.stdout.flush()

        # VW will make a prediction on each eval instance.
        # To avoid PIPE overflow
        self.current_proc.stdout.readline()
                
    @staticmethod
    def parse_example(context_vector):
        """ Parses the list of context features
        Args:
            context_vector (list or np.array): A vector of context features
        Returns:
            str: a feature string interpretable by VowpalWabbit
        """
        out_string = " ".join(["%s:%s" % (i + 1, j) for i, j in enumerate(context_vector)])
        out_string = "| %s" % out_string
        return out_string

    @staticmethod
    def load_vw_model(metadata_loc, weights_loc, test_only=True, quiet_mode=True):
        """ Initialize vw model with given metadata and weights locations
        """
        with open(metadata_loc) as f:
            metadata = f.read().strip()
        return VWModel(model_path=weights_loc, cli_args=metadata, test_only=test_only, quiet_mode=quiet_mode)

    def close(self, prediction_only=False):
        """ Close the VW process
        """
        training_info = ""
        if self.current_proc is not None:
            self.current_proc.stdin.close()
            self.current_proc.stdout.close()
            self.current_proc.stderr.flush()
            training_info = self.current_proc.stderr.read()
            self.current_proc.stderr.close()
            self.current_proc.terminate()
            self.current_proc.wait()
            self.current_proc = None

        self.closed = True

        if prediction_only:
            self.logger.info("VW model loaded only for prediction.")
        else:
            return training_info
