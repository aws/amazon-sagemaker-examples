import inspect

from smdebug_rulesconfig.profiler_rules.utils import validate_percentile, validate_positive_integer


class ProfilerRuleBase:
    def __init__(self, **rule_parameters):
        self.rule_name = self.__class__.__name__
        rule_parameters["rule_to_invoke"] = self.__class__.__name__
        self.rule_parameters = rule_parameters


class BatchSize(ProfilerRuleBase):
    def __init__(
        self,
        cpu_threshold_p95=70,
        gpu_threshold_p95=70,
        gpu_memory_threshold_p95=70,
        patience=1000,
        window=500,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU is underulitized because of the batch size being too small.
        To detect this the rule analyzes the average GPU memory footprint, CPU and GPU utilization.
        If utilization on CPU, GPU and memory footprint is on average low , it may indicate that user
        can either run on a smaller instance type or that batch size could be increased. This analysis does not
        work for frameworks that heavily over-allocate memory. Increasing batch size could potentially lead to
        a processing/dataloading bottleneck, because more data needs to be pre-processed in each iteration.

        :param cpu_threshold_p95: defines the threshold for 95th quantile of CPU utilization.Default is 70%.
        :param gpu_threshold_p95: defines the threshold for 95th quantile of GPU utilization.Default is 70%.
        :param gpu_memory_threshold_p95: defines the threshold for 95th quantile of GPU memory utilization.Default is 70%.
        :param patience: defines how many datapoints to capture before Rule runs the first evluation. Default 100
        :param window: window size for computing quantiles.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_percentile("cpu_threshold_p95", cpu_threshold_p95)
        validate_percentile("gpu_threshold_p95", gpu_threshold_p95)
        validate_percentile("gpu_memory_threshold_p95", gpu_memory_threshold_p95)
        validate_positive_integer("patience", patience)
        validate_positive_integer("window", window)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            cpu_threshold_p95=cpu_threshold_p95,
            gpu_threshold_p95=gpu_threshold_p95,
            gpu_memory_threshold_p95=gpu_memory_threshold_p95,
            patience=patience,
            window=window,
            scan_interval_us=scan_interval_us,
        )


class CPUBottleneck(ProfilerRuleBase):
    def __init__(
        self,
        threshold=50,
        gpu_threshold=10,
        cpu_threshold=90,
        patience=1000,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU is underutilized due to CPU bottlenecks. Rule returns True if number of CPU bottlenecks exceeds a predefined threshold.
        :param threshold: defines the threshold behyond which Rule should return True. Default is 50 percent. So if there is a bottleneck more than 50% of the time during the training Rule will return True.
        :param gpu_threshold: threshold that defines when GPU is considered being under-utilized. Default is 10%
        :param cpu_threshold: threshold that defines high CPU utilization. Default is above 90%
        :param patience: How many values to record before checking for CPU bottlenecks. During training initilization, GPU is likely at 0 percent, so Rule should not check for underutilization immediatly. Default 1000.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_percentile("threshold", threshold)
        validate_percentile("gpu_threshold", gpu_threshold)
        validate_percentile("cpu_threshold", cpu_threshold)
        validate_positive_integer("patience", patience)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            threshold=threshold,
            gpu_threshold=gpu_threshold,
            cpu_threshold=cpu_threshold,
            patience=patience,
            scan_interval_us=scan_interval_us,
        )


class Dataloader(ProfilerRuleBase):
    def __init__(self, min_threshold=70, max_threshold=200, scan_interval_us=60000000):
        """
        This rule helps to detect how many dataloader processes are running in parallel and whether the total number is equal the number of available CPU cores.
        :param min_threshold: how many cores should be at least used by dataloading processes. Default 70%
        :param max_threshold: how many cores should be at maximum used by dataloading processes. Default 200%
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_positive_integer("min_threshold", min_threshold)
        validate_positive_integer("max_threshold", max_threshold)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            scan_interval_us=scan_interval_us,
        )
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.scan_interval_us = scan_interval_us


class GPUMemoryIncrease(ProfilerRuleBase):
    def __init__(self, increase=5, patience=1000, window=10, scan_interval_us=60 * 1000 * 1000):
        """
        This rule helps to detect large increase in memory usage on GPUs. The rule computes the moving average of continous datapoints and compares it against the moving average of previous iteration.
        :param increase: defines the threshold for absolute memory increase.Default is 5%. So if moving average increase from 5% to 6%, the rule will fire.
        :param patience: defines how many continous datapoints to capture before Rule runs the first evluation. Default is 1000
        :param window: window size for computing moving average of continous datapoints
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_positive_integer("increase", increase)
        validate_positive_integer("patience", patience)
        validate_positive_integer("window", window)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            increase=increase, patience=patience, window=window, scan_interval_us=scan_interval_us
        )


class IOBottleneck(ProfilerRuleBase):
    def __init__(
        self,
        threshold=50,
        gpu_threshold=10,
        io_threshold=50,
        patience=1000,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU is underutilized due to IO bottlenecks. Rule returns True if number of IO bottlenecks exceeds a predefined threshold.
        :param threshold: defines the threshold when Rule should return True. Default is 50 percent. So if there is a bottleneck more than 50% of the time during the training Rule will return True.
        :param gpu_threshold: threshold that defines when GPU is considered being under-utilized. Default is 70%
        :param io_threshold: threshold that defines high IO wait time. Default is above 50%
        :param patience: How many values to record before checking for IO bottlenecks. During training initilization, GPU is likely at 0 percent, so Rule should not check for underutilization immediatly. Default 1000.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_percentile("threshold", threshold)
        validate_percentile("gpu_threshold", gpu_threshold)
        validate_percentile("io_threshold", io_threshold)
        validate_positive_integer("patience", patience)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            threshold=threshold,
            gpu_threshold=gpu_threshold,
            io_threshold=io_threshold,
            patience=patience,
            scan_interval_us=scan_interval_us,
        )


class LoadBalancing(ProfilerRuleBase):
    def __init__(self, threshold=0.5, patience=1000, scan_interval_us=60 * 1000 * 1000):
        """
        This rule helps to detect issues in workload balancing between multiple GPUs.
        It computes a histogram of utilization per GPU and measures the distance between those histograms.
        If the histogram exceeds a pre-defined threshold then rule triggers.
        :param threshold: difference between 2 histograms 0.5
        :param patience: how many values to record before checking for loadbalancing issues
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_percentile("threshold", threshold)
        validate_positive_integer("patience", patience)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(threshold=threshold, patience=patience, scan_interval_us=scan_interval_us)


class LowGPUUtilization(ProfilerRuleBase):
    def __init__(
        self,
        threshold_p95=70,
        threshold_p5=10,
        window=500,
        patience=1000,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU utilization is low or suffers from fluctuations. This is checked for each single GPU on each worker node.
        Rule returns True if 95th quantile is below threshold_p95 which indicates under-utilization.
        Rule returns true if 95th quantile is above threshold_p95 and 5th quantile is below threshold_p5 which indicates fluctuations.

        :param threshold_p95: threshold for 95th quantile below which GPU is considered to be underutilized. Default is 70 percent.
        :param threshold_p5: threshold for 5th quantile. Default is 10 percent.
        :param window: number of past datapoints which are used to compute the quantiles.
        :param patience: How many values to record before checking for underutilization/fluctuations. During training initilization, GPU is likely at 0 percent, so Rule should not check for underutilization immediately. Default 1000.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_percentile("threshold_p95", threshold_p95)
        validate_percentile("threshold_p5", threshold_p5)
        validate_positive_integer("window", window)
        validate_positive_integer("patience", patience)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            threshold_p95=threshold_p95,
            threshold_p5=threshold_p5,
            window=window,
            patience=patience,
            scan_interval_us=scan_interval_us,
        )


class MaxInitializationTime(ProfilerRuleBase):
    def __init__(self, threshold=20, scan_interval_us=60 * 1000 * 1000):
        """
        This rule helps to detect if the training intialization is taking too much time. The rule waits until first
        step is available.

        :param threshold: defines the threshold in minutes to wait for first step to become available. Default is 20 minutes.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_positive_integer("threshold", threshold)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(threshold=threshold, scan_interval_us=scan_interval_us)


class OverallSystemUsage(ProfilerRuleBase):
    def __init__(self, scan_interval_us=60 * 1000 * 1000):
        """
        This rule measures overall system usage per worker node. The rule currently only aggregates values per node
        and computes their percentiles. The rule does currently not take any threshold parameters into account
        nor can it trigger. The reason behind that is that other rules already cover cases such as underutilization and
        they do it at a more fine-grained level e.g. per GPU. We may change this in the future.

        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_positive_integer("scan_interval_us", scan_interval_us)
        super().__init__(scan_interval_us=scan_interval_us)


class StepOutlier(ProfilerRuleBase):
    def __init__(self, stddev=3, mode=None, n_outliers=10, scan_interval_us=60 * 1000 * 1000):
        """
        This rule helps to detect outlier in step durations. Rule returns True if duration is larger than stddev * standard deviation.
        :param stddev: factor by which to multiply the standard deviation. Default is 3
        :param mode: select mode under which steps have been saved and on which Rule should run on. Per default rule will run on steps from EVAL and TRAIN phase.
        :param n_outliers: How many outliers to ignore before rule returns True. Default 10.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        validate_positive_integer("stddev", stddev)
        assert mode is None or isinstance(mode, str), "Mode must be a string if specified!"
        validate_positive_integer("n_outliers", n_outliers)
        validate_positive_integer("scan_interval_us", scan_interval_us)

        super().__init__(
            stddev=stddev, mode=mode, n_outliers=n_outliers, scan_interval_us=scan_interval_us
        )


class ProfilerReport(ProfilerRuleBase):
    def __init__(self, **rule_parameters):
        """
        This rule will create a profiler report after invoking all of the rules. The parameters
        used in any of these rules can be customized by following this naming scheme:

        <rule_name>_<parameter_name> : value

        Validation is also done here to ensure that:
            1. The key names follow the above format
            2. rule_name corresponds to a valid rule name.
            3. parameter_name corresponds to a valid parameter of this rule.
            4. The parameter for this rule's parameter is valid.

        :param rule_parameters: Dictionary mapping rule + parameter name to value.
        """
        invalid_key_format_error = (
            "Key ({0}) does not follow naming scheme: <rule_name>_<parameter_name>"
        )
        invalid_rule_error = (
            "{0} is an invalid rule name! Accepted rule names (case insensitive) are: {1}"
        )
        invalid_param_error = (
            "{0} is an invalid parameter name! Accepted parameter names for {1} are: {2}"
        )

        rule_classes = [
            BatchSize,
            CPUBottleneck,
            Dataloader,
            GPUMemoryIncrease,
            IOBottleneck,
            LoadBalancing,
            LowGPUUtilization,
            MaxInitializationTime,
            OverallSystemUsage,
            StepOutlier,
        ]
        rule_names = [rule.__name__ for rule in rule_classes]
        rule_classes_by_name = {rule.__name__.lower(): rule for rule in rule_classes}

        for key, val in rule_parameters.items():
            assert key.count("_") >= 1, invalid_key_format_error.format(key)
            rule_name, *parameter_name = key.split("_")
            rule_name = rule_name.lower()
            parameter_name = "_".join(parameter_name).lower()
            assert rule_name in rule_classes_by_name, invalid_rule_error.format(
                rule_name, rule_names
            )
            rule_class = rule_classes_by_name[rule_name]
            try:
                rule_class(**{parameter_name: val})
            except TypeError:
                rule_signature = inspect.signature(rule_class.__init__)
                raise TypeError(
                    invalid_param_error.format(parameter_name, rule_class.__name__, rule_signature)
                )

        super().__init__(**rule_parameters)
