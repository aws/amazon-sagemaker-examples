"""Shared callbacks for training across different modalities."""

from transformers import TrainerCallback, TrainerState, TrainerControl


class SaveOnBestMetricCallback(TrainerCallback):
    """Save checkpoint only when metric improves."""
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        metric_value = metrics.get(args.metric_for_best_model)
        if metric_value is None:
            return control
        
        if state.best_metric is None:
            control.should_save = True
        else:
            if args.greater_is_better:
                if metric_value > state.best_metric:
                    control.should_save = True
            else:
                if metric_value < state.best_metric:
                    control.should_save = True
        
        return control
