
# Estimator Pamrs
entry_point = "training_worker.py"
source_dir = 'src'

#Training Params
default_instance_type = "ml.c4.2xlarge" #For GPU use 'ml.p3.2xlarge'
default_instance_pool = 1
default_job_duration = 3600
default_hyperparam_preset = 'src/markov/presets/preset_hyperparams.json'
tmp_hyperparam_preset = 'src/markov/presets/preset_hyperparams_tmp.py'

#Track Details:
default_track_name = 'reinvent_base'
track_name = ['reinvent_base', 'reinvent_carpet', 'reinvent_concrete', 'reinvent_wood', 'AWS_track',
              'Bowtie_track', 'Oval_track',  'Straight_track']

# Evaluation Trials
evaluation_trials = 5

## Policy and Model Meta Data
envir_file_local = 'src/markov/environments/deepracer_racetrack_env.py'
reward_file_local = 'src/markov/rewards/reward_estimator_optimal_path.py'
model_meta_file_local = 'src/markov/actions/model_metadata_10_state.json'
model_meta_file_custom_local = 'src/markov/actions/model_metadata_tmp.json'
presets_file_local = 'src/markov/presets/default.py'
