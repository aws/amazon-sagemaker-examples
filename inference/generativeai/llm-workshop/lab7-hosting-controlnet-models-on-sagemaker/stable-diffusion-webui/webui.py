import os
import sys
import time
import importlib
import signal
import re
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import paths, timer, import_hook, errors

startup_timer = timer.Timer()

import torch
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
startup_timer.record("import torch")

import gradio
startup_timer.record("import gradio")

import ldm.modules.encoders.modules
startup_timer.record("import ldm")

from modules import extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

from huggingface_hub import hf_hub_download
import boto3
import json
import shutil
import traceback

if cmd_opts.train:
    from botocore.exceptions import ClientError
    from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
    from extensions.sd_dreambooth_extension.scripts.dreambooth import start_training_from_config, create_model
    from extensions.sd_dreambooth_extension.scripts.dreambooth import performance_wizard, training_wizard
    from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
    from modules import paths
    import glob
else:
    import requests
    cache = dict()
    s3_client = boto3.client('s3')
    s3_resource= boto3.resource('s3')

startup_timer.record("other imports")


if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "1.13.1"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.16rc425"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())


def initialize():
    check_versions()

    extensions.list_extensions()
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list extensions")

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    startup_timer.record("list SD models")

    codeformer.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    modelloader.list_builtin_upscalers()
    startup_timer.record("list builtin upscalers")

    modules.scripts.load_scripts()
    startup_timer.record("load scripts")

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    try:
        modules.sd_models.load_model()
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)
    startup_timer.record("load SD checkpoint")

    shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title

    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    startup_timer.record("opts onchange")

    shared.reload_hypernetworks()
    startup_timer.record("reload hypernets")

    ui_extra_networks.intialize()
    ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

    extra_networks.initialize()
    extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
    startup_timer.record("extra networks")

    if cmd_opts.tls_keyfile is not None and cmd_opts.tls_keyfile is not None:

        try:
            if not os.path.exists(cmd_opts.tls_keyfile):
                print("Invalid path to TLS keyfile given")
            if not os.path.exists(cmd_opts.tls_certfile):
                print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
        except TypeError:
            cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
            print("TLS setup invalid, running webui without TLS")
        else:
            print("Running with TLS")
        startup_timer.record("TLS")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def setup_middleware(app):
    app.middleware_stack = None # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    app.build_middleware_stack() # rebuild middleware stack on-the-fly


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def wait_on_server(demo=None):
    while 1:
        time.sleep(0.5)
        if shared.state.need_restart:
            shared.state.need_restart = False
            time.sleep(0.5)
            demo.close()
            time.sleep(0.5)
            break


def api_only():
    initialize()

    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)

def webui():
    global cache

    launch_api = cmd_opts.api

    if launch_api:
        models_config_s3uri = os.environ.get('models_config_s3uri', None)
        if models_config_s3uri:
            bucket, key = get_bucket_and_key(models_config_s3uri)
            s3_object = s3_client.get_object(Bucket=bucket, Key=key)
            bytes = s3_object["Body"].read()
            payload = bytes.decode('utf8')
            huggingface_models = json.loads(payload).get('huggingface_models', None)
            s3_models = json.loads(payload).get('s3_models', None)
            http_models = json.loads(payload).get('http_models', None)
        else:
            huggingface_models = os.environ.get('huggingface_models', None)
            s3_models = os.environ.get('s3_models', None)
            http_models = os.environ.get('http_models', None)

        if huggingface_models:
            huggingface_models = json.loads(huggingface_models)
            for huggingface_model in huggingface_models:
                repo_id = huggingface_model['repo_id']
                filename = huggingface_model['filename']
                name = huggingface_model['name']

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=f'/tmp/models/{name}',
                    cache_dir='/tmp/cache/huggingface'
                )

        if s3_models:
            s3_models = json.loads(s3_models)
            for s3_model in s3_models:
                uri = s3_model['uri']
                name = s3_model['name']
                s3_download(uri, f'/tmp/models/{name}')

        if http_models:
            http_models = json.loads(http_models)
            for http_model in http_models:
                uri = http_model['uri']
                filename = http_model['filename']
                name = http_model['name']
                http_download(uri, f'/tmp/models/{name}/{filename}')

    initialize()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        modules.script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = modules.ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = []
        if cmd_opts.gradio_auth:
            gradio_auth_creds += [x.strip() for x in cmd_opts.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip()]
        if cmd_opts.gradio_auth_path:
            with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=server_name,
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )
        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        setup_middleware(app)

        modules.progress.setup_progress_api(app)

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        modules.script_callbacks.app_started_callback(shared.demo, app)
        startup_timer.record("scripts app_started_callback")

        print(f"Startup time: {startup_timer.summary()}.")

        wait_on_server(shared.demo)
        print('Restarting UI...')

        startup_timer.reset()

        sd_samplers.set_samplers()

        modules.script_callbacks.script_unloaded_callback()
        extensions.list_extensions()
        startup_timer.record("list extensions")

        localization.list_localizations(cmd_opts.localizations_dir)

        modelloader.forbid_loaded_nonbuiltin_upscalers()
        modules.scripts.reload_scripts()
        startup_timer.record("load scripts")

        modules.script_callbacks.model_loaded_callback(shared.sd_model)
        startup_timer.record("model loaded callback")

        modelloader.load_upscalers()
        startup_timer.record("load upscalers")

        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

        modules.sd_models.list_models()
        startup_timer.record("list SD models")

        shared.reload_hypernetworks()
        startup_timer.record("reload hypernetworks")

        ui_extra_networks.intialize()
        ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
        ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
        ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

        extra_networks.initialize()
        extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
        startup_timer.record("initialize extra networks")

if cmd_opts.train:
    def upload_s3files(s3uri, file_path_with_pattern):
        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]

        s3_resource = boto3.resource('s3')
        s3_bucket = s3_resource.Bucket(bucket)

        try:
            for file_path in glob.glob(file_path_with_pattern):
                file_name = os.path.basename(file_path)
                __s3file = f'{key}{file_name}'
                print(file_path, __s3file)
                s3_bucket.upload_file(file_path, __s3file)
        except ClientError as e:
            print(e)
            return False
        return True

    def upload_s3folder(s3uri, file_path):
        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]

        s3_resource = boto3.resource('s3')
        s3_bucket = s3_resource.Bucket(bucket)

        try:
            for path, _, files in os.walk(file_path):
                for file in files:
                    dest_path = path.replace(file_path,"")
                    __s3file = f'{key}{dest_path}/{file}'
                    __local_file = os.path.join(path, file)
                    print(__local_file, __s3file)
                    s3_bucket.upload_file(__local_file, __s3file)
        except Exception as e:
            print(e)

    def train():
        initialize()

        train_args = json.loads(cmd_opts.train_args)

        sd_models_s3uri = cmd_opts.sd_models_s3uri
        db_models_s3uri = cmd_opts.db_models_s3uri
        lora_models_s3uri = cmd_opts.lora_models_s3uri

        db_create_new_db_model = train_args['train_dreambooth_settings']['db_create_new_db_model']
        db_use_txt2img = train_args['train_dreambooth_settings']['db_use_txt2img']
        db_train_wizard_person = train_args['train_dreambooth_settings']['db_train_wizard_person']
        db_train_wizard_object = train_args['train_dreambooth_settings']['db_train_wizard_object']
        db_performance_wizard = train_args['train_dreambooth_settings']['db_performance_wizard']

        if db_create_new_db_model:
            db_new_model_name = train_args['train_dreambooth_settings']['db_new_model_name']
            db_new_model_src = train_args['train_dreambooth_settings']['db_new_model_src']
            db_new_model_scheduler = train_args['train_dreambooth_settings']['db_new_model_scheduler']
            db_create_from_hub = train_args['train_dreambooth_settings']['db_create_from_hub']
            db_new_model_url = train_args['train_dreambooth_settings']['db_new_model_url']
            db_new_model_token = train_args['train_dreambooth_settings']['db_new_model_token']
            db_new_model_extract_ema = train_args['train_dreambooth_settings']['db_new_model_extract_ema']
            db_train_unfrozen = train_args['train_dreambooth_settings']['db_train_unfrozen']
            db_512_model = train_args['train_dreambooth_settings']['db_512_model']
            db_save_safetensors = train_args['train_dreambooth_settings']['db_save_safetensors']

            db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = create_model(
                db_new_model_name,
                db_new_model_src,
                db_new_model_scheduler,
                db_create_from_hub,
                db_new_model_url,
                db_new_model_token,
                db_new_model_extract_ema,
                db_train_unfrozen,
                db_512_model
            )
            dreambooth_config_id = cmd_opts.dreambooth_config_id
            try:
                with open(f'/opt/ml/input/data/config/{dreambooth_config_id}.json', 'r') as f:
                    content = f.read()
            except Exception:
                content = None

            if content:
                params_dict = json.loads(content)

                params_dict['db_model_name'] = db_model_name
                params_dict['db_model_path'] = db_model_path
                params_dict['db_revision'] = db_revision
                params_dict['db_epochs'] = db_epochs
                params_dict['db_scheduler'] = db_scheduler
                params_dict['db_src'] = db_src
                params_dict['db_has_ema'] = db_has_ema
                params_dict['db_v2'] = db_v2
                params_dict['db_resolution'] = db_resolution

                if db_train_wizard_person or db_train_wizard_object:
                    db_num_train_epochs, \
                    c1_num_class_images_per, \
                    c2_num_class_images_per, \
                    c3_num_class_images_per, \
                    c4_num_class_images_per = training_wizard(db_train_wizard_person if db_train_wizard_person else db_train_wizard_object)

                    params_dict['db_num_train_epochs'] = db_num_train_epochs
                    params_dict['c1_num_class_images_per'] = c1_num_class_images_per
                    params_dict['c1_num_class_images_per'] = c2_num_class_images_per
                    params_dict['c1_num_class_images_per'] = c3_num_class_images_per
                    params_dict['c1_num_class_images_per'] = c4_num_class_images_per
                if db_performance_wizard:
                    attention, \
                    gradient_checkpointing, \
                    gradient_accumulation_steps, \
                    mixed_precision, \
                    cache_latents, \
                    sample_batch_size, \
                    train_batch_size, \
                    stop_text_encoder, \
                    use_8bit_adam, \
                    use_lora, \
                    use_ema, \
                    save_samples_every, \
                    save_weights_every = performance_wizard()

                    params_dict['attention'] = attention
                    params_dict['gradient_checkpointing'] = gradient_checkpointing
                    params_dict['gradient_accumulation_steps'] = gradient_accumulation_steps
                    params_dict['mixed_precision'] = mixed_precision
                    params_dict['cache_latents'] = cache_latents
                    params_dict['sample_batch_size'] = sample_batch_size
                    params_dict['train_batch_size'] = train_batch_size
                    params_dict['stop_text_encoder'] = stop_text_encoder
                    params_dict['use_8bit_adam'] = use_8bit_adam
                    params_dict['use_lora'] = use_lora
                    params_dict['use_ema'] = use_ema
                    params_dict['save_samples_every'] = save_samples_every 
                    params_dict['params_dict'] = save_weights_every

                db_config = DreamboothConfig(db_model_name)
                concept_keys = ["c1_", "c2_", "c3_", "c4_"]
                concepts_list = []
                # If using a concepts file/string, keep concepts_list empty.
                if params_dict["db_use_concepts"] and params_dict["db_concepts_path"]:
                    concepts_list = []
                    params_dict["concepts_list"] = concepts_list
                else:
                    for concept_key in concept_keys:
                        concept_dict = {}
                        for key, param in params_dict.items():
                            if concept_key in key and param is not None:
                                concept_dict[key.replace(concept_key, "")] = param
                        concept_test = Concept(concept_dict)
                        if concept_test.is_valid:
                            concepts_list.append(concept_test.__dict__)
                    existing_concepts = params_dict["concepts_list"] if "concepts_list" in params_dict else []
                    if len(concepts_list) and not len(existing_concepts):
                        params_dict["concepts_list"] = concepts_list

                db_config.load_params(params_dict)
        else:
            db_model_name = train_args['train_dreambooth_settings']['db_model_name']
            db_config = DreamboothConfig(db_model_name)

        print(vars(db_config))
        start_training_from_config(
            db_config,
            db_use_txt2img,
        )

        cmd_sd_models_path = cmd_opts.ckpt_dir
        sd_models_dir = os.path.join(shared.models_path, "Stable-diffusion")
        if cmd_sd_models_path is not None:
            sd_models_dir = cmd_sd_models_path

        try:
            cmd_dreambooth_models_path = cmd_opts.dreambooth_models_path
        except:
            cmd_dreambooth_models_path = None

        try:
            cmd_lora_models_path = shared.cmd_opts.lora_models_path
        except:
            cmd_lora_models_path = None

        db_model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        db_model_dir = os.path.join(db_model_dir, "dreambooth")

        lora_model_dir = os.path.dirname(cmd_lora_models_path) if cmd_lora_models_path else paths.models_path
        lora_model_dir = os.path.join(lora_model_dir, "lora")

        try:
            print('Uploading SD Models...')
            upload_s3files(
                sd_models_s3uri,
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.yaml')
            )
            if db_save_safetensors:
                upload_s3files(
                    sd_models_s3uri,
                    os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.safetensors')
                )
            else:
                upload_s3files(
                    sd_models_s3uri,
                    os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.ckpt')
                )
            print('Uploading DB Models...')
            upload_s3folder(
                f'{db_models_s3uri}{db_model_name}',
                os.path.join(db_model_dir, db_model_name)
            )
            if db_config.use_lora:
                print('Uploading Lora Models...')
                upload_s3files(
                    lora_models_s3uri,
                    os.path.join(lora_model_dir, f'{db_model_name}_*.pt')
                )
            os.makedirs(os.path.dirname("/opt/ml/model/"), exist_ok=True)
            os.makedirs(os.path.dirname("/opt/ml/model/Stable-diffusion/"), exist_ok=True)
            os.makedirs(os.path.dirname("/opt/ml/model/ControlNet/"), exist_ok=True)
            train_steps=int(db_config.revision)
            model_file_basename = f'{db_model_name}_{train_steps}_lora' if db_config.use_lora else f'{db_model_name}_{train_steps}'
            f1=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.yaml')
            if os.path.exists(f1):
                shutil.copy(f1,"/opt/ml/model/Stable-diffusion/")
            if db_save_safetensors:
                f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.safetensors')
                if os.path.exists(f2):
                    shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
            else:
                f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.ckpt')
                if os.path.exists(f2):
                    shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
        except Exception as e:
            traceback.print_exc()
            print(e)
else:
    def get_bucket_and_key(s3uri):
        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]
        return bucket, key

    def s3_download(s3uri, path):
        global cache

        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]

        s3_bucket = s3_resource.Bucket(bucket)
        objs = list(s3_bucket.objects.filter(Prefix=key))

        if os.path.isfile('cache'):
            cache = json.load(open('cache', 'r'))

        for obj in objs:
            response = s3_client.head_object(
                Bucket = bucket,
                Key =  obj.key
            )
            obj_key = 's3://{0}/{1}'.format(bucket, obj.key)
            if obj_key not in  cache or cache[obj_key] != response['ETag']:
                filename = obj.key[obj.key.rfind('/') + 1 : ]

                s3_client.download_file(bucket, obj.key, os.path.join(path, filename))
                cache[obj_key] = response['ETag']

        json.dump(cache, open('cache', 'w'))

    def http_download(httpuri, path):
        with requests.get(httpuri, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

if __name__ == "__main__":
    if cmd_opts.train:
        train()
    elif cmd_opts.nowebui:
        api_only()
    else:
        webui()
