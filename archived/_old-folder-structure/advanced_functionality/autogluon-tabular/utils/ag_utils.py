import os
import tarfile

import boto3
import ipywidgets as ipyw
import pandas as pd
from IPython.display import IFrame, display
from ipywidgets import HTML, GridspecLayout, Image, Layout, VBox, interact, widgets


def search_training_jobs(job_tag_name, job_tag_value):
    search_params = {
        "Resource": "TrainingJob",
        "SearchExpression": {
            "Filters": [
                {"Name": f"Tags.{job_tag_name}", "Operator": "Equals", "Value": job_tag_value},
                {"Name": "TrainingJobStatus", "Operator": "Equals", "Value": "Completed"},
            ]
        },
    }

    smclient = boto3.client(service_name="sagemaker")
    results = smclient.search(**search_params)

    return results


def get_training_job_list(tag_key, tag_value):

    training_jobs = search_training_jobs(tag_key, tag_value)

    ag_jobs = {}

    for training_job in training_jobs["Results"]:
        training_job_desc = training_job["TrainingJob"]
        training_job_name = training_job_desc["TrainingJobName"]
        ag_jobs[training_job_name] = training_job_desc

    return ag_jobs


def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    return bucket, key


def get_html_text(text, color):
    html = HTML(
        """<style>
        body
        {
        background-color:"""
        + color
        + """
        }

        </style>"""
        + text
    )

    html = HTML("<b><font size=4 color='" + color + "'>" + text + "</font></b>")

    return html


def show_in_html(text, color):
    text_out = get_html_text(text, color)
    display(text_out, metadata=dict(isolated=True))


def show_metrics(job_desc):
    metrics = job_desc["FinalMetricDataList"]

    for metric in metrics:
        metric_name = metric["MetricName"]
        metric_value = metric["Value"]
        show_in_html(f"<b>{metric_name}</b> : {metric_value:.2f}", "#ff9900")


def show_leaderboard(Job_Name):

    leaderboard_fname = f"./tmp/{Job_Name}/leaderboard.csv"

    if os.path.exists(leaderboard_fname):
        df = pd.read_csv(leaderboard_fname)
        df.set_index(df.columns[0], inplace=True)
        show_in_html("Leaderboard", "#ff9900")
        display(df)


def show_modelsummary(Job_Name):

    modelsummary_fname = f"./tmp/{Job_Name}/SummaryOfModels.html"

    if os.path.exists(modelsummary_fname):
        show_in_html("Summary of models", "#ff9900")
        display(IFrame(src=modelsummary_fname, width=700, height=700))


def show_leaderboard_modelsummary(Job_Name):
    tab_names = ["Leaderboard", "Summary of models"]

    leaderboard_fname = f"./tmp/{Job_Name}/leaderboard.csv"
    modelsummary_fname = f"./tmp/{Job_Name}/SummaryOfModels.html"

    if os.path.exists(leaderboard_fname):
        lb_df = pd.read_csv(leaderboard_fname)
        lb_df.set_index(lb_df.columns[0], inplace=True)
        df_html = lb_df.style.set_table_attributes('class="table"').render()
        html_widget = widgets.HTML(df_html)

    if os.path.exists(modelsummary_fname):
        html_file = open(modelsummary_fname, "r")
        html_content = html_file.read()
        print(html_content)
        modelsummary = widgets.VBox([IFrame(src=modelsummary_fname, width=700, height=700)])

    children = [html_widget, modelsummary]

    tab = ipyw.Tab()
    tab.children = children
    for i in range(len(tab_names)):
        tab.set_title(i, tab_names[i])

    display(tab)


def show_classification_report_confusion_matrix(Job_Name):

    classificationreport_fname = f"./tmp/{Job_Name}/classification_report.csv"
    featureimportance_fname = f"./tmp/{Job_Name}/feature_importance.csv"
    confusionmatrix_fname = f"./tmp/{Job_Name}/confusion_matrix.png"
    roc_auc_curve_fname = f"./tmp/{Job_Name}/roc_auc_curve.png"

    has_data = False

    # Classification report
    if os.path.exists(classificationreport_fname):
        df = pd.read_csv(classificationreport_fname)
        df = df.rename(columns={"Unnamed: 0": "Label"})
        df.set_index("Label", inplace=True)
        df = df.applymap("{0:.2f}".format)
        df_html = df.style.set_table_attributes('class="table"').render()
        cr_widget_html = HTML(df_html)
        has_data = True
    else:
        cr_widget_html = VBox([])

    # Feature importance
    if os.path.exists(featureimportance_fname):
        df = pd.read_csv(featureimportance_fname)
        df.set_index(df.columns[0], inplace=True)
        df = df.applymap("{0:.3f}".format)
        df_html = df.style.set_table_attributes('class="table"').render()
        fi_widget_html = HTML(df_html)
        has_data = True
    else:
        fi_widget_html = VBox([])

    cr_title_html = get_html_text("Classification report", "#ff9900")
    fi_title_html = get_html_text("Feature importance", "#ff9900")
    widget_tables = VBox([cr_title_html, cr_widget_html, fi_title_html, fi_widget_html])

    # Confusion matrix
    if os.path.exists(confusionmatrix_fname):
        img_file = open(confusionmatrix_fname, "rb")
        image = img_file.read()
        widget_cm_img = Image(value=image, format="png")
        has_data = True
    else:
        widget_cm_img = VBox([])

    # ROC curve
    if os.path.exists(roc_auc_curve_fname):
        img_file = open(roc_auc_curve_fname, "rb")
        image = img_file.read()
        widget_roc_img = Image(value=image, format="png")
        has_data = True
    else:
        widget_roc_img = VBox([])

    cm_title_html = get_html_text("Confusion matrix", "#ff9900")
    roc_title_html = get_html_text("ROC Curve", "#ff9900")
    widget_imgs = VBox(
        [cm_title_html, widget_cm_img, roc_title_html, widget_roc_img],
        layout=Layout(margin="0 0 0 10px"),
    )

    if has_data:
        show_in_html("Model analysis on test dataset", "#000099")

        grid = GridspecLayout(1, 2)
        grid[0, 0] = widget_tables
        grid[0, 1] = widget_imgs
        display(grid)


def show_ensemble_model(Job_Name):

    ensemblemodel_fname = f"./tmp/{Job_Name}/ensemble-model.png"

    if os.path.exists(ensemblemodel_fname):
        show_in_html("Ensemble model architecture", "#ff9900")
        img_file = open(ensemblemodel_fname, "rb")
        image = img_file.read()
        widget_img = Image(value=image, format="png")

    display(widget_img)


def launch_viewer(tag_key="AlgorithmName", tag_value="AutoGluon-Tabular", is_debug=False):
    global ag_jobs

    # To disable ipywidget output scrollable
    style = """
        <style>
            .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
        </style>
        """
    display(HTML(style))

    show_in_html("AutoGluon Model Performance Viewer", "#000099")

    ag_jobs = get_training_job_list(tag_key, tag_value)
    job_names = ["-- Select job --"] + list(ag_jobs.keys())

    def on_change(Job_Name):

        if "Select job" in Job_Name:
            show_in_html("Please choose one training job from the list", "#0000ff")
        else:
            job_desc = ag_jobs[Job_Name]

            show_in_html(f"<b>Training Job</b> : {Job_Name}", "#ff9900")

            show_in_html("<b>Training dataset</b> ", "#000099")
            for channel in job_desc["InputDataConfig"]:
                channel_name = channel["ChannelName"]
                channel_source = channel["DataSource"]["S3DataSource"]["S3Uri"]
                print(f"{channel_name}: {channel_source}")

            show_model_evaluation(Job_Name, job_desc, is_debug)

    interact(on_change, Job_Name=job_names)


def show_model_evaluation(Job_Name, job_desc, is_debug):
    global ag_jobs

    if download_model_output(Job_Name, job_desc, is_debug):
        #         show_leaderboard_modelsummary(Job_Name)
        show_leaderboard(Job_Name)
        show_modelsummary(Job_Name)

        show_classification_report_confusion_matrix(Job_Name)

        show_ensemble_model(Job_Name)


def download_model_output(Job_name, job_desc, is_debug):

    # download the s3 file into local temp directory and extract it
    s3 = boto3.client("s3")

    job_output_s3_uri = job_desc["ModelArtifacts"]["S3ModelArtifacts"]

    tmp_dir = f"./tmp/{Job_name}"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    def is_output_file_exists(s3_bucket, s3_prefix):
        try:
            results = s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
            return True
        except:
            return False

    s3_bucket, s3_key = split_s3_path(job_output_s3_uri)
    s3_key_output = s3_key.replace("model.tar.gz", "output.tar.gz")

    if is_output_file_exists(s3_bucket, s3_key_output):
        s3_key = s3_key_output
    else:
        return False

    local_tar_fname = os.path.join(tmp_dir, "output.tar.gz")

    s3.download_file(s3_bucket, s3_key, local_tar_fname)

    tar = tarfile.open(local_tar_fname)
    if is_debug is True:
        show_in_html("Files found in output.tar.gz", "#000099")
        for member in tar.getmembers():
            fname = member.name
            print(fname)

    tar.extractall(path=tmp_dir)

    return True
