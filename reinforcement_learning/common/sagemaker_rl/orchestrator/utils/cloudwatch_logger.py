import time
import json


class CloudWatchLogger():

    def __init__(self, cw_client, region_name):
        self.region_name = region_name
        self.cw_client = cw_client
    
    def get_cloudwatch_dashboard_details(self, experiment_id):
        # update for non-commercial region
        cw_dashboard_url = f"https://{self.region_name}.console.aws.amazon.com/cloudwatch/home?region={self.region_name}#dashboards:name={experiment_id};start=PT1H"
        text = f"You can monitor your Training/Hosting evaluation metrics on this [CloudWatch Dashboard]({cw_dashboard_url})"
        text += "\n\n(Note: This would need Trained/Hosted Models to be evaluated in order to publish Evaluation Scores)"
        return text
    
    def publish_latest_hosting_information(
            self,
            experiment_id,
            latest_hosted_model_id,
            latest_hosted_model_score
            ):
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "latest_hosted_model_id_continuous",
                    "Timestamp": time.time(),
                    "Value": int(latest_hosted_model_id.split('-')[-1])
                }
            ]
        )
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "latest_hosted_model_score_continuous",
                    "Timestamp": time.time(),
                    "Value": float(latest_hosted_model_score)
                }
            ]
        )
    
    def publish_latest_training_information(
            self,
            experiment_id,
            latest_trained_model_id,
            latest_trained_model_score
            ):
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "latest_trained_model_id_continuous",
                    "Timestamp": time.time(),
                    "Value": int(latest_trained_model_id.split('-')[-1])
                }
            ]
        )
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "latest_trained_model_score_continuous",
                    "Timestamp": time.time(),
                    "Value": float(latest_trained_model_score)
                }
            ]
        )
    
    def publish_newly_trained_model_eval_information(
        self,
        experiment_id,
        new_trained_model_id,
        new_trained_model_score
    ):
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "newly_trained_model_id",
                    "Timestamp": time.time(),
                    "Value": int(new_trained_model_id.split('-')[-1])
                }
            ]
        )
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "newly_trained_model_score",
                    "Timestamp": time.time(),
                    "Value": float(new_trained_model_score)
                }
            ]
        )
    
    def publish_rewards_for_simulation(
            self,
            experiment_id,
            reported_rewards_sum
            ):
        self.cw_client.put_metric_data(
            Namespace=experiment_id,
            MetricData=[
                {
                    "MetricName": "reported_rewards_score",
                    "Timestamp": time.time(),
                    "Value": float(reported_rewards_sum)
                }
            ]
        )

    def create_cloudwatch_dashboard_from_experiment_id(
            self,
            experiment_id
            ):
        cw_json = self.get_cloudwatch_dashboard_json_for_experiment_id(
            experiment_id,
            self.region_name
            )
        self.cw_client.put_dashboard(
            DashboardName=experiment_id,
            DashboardBody=cw_json
        )

    def get_cloudwatch_dashboard_json_for_experiment_id(
            self,
            experiment_id,
            region_name
            ):
        dashboard_json = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 9,
                    "height": 3,
                    "properties": {
                        "metrics": [
                            [
                                experiment_id,
                                "latest_hosted_model_id_continuous",
                                {
                                    "label": "(ModelId suffix part only)"
                                }
                            ]
                        ],
                        "view": "singleValue",
                        "region": region_name,
                        "title": "Currently Hosted Model Id",
                        "period": 60,
                        "stat": "Maximum"
                    }
                },
                {
                    "type": "metric",
                    "x": 9,
                    "y": 0,
                    "width": 9,
                    "height": 3,
                    "properties": {
                        "metrics": [
                            [
                                experiment_id,
                                "latest_hosted_model_score_continuous",
                                {"label": "EvalScore" }
                            ]
                        ],
                        "view": "singleValue",
                        "region": region_name,
                        "title": "Currently Hosted Model Eval Score (On latest data)",
                        "period": 60,
                        "stat": "Minimum"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 3,
                    "width": 9,
                    "height": 3,
                    "properties": {
                        "metrics": [
                            [
                                experiment_id,
                                "latest_trained_model_id_continuous",
                                { "label": "(ModelId suffix only)" }
                            ]
                        ],
                        "view": "singleValue",
                        "region": region_name,
                        "title": "Latest Trained Model Id",
                        "stat": "Maximum",
                        "period": 60,
                        "setPeriodToTimeRange": False,
                        "stacked": True
                    }
                },
                {
                    "type": "metric",
                    "x": 9,
                    "y": 3,
                    "width": 9,
                    "height": 3,
                    "properties": {
                        "metrics": [
                            [
                                experiment_id,
                                "latest_trained_model_score_continuous",
                                { "label": "EvalScore" }
                            ]
                        ],
                        "view": "singleValue",
                        "region": region_name,
                        "title": "Latest Trained Model Eval Score",
                        "period": 60,
                        "stat": "Maximum"
                    }
                },
                {
                    "type": "metric",
                    "x": 9,
                    "y": 6,
                    "width": 9,
                    "height": 9,
                    "properties": {
                        "metrics": [
                            [ 
                                experiment_id,
                                "newly_trained_model_score",
                                {"label": "EvalScore" }    
                            ]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": region_name,
                        "stat": "Maximum",
                        "period": 60,
                        "title": "New Model Eval Score Over Time",
                        "yAxis": {
                            "left": {
                                "min": 0,
                                "max": 1
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 9,
                    "height": 9,
                    "properties": {
                        "metrics": [
                            [
                                experiment_id,
                                "reported_rewards_score",
                                {"label": "Rewards" }
                            ]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": region_name,
                        "stat": "Maximum",
                        "period": 60,
                        "title": "Experiment's Reported Rewards",
                        "yAxis": {
                            "left": {
                                "min": 0,
                                "max": 1
                            }
                        },
                        "liveData": True,
                        "legend": {
                            "position": "bottom"
                        }
                    }
                }
            ]
        }
        return json.dumps(dashboard_json)
