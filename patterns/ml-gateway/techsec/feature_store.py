import sys
import time
from sagemaker.feature_store.feature_definition import FeatureDefinition
from sagemaker.feature_store.feature_group import FeatureGroup


class StatusIndicator:
    def __init__(self):
        self.previous_status = None
        self.need_newline = False

    def update(self, status):
        if self.previous_status != status:
            if self.need_newline:
                sys.stdout.write("\n")
            sys.stdout.write(status + " ")
            self.need_newline = True
            self.previous_status = status
        else:
            sys.stdout.write(".")
            self.need_newline = True
        sys.stdout.flush()

    def end(self):
        if self.need_newline:
            sys.stdout.write("\n")


def get_feature_definitions(df, feature_group):
    """
    Get datatypes from pandas DataFrame and map them
    to Feature Store datatypes.

    :param df: pandas.DataFrame
    :param  feature_group: FeatureGroup
    :return: list
    """
    # Dtype int_, int8, int16, int32, int64, uint8, uint16, uint32
    # and uint64 are mapped to Integral feature type.

    # Dtype float_, float16, float32 and float64
    # are mapped to Fractional feature type.

    # string dtype is mapped to String feature type.

    # Our schema of our data that we expect
    # _after_ SageMaker Processing
    feature_definitions = []
    for column in df.columns:
        feature_type = feature_group._DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.get(
            str(df[column].dtype), None
        )
        feature_definitions.append(
            FeatureDefinition(column, feature_type)
        )  # you can alternatively define your own schema
    return feature_definitions


def wait_for_feature_group_creation_complete(feature_group):
    """
    Wait for a FeatureGroup to finish creating.

    :param feature_group: FeatureGroup
    :return: None
    """
    status_indicator = StatusIndicator()
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        status_indicator.update(status)
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    status_indicator.end()
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")


def ingest_df_to_feature_group(df, feature_group_name, feature_store_client):
    """
    Take a pandas DataFrame and put it in a FeatureGroup.

    :param df: pandas.DataFrame
    :param feature_group_name: str
    :param feature_store_client: boto3.client('sagemaker-featurestore-runtime')
    :return: None
    """
    success, fail = 0, 0
    for row_num, row_series in df.astype(str).iterrows():
        record = []
        for key, value in row_series.to_dict().items():
            record.append({"FeatureName": key, "ValueAsString": str(value)})
        response = feature_store_client.put_record(
            FeatureGroupName=feature_group_name, Record=record
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            success += 1
        else:
            fail += 1
    print(f"Success = {success}")
    print(f"Fail = {fail}")


def get_datatypes():
    """
    Get pandas DataFrame datatypes.

    :return: tuple(dict, dict)
    """
    claims_dtypes = {
        "policy_id": int,
        "incident_severity": int,
        "num_vehicles_involved": int,
        "num_injuries": int,
        "num_witnesses": int,
        "police_report_available": int,
        "injury_claim": float,
        "vehicle_claim": float,
        "total_claim_amount": float,
        "incident_month": int,
        "incident_day": int,
        "incident_dow": int,
        "incident_hour": int,
        "fraud": int,
        "driver_relationship_self": int,
        "driver_relationship_na": int,
        "driver_relationship_spouse": int,
        "driver_relationship_child": int,
        "driver_relationship_other": int,
        "incident_type_collision": int,
        "incident_type_breakin": int,
        "incident_type_theft": int,
        "collision_type_front": int,
        "collision_type_rear": int,
        "collision_type_side": int,
        "collision_type_na": int,
        "authorities_contacted_police": int,
        "authorities_contacted_none": int,
        "authorities_contacted_fire": int,
        "authorities_contacted_ambulance": int,
        "event_time": float,
    }

    customers_dtypes = {
        "policy_id": int,
        "customer_age": int,
        "customer_education": int,
        "months_as_customer": int,
        "policy_deductable": int,
        "policy_annual_premium": int,
        "policy_liability": int,
        "auto_year": int,
        "num_claims_past_year": int,
        "num_insurers_past_5_years": int,
        "customer_gender_male": int,
        "customer_gender_female": int,
        "policy_state_ca": int,
        "policy_state_wa": int,
        "policy_state_az": int,
        "policy_state_or": int,
        "policy_state_nv": int,
        "policy_state_id": int,
        "event_time": float,
    }

    return claims_dtypes, customers_dtypes


def create_feature_group(
    feature_group_name,
    feature_group_description,
    df,
    id_name,
    event_time_name,
    offline_feature_group_bucket,
    sagemaker_session,
    role,
):
    """
    Create a new FeatureGroup.

    :param feature_group_name: str
    :param feature_group_description: str
    :param df: pandas.DataFrame
    :param id_name: str
    :param event_time_name: str
    :param offline_feature_group_bucket: str
    :param sagemaker_session: sagemaker.Session()
    :param role: str
    :return: tuple(FeatureGroup, bool)
    """
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    feature_definitions = get_feature_definitions(df, feature_group)
    feature_group.feature_definitions = feature_definitions
    feature_group_already_exists = False
    try:
        print(f"Trying to create feature group {feature_group_description} \n")
        feature_group.create(
            description=feature_group_description,
            record_identifier_name=id_name,
            event_time_feature_name=event_time_name,
            role_arn=role,
            s3_uri=offline_feature_group_bucket,
            enable_online_store=True,
        )
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        code = e.response.get("Error").get("Code")
        if code == "ResourceInUse":
            print(f"Using existing feature group: {feature_group_name}")
            feature_group_already_exists = True
        else:
            raise (e)
    return feature_group, feature_group_already_exists
