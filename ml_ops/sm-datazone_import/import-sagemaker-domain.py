import argparse
import boto3
import botocore
import datetime
from dateutil.tz import tzlocal
from botocore.exceptions import ClientError

"""
The purpose of this script is run the SageMaker Hulk - Bring Your Own Domain
feature end to end. This will walk the user through the SageMaker Domains/Users
in their account and pick a DataZone Project,User, + Domain to bring the SageMaker
Domain into. See https://quip-amazon.com/5MrGAr72EMXP/Hulk-BYOD-Script-Draft for
more information.
"""


def assumed_role_session(account_id: str, role_name: str, base_session: botocore.session.Session = None):
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    base_session = base_session or boto3.session.Session()._session
    fetcher = botocore.credentials.AssumeRoleCredentialFetcher(
        client_creator=base_session.create_client,
        source_credentials=base_session.get_credentials(),
        role_arn=role_arn,
        extra_args={},
    )
    creds = botocore.credentials.DeferredRefreshableCredentials(
        method="assume-role",
        refresh_using=fetcher.fetch_credentials,
        time_fetcher=lambda: datetime.datetime.now(tzlocal()),
    )
    botocore_session = botocore.session.Session()
    botocore_session._credentials = creds
    return boto3.Session(botocore_session=botocore_session)


class SageMakerDomainImporter:
    def __init__(self, region, stage, account_id) -> None:
        self.region = region
        self.stage = stage

        # Setup client.
        sm_endpoint_url = "https://api.sagemaker." + region + ".amazonaws.com"  # prod
        dz_endpoint_url = "https://datazone." + region + ".api.aws"  # prod
        if stage in ["alpha", "beta", "gamma"]:
            sm_endpoint_url = (
                "https://api.sagemaker."
                + stage
                + "."
                + region
                + ".ml-platform.aws.a2z.com"
            )
            dz_endpoint_url = "https://iceland-" + stage + "." + region + ".api.aws"

        self.child_account_id = account_id
        self.child_assume_role_name = "Admin"  # Change this if need be for default creds.
        self.parent_account_id = None  # Will be set during the Cross-Account scenario.
        self.parent_assume_role_name = None
        self.cross_account_enabled = False

        session = assumed_role_session(self.child_account_id, self.child_assume_role_name)
        self.sm_client = session.client(
            "sagemaker", region_name=region, endpoint_url=sm_endpoint_url
        )
        self.dz_client = session.client(
            "datazone", region_name=region, endpoint_url=dz_endpoint_url
        )
        self.byod_client = session.client(
            "datazone-byod", region_name=region, endpoint_url=dz_endpoint_url
        )
        self.iam_client = session.client("iam", region_name=region)
        self.sts_client = session.client("sts", region_name=region)

    def __get_child_dz_clients(self):
        session = assumed_role_session(self.child_account_id, self.child_assume_role_name)
        dz_endpoint_url = "https://datazone." + region + ".api.aws"  # prod
        self.dz_client = session.client(
            "datazone", region_name=region, endpoint_url=dz_endpoint_url
        )
        self.byod_client = session.client(
            "datazone-byod", region_name=region, endpoint_url=dz_endpoint_url
        )

    def __get_parent_dz_clients(self):
        if self.cross_account_enabled:
            session = assumed_role_session(self.parent_account_id, self.parent_assume_role_name)
            dz_endpoint_url = "https://datazone." + region + ".api.aws"  # prod
            self.dz_client = session.client(
                "datazone", region_name=region, endpoint_url=dz_endpoint_url
            )
            self.byod_client = session.client(
                "datazone-byod", region_name=region, endpoint_url=dz_endpoint_url
            )


    def _choose_sm_domain(self):
        """
        [1] First, identify the SageMaker Domain needed to be imported by noting its DomainId.
        """
        print(
            "List of SageMaker Domains for your account. Pick the domain needed to be imported."
        )
        print("--------------------------------------------------------------------")
        sm_domain_map = {}  # save name->id map.
        for domain in self.sm_client.list_domains()["Domains"]:
            print(f'Name: {domain["DomainName"]}')
            sm_domain_map[domain["DomainName"]] = domain["DomainId"]

        self.sm_domain_name = input("SageMaker Domain Name: ")
        self.sm_domain_id = sm_domain_map[self.sm_domain_name]
        print(
            "Chosen SageMaker Domain [{}] with Domain Id [{}]".format(
                self.sm_domain_name, self.sm_domain_id
            )
        )
        sm_domain = self.sm_client.describe_domain(DomainId=self.sm_domain_id)
        self.auth_mode = sm_domain["AuthMode"]
        self.default_execution_role = sm_domain["DefaultUserSettings"]["ExecutionRole"]
        return self.sm_domain_id

    def _choose_dz_domain(self):
        """
        [2] Select the DataZone Domain and Project.
        Next identify the DataZone Domain and Project to put the imported SageMaker Domain into
        by noting the domain’s id and the project’s id.
        """

        print("--------------------------------------------------------------------")
        print(
            "List of DataZone Domains. Pick Domain to select projects to import SageMaker Domain into."
        )
        print("--------------------------------------------------------------------")
        dz_domain_map = {}
        self.__get_parent_dz_clients() # Toggle to parent clients for list domains/projects.
        for domain in self.dz_client.list_domains()["items"]:
            print(f'Name: {domain["name"]}')
            dz_domain_map[domain["name"]] = domain["id"]

        self.dz_domain_name = input("DataZone Domain Name: ")
        self.dz_domain_id = dz_domain_map[self.dz_domain_name]
        print(
            "Chosen DataZone Domain [{}] with Domain Id [{}]".format(
                self.dz_domain_name, self.dz_domain_id
            )
        )
        return self.dz_domain_id

    def _choose_dz_project(self):
        print("--------------------------------------------------------------------")
        print(
            "List of DataZone Projects. Pick Project to import SageMaker Domain into."
        )
        print("--------------------------------------------------------------------")
        dz_project_map = {}
        for project in self.dz_client.list_projects(domainIdentifier=self.dz_domain_id)[
            "items"
        ]:
            print(f'Name: {project["name"]}')
            dz_project_map[project["name"]] = project["id"]

        self.dz_project_name = input("DataZone Project Name: ")
        self.dz_project_id = dz_project_map[self.dz_project_name]
        print(
            "Chosen DataZone Project [{}] with Project Id [{}]".format(
                self.dz_project_name, self.dz_project_id
            )
        )
        return self.dz_project_id

    def _tag_sm_domain(self):
        """
        [3.5 Tagging] Before getting started on byod-e2e, ensure that CX has the SageMaker domain and ExecutionRole's tagged accordingly.
            1. Tag the SageMaker domain by admin (DZ DomainId and tag the stage, and domainAccountId)
            2. Tag the execution role with DZ domainId and projectId
        """

        domain_tag = {"Key": "AmazonDataZoneDomain", "Value": self.dz_domain_id}
        account_tag = {"Key": "AmazonDataZoneDomainAccount", "Value": self.child_account_id}
        stage_tag = {"Key": "AmazonDataZoneStage", "Value": self.stage}
        sm_domain_tags = [domain_tag, account_tag, stage_tag]
        sm_domain_arn = "arn:aws:sagemaker:{}:{}:domain/{}".format(
            self.region, self.child_account_id, self.sm_domain_id
        )

        self.sm_client.add_tags(ResourceArn=sm_domain_arn, Tags=sm_domain_tags)
        print("--------------------------------------------------------------------")
        print(
            "Adding the following tags to SageMaker Domain [{}]".format(
                self.sm_domain_name
            )
        )
        for t in sm_domain_tags:
            print(t)

    def _map_users(self):
        """
        [3] Get SageMaker Users
        Next create a mapping between SageMaker User Profiles and DataZone users.
              - Aggregate Users in the SageMaker Domain
              - Collect SageMaker user profile data needed for mapping.
              - User profile Name and tag to map the DZ user via Tag APIs.
        """

        sm_users = []
        for user in self.sm_client.list_user_profiles(DomainIdEquals=self.sm_domain_id)[
            "UserProfiles"
        ]:
            sm_users.append(user["UserProfileName"])

        print("--------------------------------------------------------------------")
        print("SageMaker Users in SageMaker Domain " + self.sm_domain_name)
        print("--------------------------------------------------------------------")
        for user in sm_users:
            print("User:", user)
        sm_user_name = input("Which user should be used?: ")

        sm_user_profile_full = self.sm_client.describe_user_profile(
            DomainId=self.sm_domain_id, UserProfileName=sm_user_name
        )
        self.sm_user_info = {}
        self.sm_user_info["name"] = sm_user_name
        self.sm_user_info["arn"] = sm_user_profile_full["UserProfileArn"]
        exec_role = None
        if "UserSettings" in sm_user_profile_full:
            user_settings = sm_user_profile_full["UserSettings"]
            if "ExecutionRole" in user_settings:
                exec_role = user_settings["ExecutionRole"]

        if exec_role is None:
            print(
                f"User {sm_user_name} has no execution role set, using default from domain."
            )
            exec_role = self.default_execution_role

        self.sm_user_info["exec_role_arn"] = exec_role

        self.sm_user_info["id"] = sm_user_profile_full[
            "HomeEfsFileSystemUid"
        ]  # e.g. d-7d4uvydb9rcy

        sm_exec_role_tags = []
        domain_tag = {"Key": "AmazonDataZoneDomain", "Value": self.dz_domain_id}
        project_tag = {"Key": "AmazonDataZoneProject", "Value": self.dz_project_id}
        # We need this to display all necessary information in UI, as the env_id is plucked
        # from the tag of the ExecutionRole.
        env_tag = {"Key": "AmazonDataZoneEnvironment", "Value": self.env_id}
        sm_exec_role_tags = [domain_tag, project_tag, env_tag]

        # get role name from arn "arn:aws:iam::047923724610:role/service-role/AmazonSageMaker-ExecutionRole-20241008T155288"
        role_name = self.sm_user_info["exec_role_arn"][
            self.sm_user_info["exec_role_arn"].rfind("/") + 1 :
        ]  # rfind searches from back of string
        self.iam_client.tag_role(RoleName=role_name, Tags=sm_exec_role_tags)
        print(
            "Adding the following tags to SageMaker ExecutionRole [{}]".format(
                role_name
            )
        )
        for t in sm_exec_role_tags:
            print(t)

        print("--------------------------------------------------------------------")
        print("Getting IAM DataZone UserProfiles in account...")

        user_types = [
            "SSO_USER",
            # remove DATAZONE_USER as these are redundant with others.
            "DATAZONE_SSO_USER",
            "DATAZONE_IAM_USER",
        ]
        all_dz_users = []  # [( payload, type ), ... ]
        for user_type in user_types:
            search_response = self.dz_client.search_user_profiles(
                domainIdentifier=self.dz_domain_id, userType=user_type
            )["items"]
            dz_user_map = {"Items": search_response, "Type": user_type}
            all_dz_users.append(dz_user_map)

        # For all user types, iterate through all users.
        for dz_user_map in all_dz_users:
            dz_users, user_type = dz_user_map["Items"], dz_user_map["Type"]
            for dz_user in dz_users:
                user_name = "None"
                if user_type == "DATAZONE_IAM_USER":
                    user_name = dz_user["details"]["iam"]["arn"]
                if user_type == "SSO_USER" or user_type == "DATAZONE_SSO_USER":
                    user_name = dz_user["details"]["sso"]["username"]

                dz_user_id = dz_user["id"]
                print(f"UserId: {dz_user_id}\tUserType: {user_type}\tUser: {user_name}")

        self.dz_users_id_list = []
        while True:
            dz_uzer = input("Enter a DataZone UserId to map to ('done' to finish): ")
            if dz_uzer == "done":
                break
            self.dz_users_id_list.append(dz_uzer)

    def _link_multiple_users_and_projects(self):
        """
        Add the option for the user to attach Users in subset B to Project B.
        """
        print("--------------------------------------------------------------------")
        decision = input(
            "Would you like to onboard an additional subset of user profiles to another project? "
            "(This would require you to have another project created. In this new project, you will create"
            " a new environment if not already created, as well) [y/n]: "
        )
        if decision == "y":
            self._choose_dz_project()
            self._configure_blueprint()
            self._configure_environment()
            self._tag_sm_domain()
            self._map_users()
            self._associate_fed_role()
            self._add_environment_action()
            self._link_domain()
            self._link_users()
            self._debug_print_results()
            self._get_env_link()
            self._link_multiple_users_and_projects()

    def _configure_blueprint(self):
        # [4] Create environment profile + environment and use new API BatchPutLinkedTypes to connect DataZone and SageMaker entities.

        print("--------------------------------------------------------------------")
        print("Listing Blueprints in Customer Account")
        # Toggle to Child Account here - Make the call from original child account.
        self.__get_child_dz_clients()
        blueprints = self.dz_client.list_environment_blueprints(
            domainIdentifier=self.dz_domain_id, managed=True
        )

        managed_key = "CustomAwsService"
        self.managed_blueprint_id = None
        for bp in blueprints["items"]:
            print("Id: ", bp["id"], " Name: ", bp["name"])
            if bp["name"] == managed_key:
                self.managed_blueprint_id = bp["id"]
        print(
            "Choosing managed blueprint {} with id [{}]".format(
                managed_key, self.managed_blueprint_id
            )
        )

        # if already enabled returns success
        self.dz_client.put_environment_blueprint_configuration(
            domainIdentifier=self.dz_domain_id,
            environmentBlueprintIdentifier=self.managed_blueprint_id,
            enabledRegions=[self.region],
        )

        print("--------------------------------------------------------------------")

        return self.managed_blueprint_id

    def _configure_environment(self):
        decision_env = input(
            "Do you need to create a new DataZone environment? [y/n]: "
        )
        if decision_env == "y":
            self.env_name = input(
                "Create DataZone Environment in project [{}] with name: ".format(
                    self.dz_project_name
                )
            )
            self.env_id = None
            try:
                create_env_response = self.dz_client.create_environment(  # caller from child account
                    domainIdentifier=self.dz_domain_id,
                    name=self.env_name,
                    environmentBlueprintIdentifier=self.managed_blueprint_id,
                    projectIdentifier=self.dz_project_id,
                    environmentAccountIdentifier=self.child_account_id,
                    environmentAccountRegion=self.region,
                )
                self.env_id = create_env_response["id"]
                print(
                    "Created Environment with EnvironmentId [{}] using managed blueprint".format(
                        self.env_id
                    )
                )
            except ClientError as e:
                if "already exists within this project" in str(e):
                    envs = self.dz_client.list_environments(
                        domainIdentifier=self.dz_domain_id,
                        projectIdentifier=self.dz_project_id,
                    )["items"]
                    for e in envs:
                        if e["name"] == self.env_name:
                            self.env_id = e["id"]
                            print(
                                "Environment [{}] already created with id [{}], skipping create call ...".format(
                                    self.env_name, self.env_id
                                )
                            )
        else:
            print(
                "--------------------------------------------------------------------"
            )
            print("List of DataZone Environments.")
            print(
                "--------------------------------------------------------------------"
            )
            dz_env_map = {}
            for env in self.dz_client.list_environments(
                domainIdentifier=self.dz_domain_id, projectIdentifier=self.dz_project_id
            )["items"]:
                print(f'Name: {env["name"]}')
                dz_env_map[env["name"]] = env["id"]
            self.env_name = input("Please provide the name of DataZone environment: ")
            self.env_id = dz_env_map[self.env_name]

        return self.env_id

    def _add_environment_action(self):
        items = self.dz_client.list_environment_actions(
            domainIdentifier=self.dz_domain_id, environmentIdentifier=self.env_id
        )["items"]
        sm_env_action = None
        for item in items:
            if "sageMaker" in item["parameters"]:
                sm_env_action = item

        if sm_env_action is None:
            self.byod_client.create_environment_action(
                domainIdentifier=self.dz_domain_id,
                environmentIdentifier=self.env_id,
                name="SageMaker Environment Action Link",
                description="Link from DataZone Data Portal to SageMaker Studio",
                parameters={"sageMaker": {}},
            )

    def _associate_fed_role(self):
        # Associate fed role
        print("--------------------------------------------------------------------")
        self.federation_role = input(
            "Federation Role Arn to federate into sagemaker studio from datazone portal: "
        )
        print(
            "Associating Environment Role using Federation Role [{}] ...".format(
                self.federation_role
            )
        )
        try:
            self.dz_client.associate_environment_role(
                domainIdentifier=self.dz_domain_id,
                environmentIdentifier=self.env_id,
                environmentRoleArn=self.federation_role,
            )
            print(
                "Associating Environment Role using Federation Role [{}] COMPLETE".format(
                    self.federation_role
                )
            )
        except ClientError as e:
            if "Environment has a role configured already" in str(e) \
                    or "(ConflictException) when calling the AssociateEnvironmentRole operation" in str(e):
                print(
                    "Environment has a role configured already. Skipping role association ..."
                )
            else:
                print(f"Caught error: {repr(e)}")

    def _cross_account_action(self):
        # Decision to link domain from X-account. Required to assumeRole from child-account containing SageMaker domain.

        print("--------------------------------------------------------------------")
        decision = input("Need to import SageMaker Profiles into a different account (Cross Account)? [y/n]: ")
        if decision == "y":
            print("--------------------------------------------------------------------")
            print("Provide the Cross Account Id and Role Name for the current session to assume. \n"
                  "This would be the role in the account that originally created and houses the DataZone domain,\n"
                  "aka the parent account in the Cross Account scenario. See ReadMe for how to configure permission "
                  "for assumeRole.")
            self.parent_account_id = input("Cross Account Id: ")
            self.parent_assume_role_name = input("Cross Account Role Name to Assume: ")
            self.__get_parent_dz_clients()
            # Toggle this on such that we can switch between clients
            self.cross_account_enabled = True

    def _link_domain(self):
        # attach SAGEMAKER_DOMAIN
        linked_domain_items = [
            {
                "itemIdentifier": f"arn:aws:sagemaker:{self.region}:{self.child_account_id}:domain/{self.sm_domain_id}",
                "itemType": "SAGEMAKER_DOMAIN",
                "configuration": {"AuthMode": self.auth_mode},
                "connectedEntities": [
                    {
                        "connectedEntityIdentifier": self.env_id,
                        "connectedEntityType": "ENVIRONMENT",
                        "connectedEntityConnectionType": "CONSUMED_BY",
                    }
                ],
            }
        ]

        # Instead of update_environment_configuration, we use the new BatchPutLinkedType API to update.
        print("--------------------------------------------------------------------")
        print(
            "Calling BatchPutLinkedType for Environment with Id [{}]".format(
                self.env_id
            )
        )
        print("--------------------------------------------------------------------")

        print(f"Linking SageMaker Domain using project id [{self.dz_project_id}]")
        self.__get_parent_dz_clients()
        link_domain_response = self.byod_client.batch_put_linked_types(
            domainIdentifier=self.dz_domain_id,
            projectIdentifier=self.dz_project_id,
            environmentIdentifier=self.env_id,
            items=linked_domain_items,
        )
        print(link_domain_response)
        print("Linked SageMaker Domain.")

    def _link_users(self):
        # attach SAGEMAKER_USER_PROFILE
        linkedUserItems = []
        for dz_user_id in self.dz_users_id_list:
            linkedUserItem = {
                "itemIdentifier": f'arn:aws:sagemaker:{self.region}:{self.child_account_id}:user-profile/{self.sm_domain_id}/{self.sm_user_info["name"]}',
                "itemType": "SAGEMAKER_USER_PROFILE",
                "name": self.sm_user_info["name"],
                "authorizedPrincipals": [
                    {
                        "principalIdentifier": dz_user_id,
                        "principalType": "DATAZONE_USER_PROFILE",
                    }
                ],
                "connectedEntities": [
                    {
                        "connectedEntityConnectionType": "BELONGS_TO",
                        "connectedEntityIdentifier": f"arn:aws:sagemaker:{self.region}:{self.child_account_id}:domain/{self.sm_domain_id}",
                        "connectedEntityType": "SAGEMAKER_DOMAIN",
                    }
                ],
            }
            linkedUserItems.append(linkedUserItem)

        print("--------------------------------------------------------------------")
        print(
            f"Linking SageMaker User Profiles using project id [{self.dz_project_id}]"
        )
        link_users_response = self.byod_client.batch_put_linked_types(
            domainIdentifier=self.dz_domain_id,
            projectIdentifier=self.dz_project_id,
            environmentIdentifier=self.env_id,
            items=linkedUserItems,
        )
        print(link_users_response)
        print("Linked SageMaker User Profiles.")
        print("--------------------------------------------------------------------")

    def _debug_print_results(self):
        print(
            f"Listing linked items for domain {self.dz_domain_id}, project {self.dz_project_id}, and environment {self.env_id}."
        )
        list_result = self.byod_client.list_linked_types(
            domainIdentifier=self.dz_domain_id,
            projectIdentifier=self.dz_project_id,
            environmentIdentifier=self.env_id,
        )
        print(f"Found {len(list_result)} linked items.")
        for item in list_result["items"]:
            print(item)

    def _get_env_link(self):
        print("--------------------------------------------------------------------")
        print("Getting Action Link for SageMaker Studio")

        try:
            link_response = self.byod_client.get_environment_action_link(
                domainIdentifier=self.dz_domain_id,
                environmentIdentifier=self.env_id,
                consoleType="SageMakerConsole",
            )
            link = link_response["actionLink"]
            print(link)
        except botocore.exceptions.ClientError as error:
            print(
                "Environment action link could not be generated - this is most likely due to the current principal is not a user of the DataZone project."
            )
            print(error)

    def import_interactive(self):
        print(
            "Note: double click the name to quickly select it (to then copy-paste it)."
        )
        self._choose_sm_domain() #child account
        self._cross_account_action() #parent vvvv
        self._choose_dz_domain()
        self._choose_dz_project()
        self._configure_blueprint()
        self._configure_environment()
        self._tag_sm_domain()
        self._map_users()
        self._associate_fed_role()
        self._add_environment_action()
        self._link_domain()
        self._link_users()
        self._debug_print_results()
        self._get_env_link()
        self._link_multiple_users_and_projects()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        default="us-east-2",
        help="Region with DataZone and SageMaker resources",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=False,
        default="prod",
        help="Stage to test e2e BYOD. This impacts the endpoint targeted.",
    )
    parser.add_argument(
        "--account-id",
        type=str,
        required=True,
        help="Account to create new DataZone environment in. Ensure the current session"
        + "has the correct permissions for SageMaker and DataZone actions.",
    )
    args = parser.parse_args()

    region = args.region
    stage = args.stage
    account_id = args.account_id

    print("--------------------------------------------------------------------")
    importer = SageMakerDomainImporter(region, stage, account_id)
    importer.import_interactive()
    print("--------------------------------------------------------------------")
