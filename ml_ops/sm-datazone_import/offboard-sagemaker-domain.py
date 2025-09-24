import argparse
import boto3

"""
The purpose of this script is tear down any resources created/connected during
the SageMaker Hulk - Bring Your Own Domain testing via /byod-script.py
This will walk the user through the SageMaker Domains/Users + DataZone Domains in
their account and detach the SageMaker Resources from their DataZone counterparts.
"""


class SageMakerDomainOffboarder:
    def _offboard_sm_domain(self):
        print("List of SageMaker Domains for your account.")
        print("--------------------------------------------------------------------")
        sm_domain_map = {}  # save name->id map.
        for domain in self.sm_client.list_domains()["Domains"]:
            print(f'Name: {domain["DomainName"]}')
            sm_domain_map[domain["DomainName"]] = domain["DomainId"]
        self.sm_domain_name = input("SageMaker Domain Name to remove tags for: ")
        self.sm_domain_id = sm_domain_map[self.sm_domain_name]
        # Firstly, domain remove tags.
        sm_domain_tags = [
            "AmazonDataZoneDomain",
            "AmazonDataZoneDomainAccount",
            "AmazonDataZoneStage",
        ]
        sm_domain_arn = "arn:aws:sagemaker:{}:{}:domain/{}".format(
            self.region, self.account_id, self.sm_domain_id
        )
        self.sm_client.delete_tags(ResourceArn=sm_domain_arn, TagKeys=sm_domain_tags)
        print(
            "Removing the following tags {} from SageMaker Domain [{}]".format(
                sm_domain_tags, sm_domain_arn
            )
        )

    def _offboard_sm_users(self):
        print("--------------------------------------------------------------------")
        print("SageMaker Users in SageMaker Domain " + self.sm_domain_name)
        print("--------------------------------------------------------------------")
        sm_users = []
        for user in self.sm_client.list_user_profiles(DomainIdEquals=self.sm_domain_id)[
            "UserProfiles"
        ]:
            sm_users.append(user["UserProfileName"])

        for user in sm_users:
            print("User:", user)
        self.sm_user_name = input(
            "Which user to remove tags from the execution role?: "
        )

        # Second, remove role tags.
        sm_user_profile_full = self.sm_client.describe_user_profile(
            DomainId=self.sm_domain_id, UserProfileName=self.sm_user_name
        )
        sm_user_info = {}
        sm_user_info["name"] = self.sm_user_name
        sm_user_info["arn"] = sm_user_profile_full["UserProfileArn"]
        sm_user_info["exec_role_arn"] = sm_user_profile_full["UserSettings"][
            "ExecutionRole"
        ]
        sm_user_info["id"] = sm_user_profile_full[
            "HomeEfsFileSystemUid"
        ]  # e.g. d-7d4uvydb9rcy

        # get role name from arn "arn:aws:iam::047923724610:role/service-role/AmazonSageMaker-ExecutionRole-20241008T155288"
        sm_exec_role_tags = ["AmazonDataZoneDomain", "AmazonDataZoneProject"]
        role_name = sm_user_info["exec_role_arn"][
            sm_user_info["exec_role_arn"].rfind("/") + 1 :
        ]  # rfind searches from back of string
        self.iam_client.untag_role(RoleName=role_name, TagKeys=sm_exec_role_tags)
        print(
            "Removing the following tags {} from SageMaker ExecutionRole [{}]".format(
                sm_exec_role_tags, role_name
            )
        )

    def __init__(self, region, stage, account_id) -> None:
        self.region = region
        self.stage = stage
        self.account_id = account_id
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

        self.sm_client = boto3.client(
            "sagemaker", region_name=region, endpoint_url=sm_endpoint_url
        )
        self.dz_client = boto3.client(
            "datazone", region_name=region, endpoint_url=dz_endpoint_url
        )
        self.byod_client = boto3.client(
            "datazone-byod", region_name=region, endpoint_url=dz_endpoint_url
        )
        self.iam_client = boto3.client("iam", region_name=region)

    def _select_dz_domain(self):
        print("--------------------------------------------------------------------")
        print("List of DataZone Domains.")
        print("--------------------------------------------------------------------")
        dz_domain_map = {}
        for domain in self.dz_client.list_domains()["items"]:
            print(f'Name: {domain["name"]}')
            dz_domain_map[domain["name"]] = domain["id"]

        dz_domain_name = input(
            "DataZone Domain Name to offboard/remove resources for: "
        )
        self.dz_domain_id = dz_domain_map[dz_domain_name]

    def _select_dz_project(self):
        print("--------------------------------------------------------------------")
        print("List of DataZone Projects.")
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

    def _delete_linked_items(self):
        user_profile_arn = "arn:aws:sagemaker:{}:{}:user-profile/{}/{}".format(
            self.region, self.account_id, self.sm_domain_id, self.sm_user_name
        )

        # Detach via BatchDeleteLinkedTypes
        print("--------------------------------------------------------------------")
        print("Detaching SageMaker User Profile")
        self.byod_client.batch_delete_linked_types(
            domainIdentifier=self.dz_domain_id,
            projectIdentifier=self.dz_project_id,
            itemIdentifiers=[user_profile_arn],
        )

        sm_domain_arn = "arn:aws:sagemaker:{}:{}:domain/{}".format(
            self.region, self.account_id, self.sm_domain_id
        )

        # Detach via BatchDeleteLinkedTypes
        print("--------------------------------------------------------------------")
        print("Detaching SageMaker Domain")
        self.byod_client.batch_delete_linked_types(
            domainIdentifier=self.dz_domain_id,
            projectIdentifier=self.dz_project_id,
            itemIdentifiers=[sm_domain_arn],
        )

    def _print_results(self):
        # Verify no linked entities
        print("--------------------------------------------------------------------")
        print("Listing linked entities")
        items = self.byod_client.list_linked_types(
            domainIdentifier=self.dz_domain_id, projectIdentifier=self.dz_project_id
        )["items"]
        print(f"Found {len(items)} linked items.")
        for item in items:
            print(item)

    def offboard(self):
        print("Note: Double click output to quickly select it for copy-paste.")
        self._offboard_sm_domain()
        self._offboard_sm_users()
        self._select_dz_domain()
        self._select_dz_project()
        self._delete_linked_items()
        self._print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    account_help = ""
    parser.add_argument(
        "--region", type=str, required=False, default="us-west-2", help=""
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
        help="Account to cleanup DataZone resources for. Ensure the current session has the correct permissions for SageMaker and DataZone actions.",
    )
    args = parser.parse_args()

    region = args.region
    stage = args.stage
    account_id = args.account_id

    print("--------------------------------------------------------------------")
    offboarder = SageMakerDomainOffboarder(region, stage, account_id)
    offboarder.offboard()
    print("--------------------------------------------------------------------")
