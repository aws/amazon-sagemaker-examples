import boto3  # your python gate way to all aws services
import pprint # print readable dictionary
import json
import time

pp = pprint.PrettyPrinter(indent=1)
iam = boto3.client('iam')

# get the ARN of the user
caller_arn = boto3.client('sts').get_caller_identity()['Arn']


def create_execution_role(role_name="basic-role"):
    """Create an service role to procure services on your behalf

    
    Args:
        role_name (str): name of the role
    
    Return:
        dict
    """    
    # if the role already exists, delete it
    # Note: you need to make sure the role is not
    # used in production, because the code below
    # will delete the role and create a new one
    
    def find_role(role_res, role_name):
        for r in role_res['Roles']:
            if r['RoleName'] == role_name:
                return True
        return False

    def delete_role(role_res, role_name):
        if find_role(role_res, role_name):
            role = boto3.resource('iam').Role(role_name)
            for p in role.attached_policies.all():
                role.detach_policy(PolicyArn=p.arn)

            iam.delete_role(RoleName=role.name)
            return

    role_res = iam.list_roles(MaxItems=10)
    delete_role(role_res, role_name)
    
    # pagination 
    while 'Marker' in role_res:
        role_res = iam.list_roles(MaxItems=10, Marker=role_res['Marker'])
        delete_role(role_res, role_name)
    
    # Trust policy document
    trust_relation_policy_doc = {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "AWS": caller_arn, # Allow caller to take this role
            "Service": [
              "sagemaker.amazonaws.com" # Allow SageMaker to take the role
            ],
          },
          "Action": "sts:AssumeRole",
        }
      ]
    }
    
    
    res = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_relation_policy_doc)
    )
    return res


def attach_permission(role_name, policy_name, policy_doc):
    """Attach a basic permission policy to the role"""

    # Create the policy
    # If the policy with policy name $policy_name already exists,
    # then we need to delete it first
    
    # Note: you need to make sure that you do not have a policy 
    # with $policy_name in production, because we will delete it
    # and create a new one with the policy document given by 
    # $policy_doc
    
    policy = None
    for p in iam.list_policies()['Policies']:
        if p['PolicyName']==policy_name:
            # Before we delete the policy, we need to detach it
            # from all IAM entities 
            policy = boto3.resource('iam').Policy(p['Arn'])
            
            # 1. detach from all groups
            for grp in policy.attached_groups.all():
                policy.detach_group(GroupName=grp.name)
                
            # 2. detach from all users
            for usr in policy.attached_users.all():
                policy.detach_user(UserName=usr.name)
            
            # 3. detach from all roles
            for rol in policy.attached_roles.all():
                policy.detach_role(RoleName=rol.name)
                
            break
    
    if policy is not None:
        iam.delete_policy(PolicyArn=policy.arn)   
    
    # create a new policy
    policy = iam.create_policy(
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy_doc))['Policy']
    
    # attach the policy to the role
    res = iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy['Arn']
        )
    return res
