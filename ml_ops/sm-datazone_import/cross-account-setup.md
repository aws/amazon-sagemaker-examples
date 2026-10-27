### Cross Account Setup
In this scenario, we will use two accounts, a parent account hosting the DataZone domain, and a child account that contains
a SageMaker Domain nd UserProfile that we would like to link, and import into the DataZone domain. Here AccountA is the parent
account and AccountB is the child account.

1. **[In the Parent Account]** Create a DataZone domain (make sure you are NOT using the Unified UI). It should say “Create a DataZone Domain”.
   1. If we will be using IAM Users, we need to make sure that we have enabled the IAM Identity Center for our created domain. 
   2. We can easily do this in the console by clicking "Enable". This is required such that when importing user-profiles from our Child Account, we do not run into the following error
   ```
   botocore.errorfactory.ResourceNotFoundException: An error occurred (ResourceNotFoundException) when 
   calling the SearchUserProfiles operation: IAM Identity Center application not found for domain 
   'dzd_4dfv1ls60cg1ev', please ensure IAM Identity Center has been enabled
   ```


2. **[In the Parent Account]** Create an association to the Child Account. 

   1. Click in to Domain - request Association by providing the Child Account number.
   2. This will create a `AWSRAMPermissionDataZoneDefault` policy to allow access from the Child Account. 
   3. In the Child Account, Accept the resource Share in the DataZone UI (Unified UI)


3. **[In the Parent Account]** Create A Project, select the parent domain as the “DomainUnit”.


4. **[In the Parent Account]** Add in the Child Account user that will access this project. 
    1. In the **User Management** tab, add the child user’s role (as an IAM user or SSO user) that requires access, from the Child Account. Choose the **AssociatedAccount** option.
    2. In the **Projects** tab, add the child user as a **Project Member**. They should be available in the drop-down menu. Set the respective permissions to your liking. 

> NOTE :: If this step (i) is not done, the Cross Account user will see the following error in their projects tab when clicking into the Associated Domain  

```
Not a DataZone user
You cannot view or create a project because you have not been added 
as a Amazon DataZone user. Please contact your domain admin to add 
your IAM role: arn:aws:iam::211125770549:role/Admin as a DataZone user.
```

Now, when you login to the child account, you will be able to see the created project. Make sure that you
enabled `CustomAWSBluePrint` in child account, as this will be required when creating the datazone environment when the script runs.

5. **[In the Child Account]** Create a SageMaker Domain. Ensure that you do this from the Amazon SageMaker AI console, not the Amazon SageMaker platform console (this is the unified experience, separate from this current workflow). Add users profiles to the domain


6. **[In the Child Account]** Setup a federation role that will have permission to federate into our parent account’s Datazone portal. See `/resources` for examples of trust and permission policies. 

### Running the script
For linking the SageMaker Domain + UserProfile using HULK BYOD Flow:

Make sure that the current account you are using grants access to the Child Account to sts:AssumeRole. 
For Example, AccountB is the one that houses the SageMaker Domain and UserProfile that you would like to import into the DataZone parent account, AccountA. 

* We need to be sure to add the following JSON to the TrustPolicy of the Admin (or whatever role in the parent (secondary) account you’d like to assume, with DataZone permissions to call batch-put-linked-types and link the SageMaker Domain and UserProfiles). 
* Also, make sure to add the User that the current session is using.

In the parent account (Account A), under the rol we want to assume - we should add the following.
This will allow our child account to assume parent account role during our current session, and link the SageMaker Domain + UserProfile that we interact with while running the script.
The role needs DataZone permissions.

```
  {
      "Sid": "",
      "Effect": "Allow",
      "Principal": {
          "AWS": "arn:aws:sts::<Child_Account_B>:assumed-role/<Role_Name>/<User>-Isengard"
      },
      "Action": "sts:AssumeRole"
  }
```

Also, if you have pasted another account credentials into terminal, like a dev-account, make sure that dev account is able to assumeRole into the Child Account.
For example, in the Admin role of Child Account, I have pasted the following for myself. This ensures your session can toggle between
the child and parent account clients.
```
{
   "Sid": "",
   "Effect": "Allow",
   "Principal": {
       "AWS": "arn:aws:sts::047923724610:assumed-role/Admin/svia-Isengard"
   },
   "Action": "sts:AssumeRole"
}
```

* Nothing changes in regard to the regular call to the batch-put-linked-type APIs. We will assume the parent account (AccountA) credentials. 
* From the Parent Account, the script will call batch-put-linked type using the SageMaker ARN and SageMaker UserProfile ARN from the Child Account (AccountB).
* Federation link will then work for Child Account and Parent Account from DZ portal → environment view.
