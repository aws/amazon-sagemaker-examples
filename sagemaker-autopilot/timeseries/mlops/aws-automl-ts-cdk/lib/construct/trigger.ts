import { Construct } from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as sfn from "aws-cdk-lib/aws-stepfunctions";
import * as lambda_event_sources from "aws-cdk-lib/aws-lambda-event-sources";
import * as fs from 'fs';

export interface TriggerConstructProps {
  stateMachine: sfn.StateMachine;
  resourceBucket: s3.Bucket;
  s3Prefix: string;
}

export class TriggerConstruct extends Construct {
  public readonly role: iam.Role;
  public readonly lambda: lambda.Function;
  public readonly task: sfn.TaskStateBase;

  constructor(scope: Construct, id: string, props: TriggerConstructProps) {
    super(scope, id);

    const resourceBucketArn = props.resourceBucket.bucketArn;
    
    const configRaw = fs.readFileSync('cdk-config/cdk-config.json', 'utf8');
    const config = JSON.parse(configRaw);
    const baseConstructName = config.baseConstructName
    
    // Define the policy statement allows Read Access to specified S3 bucket
    const s3BucketReadAccessPolicy = new iam.PolicyStatement({
      actions: [
        's3:GetObject',
        's3:ListBucket',
      ],
      resources: [resourceBucketArn, `${resourceBucketArn}/*`],
    });
    
    // Define a policy statement that allows starting executions of the specific Step Function
    const startSfnExecutionPolicy = new iam.PolicyStatement({
      actions: ['states:StartExecution'],
      resources: [props.stateMachine.stateMachineArn],
    });

    // IAM Role
    this.role = new iam.Role(this, `${baseConstructName}-Train-Trigger-Role`, {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      roleName: `${baseConstructName}-Train-Trigger-Role`,
      managedPolicies: [
        {managedPolicyArn: "arn:aws:iam::aws:policy/CloudWatchFullAccess" },
        {managedPolicyArn: "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"},
        {managedPolicyArn: "arn:aws:iam::aws:policy/service-role/AWSLambdaRole"},
      ],
      inlinePolicies: {
        's3BucketReadOnly': new iam.PolicyDocument({
            statements: [s3BucketReadAccessPolicy]
        }),
        'sfnStartExecution': new iam.PolicyDocument({
            statements: [startSfnExecutionPolicy]
        })
      }
    });
    
    // Define Lambda Function for Trigger
    this.lambda = new lambda.Function(this, `${baseConstructName}-Upload-Lambda`, {
        runtime: lambda.Runtime.PYTHON_3_11,
        role: this.role,
        functionName: `${baseConstructName}-Upload-Lambda`,
        code: lambda.Code.fromAsset('lambda/trigger'),
        handler: 'index.handler',
        environment: {
            STEP_FUNCTIONS_ARN: props.stateMachine.stateMachineArn
        }
    });
    
    // Add trigger from S3 to Lambda on Object Create
    this.lambda.addEventSource(new lambda_event_sources.S3EventSource(
        props.resourceBucket, {
            events: [
                s3.EventType.OBJECT_CREATED
            ],
            filters: [
                {
                    prefix: props.s3Prefix
                }
            ]
        }
    ))
  }
}
