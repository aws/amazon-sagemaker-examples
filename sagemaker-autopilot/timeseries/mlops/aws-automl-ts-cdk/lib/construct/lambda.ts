import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as sfn_tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as cdk from 'aws-cdk-lib';

export interface LambdaConstructProps {
    taskName: string;
    lambdaName: string;
    lambdaCodePath: string;
    timeout: cdk.Duration;
    resourceBucket: s3.Bucket;
    environment?: {
        [key:string]: string;
    }
}

export class LambdaConstruct extends Construct {
    public readonly role: iam.Role;
    public readonly lambda: lambda.Function;
    public readonly task: sfn.TaskStateBase;
    
    constructor(scope: Construct, id: string, props: LambdaConstructProps) {
        super(scope, id);
        
        const resourceBucketArn = props.resourceBucket.bucketArn;
        
        // Define the policy statement allows Full Access to specified S3 bucket
        const s3BucketFullAccessPolicy = new iam.PolicyStatement({
          actions: ['s3:*'],
          resources: [resourceBucketArn, `${resourceBucketArn}/*`],
        });
        
        // IAM Role
        this.role = new iam.Role(this, `${props.lambdaName}-Lambda-Role`, {
           assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
           roleName: `${props.lambdaName}-Lambda-Role`,
           managedPolicies: [
                {managedPolicyArn: 'arn:aws:iam::aws:policy/CloudWatchFullAccess'},
                {managedPolicyArn: 'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'}
            ],
            inlinePolicies: {
                's3BucketReadOnly': new iam.PolicyDocument({
                    statements: [s3BucketFullAccessPolicy]
                })
            }
        });
        
        // Lambda Function
        this.lambda = new lambda.Function(this, `${props.lambdaName}-Lambda-Function`, {
            code: lambda.Code.fromAsset(props.lambdaCodePath),
            handler: 'index.handler',
            functionName: props.lambdaName,
            runtime: lambda.Runtime.PYTHON_3_11,
            timeout: props.timeout,
            role: this.role,
            environment: props.environment
        });
        
        // Define StepFunction task for this Lambda
        this.task = new sfn_tasks.LambdaInvoke(this, `${props.taskName} Lambda Task`, {
            lambdaFunction: this.lambda,
            integrationPattern: sfn.IntegrationPattern.REQUEST_RESPONSE,
            resultPath: sfn.JsonPath.stringAt('$'),
            outputPath: sfn.JsonPath.stringAt('$.Payload')
        });
        
        this.task.addRetry({
            backoffRate: 1.0,
            errors: ['ResourcePending'],
            interval: cdk.Duration.seconds(30),
            maxAttempts: 600
        });
    }
}