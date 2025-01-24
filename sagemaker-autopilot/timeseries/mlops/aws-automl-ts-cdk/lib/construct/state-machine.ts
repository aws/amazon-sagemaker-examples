import { Construct } from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sfn from "aws-cdk-lib/aws-stepfunctions";
import * as cdk from 'aws-cdk-lib';
import * as fs from 'fs';

import { GlueConstruct } from "./glue";
import { TriggerConstruct } from './trigger';
import { LambdaConstruct } from './lambda';
import { SageMakerConstruct } from './sagemaker';

export interface StateMachineProps {
  resourceBucket: s3.Bucket;
}

export class StateMachine extends Construct {
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string, props: StateMachineProps) {
    super(scope, id);

    const resourceBucket = props.resourceBucket;
    
    const configRaw = fs.readFileSync('cdk-config/cdk-config.json', 'utf8');
    const config = JSON.parse(configRaw);
    const baseConstructName = config.baseConstructName
    
    // Define the policy statement allows Full Access to specified S3 bucket
    const s3BucketFullAccessPolicy = new iam.PolicyStatement({
      actions: ['s3:*'],
      resources: [resourceBucket.bucketArn, `${resourceBucket.bucketArn}/*`],
    });

    // IAM Role to pass to SageMaker Autopilot
    const sagemakerExecutionRole = new iam.Role(
      this,
      `${baseConstructName}-SageMaker-Execution-Role`,
      {
        assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
        roleName: `${baseConstructName}-Sagemaker-Role`,
        managedPolicies: [
          {managedPolicyArn: "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"}
        ],
        inlinePolicies: {
          's3BucketFullAccess': new iam.PolicyDocument({
            statements: [s3BucketFullAccessPolicy]
          })
        },
      },
    );
    
    
    // IAM Role for State Machine
    const stateMachineExecutionRole = new iam.Role(this, `${baseConstructName}-StateMachine-Execution-Role`, {
        assumedBy: new iam.ServicePrincipal('states.amazonaws.com'),
        roleName: `${baseConstructName}-Execution-Role`,
        managedPolicies: [
            {managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaRole'},
            {managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole'}
        ]
    })

    
    // Create Config Check Lambda
    const checkConfigFilesInS3 = new LambdaConstruct(this, `${baseConstructName}-Check-Config`, {
       taskName: `Check if Config files exist in S3`,
       lambdaName: `${baseConstructName}-Check-Config`,
       lambdaCodePath: 'lambda/check-config',
       timeout: cdk.Duration.seconds(30),
       resourceBucket: resourceBucket,
       environment: {
           RESOURCE_BUCKET: props.resourceBucket.bucketName
       }
    });
    
    
    // Preprocessing
    const preprocess = new GlueConstruct(this, `${baseConstructName}-Data-Preprocess-Glue`, {
      taskName: `Glue Data Preprocessing`,
      glueName: `${baseConstructName}-Data-Preprocessing`,
      pythonFilePath: "glue/preprocess.py",
      resourceBucket: resourceBucket,
      defaultArguments: {
        "--bucket": props.resourceBucket.bucketName,
        "--prefix": "raw/",
      },
    });
    
    // Create Autopilot TS Training Job
    const createAutopilotTrainingJob = new LambdaConstruct(this, `${baseConstructName}-Create-Autopilot-Job`, {
       taskName: `Create Autopilot Job`,
       lambdaName: `${baseConstructName}-Create-AutoML-Job`,
       lambdaCodePath: 'lambda/create-autopilot-job',
       timeout: cdk.Duration.seconds(30),
       resourceBucket: resourceBucket,
       environment: {
           SAGEMAKER_ROLE_ARN: sagemakerExecutionRole.roleArn,
           RESOURCE_BUCKET: props.resourceBucket.bucketName
       }
    });
    
    // Check Autopilot Job Status
    const checkAutopilotJobStatus = new LambdaConstruct(this, `${baseConstructName}-Autopilot-Job-Status-Check`, {
        taskName: 'Autopilot Job Status Check',
        lambdaName: `${baseConstructName}-Check-AutoML-Job`,
        lambdaCodePath: 'lambda/check-autopilot-job',
        timeout: cdk.Duration.seconds(30),
        resourceBucket: resourceBucket,
        environment: {
            SAGEMAKER_ROLE_ARN: sagemakerExecutionRole.roleArn
        }
    });
    
    
    // Waiting 5m before checking Autopilot Job Status
    const wait5minAfterTraining = new sfn.Wait(this, `Wait 5 Min Training`, {
        time: sfn.WaitTime.duration(cdk.Duration.minutes(5))
    });
    
    // Waiting 5m before checking Autopilot Job Status
    const wait5minAfterJob = new sfn.Wait(this, `Wait 5 Min Job`, {
        time: sfn.WaitTime.duration(cdk.Duration.minutes(5))
    });
    
    // Create a model from the Best trained model from AutoML
    const bestModel = new SageMakerConstruct(this, `${baseConstructName}-Best-Model`, {
      taskName: 'Create Model from Best Candidate',
      resourceBucket: resourceBucket,
      sagemakerRoleArn: sagemakerExecutionRole.roleArn
    });
    
    // Create Autopilot TS Training Job
    const createTransformJob = new LambdaConstruct(this, `${baseConstructName}-Create-Transform-Job`, {
       taskName: 'Create Transform Job',
       lambdaName: `${baseConstructName}-Create-Transform-Job`,
       lambdaCodePath: 'lambda/create-transform-job',
       timeout: cdk.Duration.seconds(30),
       resourceBucket: resourceBucket,
       environment: {
           RESOURCE_BUCKET: props.resourceBucket.bucketName
       }
    });
    
    // Check Transform Job Status
    const checkTransformationJobStatus = new LambdaConstruct(this, `${baseConstructName}-Transformation-Job-Status-Check`, {
        taskName: 'Transform Job Check',
        lambdaName: `${baseConstructName}-Check-Transform-Job`,
        lambdaCodePath: 'lambda/check-transform-job',
        timeout: cdk.Duration.seconds(30),
        resourceBucket: resourceBucket,
        environment: {
            SAGEMAKER_ROLE_ARN: sagemakerExecutionRole.roleArn
        }
    });
    
    // Finish State Machine if job failed
    const jobFailed = new sfn.Fail(this, `Autopilot MLOps Pipeline Failed`, {
      cause: 'Autopilot MLOps Pipeline Job Failed',
      error: 'Autopilot Train Job returned FAILED',
    });
    
    // Final Success State
    const success = new sfn.Succeed(this, 'We did it!');
    
    const configFileChoice = new sfn.Choice(this, 'Does Config Files Exist on S3?');
    const passCheckConfig = new sfn.Pass(this, 'Successfull pass config check');
    
    // State Machine Definition
    const definition = checkConfigFilesInS3.task
                        .next(configFileChoice
                            .when(sfn.Condition.stringEquals('$.config_status', 'FAILED'), jobFailed)
                            .otherwise(
                              preprocess.task
                              .next(createAutopilotTrainingJob.task)
                              .next(wait5minAfterTraining)
                              .next(checkAutopilotJobStatus.task)
                              .next(new sfn.Choice(this, 'AutoML Job Complete?')
                                  .when(sfn.Condition.stringEquals('$.AutoMLJobStatus', 'InProgress'), wait5minAfterTraining)
                                  .when(sfn.Condition.stringEquals('$.AutoMLJobStatus', 'Completed'), bestModel.createModelTask
                                    .next(createTransformJob.task)
                                    .next(wait5minAfterJob)
                                    .next(checkTransformationJobStatus.task)
                                    .next(new sfn.Choice(this, 'Transformation Job Complete?')
                                      .when(sfn.Condition.stringEquals('$.TransformJobStatus', 'InProgress'), wait5minAfterJob)
                                      .when(sfn.Condition.stringEquals('$.TransformJobStatus', 'Completed'), success)
                                      .otherwise(jobFailed))
                                  )
                                  .otherwise(jobFailed))));
    
    // Creating a State Machine
    const stateMachine = new sfn.StateMachine(this, `${baseConstructName}-State-Machine`, {
        definition: definition,
        role: stateMachineExecutionRole,
        stateMachineName: `${baseConstructName}`
    });
    
    // Train trigger, from S3 Object Create to Lambda, which then initiates State Machine
    const trainTrigger = new TriggerConstruct(this, `${baseConstructName}-Train-Trigger`, {
        stateMachine: stateMachine,
        resourceBucket: resourceBucket,
        s3Prefix: 'raw/'
    });
  }
}
