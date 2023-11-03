import {Construct} from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as sfn_tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';

import * as glue from "@aws-cdk/aws-glue-alpha";

export interface GlueConstructProps {
    taskName: string,
    glueName: string,
    pythonFilePath: string,
    resourceBucket: s3.Bucket,
    defaultArguments?: {
        [key:string]: string;
    },
    arguments?: {
        [key:string]: any;
    }
}

export class GlueConstruct extends Construct {
    public readonly role: iam.Role;
    public readonly task: sfn_tasks.GlueStartJobRun;
    
    constructor(scope: Construct, id: string, props: GlueConstructProps) {
        super(scope, id);
        
        const resourceBucketArn = props.resourceBucket.bucketArn;
        
        // Define the policy statement allows Full Access to specified S3 bucket
        const s3BucketFullAccessPolicy = new iam.PolicyStatement({
          actions: ['s3:*'],
          resources: [resourceBucketArn, `${resourceBucketArn}/*`],
        });
        
        // IAM Role
        this.role = new iam.Role(this, `${props.glueName}-Role`, {
            assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
            roleName: `${props.glueName}-Role`,
            managedPolicies: [
                {managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole'},
            ],
            inlinePolicies: {
                's3BucketReadOnly': new iam.PolicyDocument({
                    statements: [s3BucketFullAccessPolicy]
                })
            }
        });
        
        // Glue Python Job
        const pythonJob = new glue.Job(this, `${props.glueName}-Python-Job`, {
            executable: glue.JobExecutable.pythonShell({
               glueVersion: glue.GlueVersion.V3_0,
               pythonVersion: glue.PythonVersion.THREE_NINE,
               script: glue.Code.fromAsset(props.pythonFilePath)
            }),
            role: this.role,
            jobName: props.glueName,
            defaultArguments: props.defaultArguments
        });
        
        // StepFunction Task
        this.task = new sfn_tasks.GlueStartJobRun(this, `${props.taskName}`, {
            glueJobName: pythonJob.jobName,
            integrationPattern: sfn.IntegrationPattern.RUN_JOB,
            resultPath: sfn.JsonPath.stringAt('$.result'),
            arguments: sfn.TaskInput.fromObject(props.arguments!)
        });
    }
    
}
