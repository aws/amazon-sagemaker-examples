import {Construct} from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as sfn_tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as fs from 'fs';

export interface SageMakerConstructProps {
    taskName: string,
    resourceBucket: s3.Bucket,
    sagemakerRoleArn: string,
    defaultArguments?: {
        [key:string]: string;
    },
    arguments?: {
        [key:string]: any;
    }
}

export class SageMakerConstruct extends Construct {
    public readonly createModelTask: sfn_tasks.SageMakerCreateModel;
    public readonly createTransformJob: sfn_tasks.SageMakerCreateTransformJob;
    
    constructor(scope: Construct, id: string, props: SageMakerConstructProps) {
        super(scope, id);
        
        const resourceBucketName = props.resourceBucket.bucketName;
        const configRaw = fs.readFileSync('cdk-config/cdk-config.json', 'utf8');
        const config = JSON.parse(configRaw);
        const baseConstructName = config.baseConstructName
        
        // Take an existing role for SageMaker
        const smRole = iam.Role.fromRoleArn(this, `${baseConstructName}-SM-Role`, props.sagemakerRoleArn);
        
        // Create a Step Functions task for creating AI model from the Best model trained by SageMaker Autopilot
        this.createModelTask = new sfn_tasks.SageMakerCreateModel(this, `${props.taskName} Task`, {
          modelName: sfn.JsonPath.stringAt('$.BestCandidate.CandidateName'),
          primaryContainer: new sfn_tasks.ContainerDefinition({
            image: sfn_tasks.DockerImage.fromJsonExpression(sfn.JsonPath.stringAt('$.BestCandidate.InferenceContainer.Image')),
            mode: sfn_tasks.Mode.SINGLE_MODEL,
            modelS3Location: sfn_tasks.S3Location.fromJsonExpression('$.BestCandidate.InferenceContainer.ModelDataUrl'),
          }),
          role: smRole,
          resultPath: '$.createdModel' 
        });
        
    }
    
}