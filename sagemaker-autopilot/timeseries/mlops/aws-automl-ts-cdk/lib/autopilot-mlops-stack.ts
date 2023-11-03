import { Duration, Stack, StackProps, RemovalPolicy } from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as fs from 'fs';

import { Construct } from "constructs";
import { StateMachine } from "./construct/state-machine";

export class AutopilotMlopsStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    
    const configRaw = fs.readFileSync('cdk-config/cdk-config.json', 'utf8');
    const config = JSON.parse(configRaw);

    const resourceBucket = new s3.Bucket(
      this,
      `${config.baseConstructName}-Bucket`,
      {
        bucketName: `${config.baseResourceBucket}-${
          Stack.of(this).account
        }`,
        versioned: false,
        autoDeleteObjects: true,
        removalPolicy: RemovalPolicy.DESTROY,
      },
    );

    const stateMachine = new StateMachine(
      this,
      `${config.baseConstructName}-StateMachine`,
      {
        resourceBucket: resourceBucket,
      },
    );
  }
}
