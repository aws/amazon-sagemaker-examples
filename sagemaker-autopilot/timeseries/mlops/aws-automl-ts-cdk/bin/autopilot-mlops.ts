#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AutopilotMlopsStack } from '../lib/autopilot-mlops-stack';
import * as fs from 'fs';

const configRaw = fs.readFileSync('cdk-config/cdk-config.json', 'utf8');
const config = JSON.parse(configRaw);

const app = new cdk.App();
new AutopilotMlopsStack(app, `${config.baseConstructName}`);
