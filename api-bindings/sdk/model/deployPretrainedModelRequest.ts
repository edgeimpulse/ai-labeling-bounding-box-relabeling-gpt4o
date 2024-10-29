/**
 * Edge Impulse API
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * The version of the OpenAPI document: 1.0.0
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { DeployPretrainedModelRequestModelInfo } from './deployPretrainedModelRequestModelInfo';
import { DeploymentTargetEngine } from './deploymentTargetEngine';

export class DeployPretrainedModelRequest {
    /**
    * A base64 encoded pretrained model
    */
    'modelFileBase64': string;
    'modelFileType': DeployPretrainedModelRequestModelFileTypeEnum;
    /**
    * The name of the built target. You can find this by listing all deployment targets through `listDeploymentTargetsForProject` (via `GET /v1/api/{projectId}/deployment/targets`) and see the `format` type.
    */
    'deploymentType': string;
    'engine'?: DeploymentTargetEngine;
    'modelInfo': DeployPretrainedModelRequestModelInfo;
    /**
    * A base64 encoded .npy file containing the features from your validation set (optional for onnx and saved_model) - used to quantize your model.
    */
    'representativeFeaturesBase64'?: string;
    'deployModelType'?: DeployPretrainedModelRequestDeployModelTypeEnum;
    /**
    * Optional, use a specific converter (only for ONNX models).
    */
    'useConverter'?: DeployPretrainedModelRequestUseConverterEnum;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "modelFileBase64",
            "baseName": "modelFileBase64",
            "type": "string"
        },
        {
            "name": "modelFileType",
            "baseName": "modelFileType",
            "type": "DeployPretrainedModelRequestModelFileTypeEnum"
        },
        {
            "name": "deploymentType",
            "baseName": "deploymentType",
            "type": "string"
        },
        {
            "name": "engine",
            "baseName": "engine",
            "type": "DeploymentTargetEngine"
        },
        {
            "name": "modelInfo",
            "baseName": "modelInfo",
            "type": "DeployPretrainedModelRequestModelInfo"
        },
        {
            "name": "representativeFeaturesBase64",
            "baseName": "representativeFeaturesBase64",
            "type": "string"
        },
        {
            "name": "deployModelType",
            "baseName": "deployModelType",
            "type": "DeployPretrainedModelRequestDeployModelTypeEnum"
        },
        {
            "name": "useConverter",
            "baseName": "useConverter",
            "type": "DeployPretrainedModelRequestUseConverterEnum"
        }    ];

    static getAttributeTypeMap() {
        return DeployPretrainedModelRequest.attributeTypeMap;
    }
}


export type DeployPretrainedModelRequestModelFileTypeEnum = 'tflite' | 'onnx' | 'saved_model' | 'lgbm' | 'xgboost' | 'pickle';
export const DeployPretrainedModelRequestModelFileTypeEnumValues: string[] = ['tflite', 'onnx', 'saved_model', 'lgbm', 'xgboost', 'pickle'];

export type DeployPretrainedModelRequestDeployModelTypeEnum = 'int8' | 'float32';
export const DeployPretrainedModelRequestDeployModelTypeEnumValues: string[] = ['int8', 'float32'];

export type DeployPretrainedModelRequestUseConverterEnum = 'onnx-tf' | 'onnx2tf';
export const DeployPretrainedModelRequestUseConverterEnumValues: string[] = ['onnx-tf', 'onnx2tf'];