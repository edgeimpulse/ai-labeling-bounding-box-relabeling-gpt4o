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

import { ClassifyJobResponseAllOf } from './classifyJobResponseAllOf';
import { ClassifyJobResponseAllOfAccuracy } from './classifyJobResponseAllOfAccuracy';
import { ClassifyJobResponseAllOfAdditionalMetricsByLearnBlock } from './classifyJobResponseAllOfAdditionalMetricsByLearnBlock';
import { GenericApiResponse } from './genericApiResponse';
import { KerasModelVariantEnum } from './kerasModelVariantEnum';
import { ModelPrediction } from './modelPrediction';
import { ModelResult } from './modelResult';

export class ClassifyJobResponse {
    /**
    * Whether the operation succeeded
    */
    'success': boolean;
    /**
    * Optional error description (set if \'success\' was false)
    */
    'error'?: string;
    'result': Array<ModelResult>;
    'predictions': Array<ModelPrediction>;
    'accuracy': ClassifyJobResponseAllOfAccuracy;
    'additionalMetricsByLearnBlock': Array<ClassifyJobResponseAllOfAdditionalMetricsByLearnBlock>;
    /**
    * List of all model variants for which classification results exist
    */
    'availableVariants': Array<KerasModelVariantEnum>;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "success",
            "baseName": "success",
            "type": "boolean"
        },
        {
            "name": "error",
            "baseName": "error",
            "type": "string"
        },
        {
            "name": "result",
            "baseName": "result",
            "type": "Array<ModelResult>"
        },
        {
            "name": "predictions",
            "baseName": "predictions",
            "type": "Array<ModelPrediction>"
        },
        {
            "name": "accuracy",
            "baseName": "accuracy",
            "type": "ClassifyJobResponseAllOfAccuracy"
        },
        {
            "name": "additionalMetricsByLearnBlock",
            "baseName": "additionalMetricsByLearnBlock",
            "type": "Array<ClassifyJobResponseAllOfAdditionalMetricsByLearnBlock>"
        },
        {
            "name": "availableVariants",
            "baseName": "availableVariants",
            "type": "Array<KerasModelVariantEnum>"
        }    ];

    static getAttributeTypeMap() {
        return ClassifyJobResponse.attributeTypeMap;
    }
}
