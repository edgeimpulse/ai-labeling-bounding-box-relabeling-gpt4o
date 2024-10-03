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

import { DetailedImpulseMetricFilteringType } from './detailedImpulseMetricFilteringType';

export class GetAllDetailedImpulsesResponseAllOfMetricKeys {
    'name': string;
    'description': string;
    'type': GetAllDetailedImpulsesResponseAllOfMetricKeysTypeEnum;
    'filteringType'?: DetailedImpulseMetricFilteringType;
    'showInTable': boolean;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "name",
            "baseName": "name",
            "type": "string"
        },
        {
            "name": "description",
            "baseName": "description",
            "type": "string"
        },
        {
            "name": "type",
            "baseName": "type",
            "type": "GetAllDetailedImpulsesResponseAllOfMetricKeysTypeEnum"
        },
        {
            "name": "filteringType",
            "baseName": "filteringType",
            "type": "DetailedImpulseMetricFilteringType"
        },
        {
            "name": "showInTable",
            "baseName": "showInTable",
            "type": "boolean"
        }    ];

    static getAttributeTypeMap() {
        return GetAllDetailedImpulsesResponseAllOfMetricKeys.attributeTypeMap;
    }
}


export type GetAllDetailedImpulsesResponseAllOfMetricKeysTypeEnum = 'core' | 'additional';
export const GetAllDetailedImpulsesResponseAllOfMetricKeysTypeEnumValues: string[] = ['core', 'additional'];