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

import { OrganizationAddDatasetRequestBucket } from './organizationAddDatasetRequestBucket';
import { OrganizationDatasetTypeEnum } from './organizationDatasetTypeEnum';

export class OrganizationAddDatasetRequest {
    'dataset': string;
    'tags': Array<string>;
    'category': string;
    'type': OrganizationDatasetTypeEnum;
    'bucket': OrganizationAddDatasetRequestBucket;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "dataset",
            "baseName": "dataset",
            "type": "string"
        },
        {
            "name": "tags",
            "baseName": "tags",
            "type": "Array<string>"
        },
        {
            "name": "category",
            "baseName": "category",
            "type": "string"
        },
        {
            "name": "type",
            "baseName": "type",
            "type": "OrganizationDatasetTypeEnum"
        },
        {
            "name": "bucket",
            "baseName": "bucket",
            "type": "OrganizationAddDatasetRequestBucket"
        }    ];

    static getAttributeTypeMap() {
        return OrganizationAddDatasetRequest.attributeTypeMap;
    }
}
