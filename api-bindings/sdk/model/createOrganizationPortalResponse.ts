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

import { CreateOrganizationPortalResponseAllOf } from './createOrganizationPortalResponseAllOf';
import { GenericApiResponse } from './genericApiResponse';

export class CreateOrganizationPortalResponse {
    /**
    * Whether the operation succeeded
    */
    'success': boolean;
    /**
    * Optional error description (set if \'success\' was false)
    */
    'error'?: string;
    /**
    * Portal ID for the new upload portal
    */
    'id': number;
    /**
    * URL to the portal
    */
    'url': string;
    /**
    * pre-signed upload URL. Only set if using a non-built-in bucket.
    */
    'signedUrl'?: string;
    /**
    * Only set if using a non-built-in bucket.
    */
    'bucketBucket'?: string;

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
            "name": "id",
            "baseName": "id",
            "type": "number"
        },
        {
            "name": "url",
            "baseName": "url",
            "type": "string"
        },
        {
            "name": "signedUrl",
            "baseName": "signedUrl",
            "type": "string"
        },
        {
            "name": "bucketBucket",
            "baseName": "bucketBucket",
            "type": "string"
        }    ];

    static getAttributeTypeMap() {
        return CreateOrganizationPortalResponse.attributeTypeMap;
    }
}

