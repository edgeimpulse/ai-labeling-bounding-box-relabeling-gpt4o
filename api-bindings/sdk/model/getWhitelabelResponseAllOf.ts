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

import { Whitelabel } from './whitelabel';

export class GetWhitelabelResponseAllOf {
    'whitelabel'?: Whitelabel;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "whitelabel",
            "baseName": "whitelabel",
            "type": "Whitelabel"
        }    ];

    static getAttributeTypeMap() {
        return GetWhitelabelResponseAllOf.attributeTypeMap;
    }
}

