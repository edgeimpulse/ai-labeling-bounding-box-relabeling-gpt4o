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

import { GetStudioConfigResponseAllOfConfig } from './getStudioConfigResponseAllOfConfig';

export class GetStudioConfigResponseAllOf {
    /**
    * List of config items
    */
    'config': Array<GetStudioConfigResponseAllOfConfig>;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "config",
            "baseName": "config",
            "type": "Array<GetStudioConfigResponseAllOfConfig>"
        }    ];

    static getAttributeTypeMap() {
        return GetStudioConfigResponseAllOf.attributeTypeMap;
    }
}
