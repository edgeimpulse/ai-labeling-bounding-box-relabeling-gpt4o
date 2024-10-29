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


export class UpdateWhitelabelInternalRequest {
    /**
    * The maximum number of organizations that can be created under this white label.
    */
    'organizationsLimit'?: number | null;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "organizationsLimit",
            "baseName": "organizationsLimit",
            "type": "number"
        }    ];

    static getAttributeTypeMap() {
        return UpdateWhitelabelInternalRequest.attributeTypeMap;
    }
}
