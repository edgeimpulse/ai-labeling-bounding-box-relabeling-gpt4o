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


export class OrganizationAddDataFolderResponseAllOf {
    'dataItemCount': number;
    'dataFirstItems': Array<string>;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "dataItemCount",
            "baseName": "dataItemCount",
            "type": "number"
        },
        {
            "name": "dataFirstItems",
            "baseName": "dataFirstItems",
            "type": "Array<string>"
        }    ];

    static getAttributeTypeMap() {
        return OrganizationAddDataFolderResponseAllOf.attributeTypeMap;
    }
}

