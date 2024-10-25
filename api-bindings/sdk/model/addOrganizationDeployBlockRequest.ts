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


export class AddOrganizationDeployBlockRequest {
    'name': string;
    'dockerContainer': string;
    'description': string;
    'cliArguments': string;
    'requestsCpu'?: number;
    'requestsMemory'?: number;
    'limitsCpu'?: number;
    'limitsMemory'?: number;
    'photo'?:{ fieldname: string, originalname: string, encoding: string, mimetype: string, buffer: Buffer, size: number }[];
    'integrateUrl'?: string;
    'privileged'?: boolean;
    'mountLearnBlock'?: boolean;
    'supportsEonCompiler'?: boolean;
    'showOptimizations'?: boolean;
    'category'?: AddOrganizationDeployBlockRequestCategoryEnum;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "name",
            "baseName": "name",
            "type": "string"
        },
        {
            "name": "dockerContainer",
            "baseName": "dockerContainer",
            "type": "string"
        },
        {
            "name": "description",
            "baseName": "description",
            "type": "string"
        },
        {
            "name": "cliArguments",
            "baseName": "cliArguments",
            "type": "string"
        },
        {
            "name": "requestsCpu",
            "baseName": "requestsCpu",
            "type": "number"
        },
        {
            "name": "requestsMemory",
            "baseName": "requestsMemory",
            "type": "number"
        },
        {
            "name": "limitsCpu",
            "baseName": "limitsCpu",
            "type": "number"
        },
        {
            "name": "limitsMemory",
            "baseName": "limitsMemory",
            "type": "number"
        },
        {
            "name": "photo",
            "baseName": "photo",
            "type": "RequestFile"
        },
        {
            "name": "integrateUrl",
            "baseName": "integrateUrl",
            "type": "string"
        },
        {
            "name": "privileged",
            "baseName": "privileged",
            "type": "boolean"
        },
        {
            "name": "mountLearnBlock",
            "baseName": "mountLearnBlock",
            "type": "boolean"
        },
        {
            "name": "supportsEonCompiler",
            "baseName": "supportsEonCompiler",
            "type": "boolean"
        },
        {
            "name": "showOptimizations",
            "baseName": "showOptimizations",
            "type": "boolean"
        },
        {
            "name": "category",
            "baseName": "category",
            "type": "AddOrganizationDeployBlockRequestCategoryEnum"
        }    ];

    static getAttributeTypeMap() {
        return AddOrganizationDeployBlockRequest.attributeTypeMap;
    }
}


export type AddOrganizationDeployBlockRequestCategoryEnum = 'library' | 'firmware';
export const AddOrganizationDeployBlockRequestCategoryEnumValues: string[] = ['library', 'firmware'];
