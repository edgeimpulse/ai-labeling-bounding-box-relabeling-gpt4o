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

import { BlockType } from './blockType';

export class InputBlock {
    'type': InputBlockTypeEnum;
    'title': string;
    'author': string;
    'description': string;
    'name': string;
    'recommended'?: boolean;
    'blockType': BlockType;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "type",
            "baseName": "type",
            "type": "InputBlockTypeEnum"
        },
        {
            "name": "title",
            "baseName": "title",
            "type": "string"
        },
        {
            "name": "author",
            "baseName": "author",
            "type": "string"
        },
        {
            "name": "description",
            "baseName": "description",
            "type": "string"
        },
        {
            "name": "name",
            "baseName": "name",
            "type": "string"
        },
        {
            "name": "recommended",
            "baseName": "recommended",
            "type": "boolean"
        },
        {
            "name": "blockType",
            "baseName": "blockType",
            "type": "BlockType"
        }    ];

    static getAttributeTypeMap() {
        return InputBlock.attributeTypeMap;
    }
}


export type InputBlockTypeEnum = 'time-series' | 'image' | 'features';
export const InputBlockTypeEnumValues: string[] = ['time-series', 'image', 'features'];