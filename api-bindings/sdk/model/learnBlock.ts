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

import { BlockDisplayCategory } from './blockDisplayCategory';
import { BlockType } from './blockType';
import { PublicProjectTierAvailability } from './publicProjectTierAvailability';

export class LearnBlock {
    'type': string;
    'title': string;
    'author': string;
    'description': string;
    'name': string;
    'recommended'?: boolean;
    'organizationModelId'?: number;
    'publicProjectTierAvailability'?: PublicProjectTierAvailability;
    /**
    * Whether this block is publicly available to only enterprise users
    */
    'isPublicEnterpriseOnly'?: boolean;
    'blockType': BlockType;
    'displayCategory'?: BlockDisplayCategory;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "type",
            "baseName": "type",
            "type": "string"
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
            "name": "organizationModelId",
            "baseName": "organizationModelId",
            "type": "number"
        },
        {
            "name": "publicProjectTierAvailability",
            "baseName": "publicProjectTierAvailability",
            "type": "PublicProjectTierAvailability"
        },
        {
            "name": "isPublicEnterpriseOnly",
            "baseName": "isPublicEnterpriseOnly",
            "type": "boolean"
        },
        {
            "name": "blockType",
            "baseName": "blockType",
            "type": "BlockType"
        },
        {
            "name": "displayCategory",
            "baseName": "displayCategory",
            "type": "BlockDisplayCategory"
        }    ];

    static getAttributeTypeMap() {
        return LearnBlock.attributeTypeMap;
    }
}
