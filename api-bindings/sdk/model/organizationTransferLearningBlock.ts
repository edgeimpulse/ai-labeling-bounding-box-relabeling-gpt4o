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
import { CreatedUpdatedByUser } from './createdUpdatedByUser';
import { ImageInputScaling } from './imageInputScaling';
import { ObjectDetectionLastLayer } from './objectDetectionLastLayer';
import { OrganizationTransferLearningBlockCustomVariant } from './organizationTransferLearningBlockCustomVariant';
import { OrganizationTransferLearningOperatesOn } from './organizationTransferLearningOperatesOn';
import { PublicProjectTierAvailability } from './publicProjectTierAvailability';

export class OrganizationTransferLearningBlock {
    'id': number;
    'name': string;
    'dockerContainer': string;
    'dockerContainerManagedByEdgeImpulse': boolean;
    'created': Date;
    'createdByUser'?: CreatedUpdatedByUser;
    'lastUpdated'?: Date;
    'lastUpdatedByUser'?: CreatedUpdatedByUser;
    'description': string;
    'userId'?: number;
    'userName'?: string;
    'operatesOn': OrganizationTransferLearningOperatesOn;
    'objectDetectionLastLayer'?: ObjectDetectionLastLayer;
    'implementationVersion': number;
    /**
    * Whether this block is publicly available to Edge Impulse users (if false, then only for members of the owning organization)
    */
    'isPublic': boolean;
    /**
    * If `isPublic` is true, the list of devices (from latencyDevices) for which this model can be shown.
    */
    'isPublicForDevices': Array<string>;
    'publicProjectTierAvailability'?: PublicProjectTierAvailability;
    /**
    * Whether this block is publicly available to only enterprise users
    */
    'isPublicEnterpriseOnly': boolean;
    /**
    * Whether this block is available to only enterprise users
    */
    'enterpriseOnly'?: boolean;
    /**
    * URL to the source code of this custom learn block.
    */
    'repositoryUrl'?: string;
    /**
    * List of parameters, spec\'ed according to https://docs.edgeimpulse.com/docs/tips-and-tricks/adding-parameters-to-custom-blocks
    */
    'parameters': Array<object>;
    'imageInputScaling'?: ImageInputScaling;
    /**
    * If set, requires this block to be scheduled on GPU.
    */
    'indRequiresGpu': boolean;
    'sourceCodeAvailable': boolean;
    'displayCategory'?: BlockDisplayCategory;
    /**
    * List of custom model variants produced when this block is trained. This is experimental and may change in the future.
    */
    'customModelVariants'?: Array<OrganizationTransferLearningBlockCustomVariant>;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "id",
            "baseName": "id",
            "type": "number"
        },
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
            "name": "dockerContainerManagedByEdgeImpulse",
            "baseName": "dockerContainerManagedByEdgeImpulse",
            "type": "boolean"
        },
        {
            "name": "created",
            "baseName": "created",
            "type": "Date"
        },
        {
            "name": "createdByUser",
            "baseName": "createdByUser",
            "type": "CreatedUpdatedByUser"
        },
        {
            "name": "lastUpdated",
            "baseName": "lastUpdated",
            "type": "Date"
        },
        {
            "name": "lastUpdatedByUser",
            "baseName": "lastUpdatedByUser",
            "type": "CreatedUpdatedByUser"
        },
        {
            "name": "description",
            "baseName": "description",
            "type": "string"
        },
        {
            "name": "userId",
            "baseName": "userId",
            "type": "number"
        },
        {
            "name": "userName",
            "baseName": "userName",
            "type": "string"
        },
        {
            "name": "operatesOn",
            "baseName": "operatesOn",
            "type": "OrganizationTransferLearningOperatesOn"
        },
        {
            "name": "objectDetectionLastLayer",
            "baseName": "objectDetectionLastLayer",
            "type": "ObjectDetectionLastLayer"
        },
        {
            "name": "implementationVersion",
            "baseName": "implementationVersion",
            "type": "number"
        },
        {
            "name": "isPublic",
            "baseName": "isPublic",
            "type": "boolean"
        },
        {
            "name": "isPublicForDevices",
            "baseName": "isPublicForDevices",
            "type": "Array<string>"
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
            "name": "enterpriseOnly",
            "baseName": "enterpriseOnly",
            "type": "boolean"
        },
        {
            "name": "repositoryUrl",
            "baseName": "repositoryUrl",
            "type": "string"
        },
        {
            "name": "parameters",
            "baseName": "parameters",
            "type": "Array<object>"
        },
        {
            "name": "imageInputScaling",
            "baseName": "imageInputScaling",
            "type": "ImageInputScaling"
        },
        {
            "name": "indRequiresGpu",
            "baseName": "indRequiresGpu",
            "type": "boolean"
        },
        {
            "name": "sourceCodeAvailable",
            "baseName": "sourceCodeAvailable",
            "type": "boolean"
        },
        {
            "name": "displayCategory",
            "baseName": "displayCategory",
            "type": "BlockDisplayCategory"
        },
        {
            "name": "customModelVariants",
            "baseName": "customModelVariants",
            "type": "Array<OrganizationTransferLearningBlockCustomVariant>"
        }    ];

    static getAttributeTypeMap() {
        return OrganizationTransferLearningBlock.attributeTypeMap;
    }
}
