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

import { GenericApiResponse } from './genericApiResponse';
import { PortalFile } from './portalFile';
import { PreviewDefaultFilesInFolderResponseAllOf } from './previewDefaultFilesInFolderResponseAllOf';

export class PreviewDefaultFilesInFolderResponse {
    /**
    * Whether the operation succeeded
    */
    'success': boolean;
    /**
    * Optional error description (set if \'success\' was false)
    */
    'error'?: string;
    'files': Array<PortalFile>;
    /**
    * True if results are truncated.
    */
    'isTruncated'?: boolean;
    /**
    * Explains why results are truncated; only present in the response if isTruncated is true. Results can be truncated if there are too many results (more than 500 matches), or if searching for more results is too expensive (for example, the dataset contains many items but very few match the given wildcard). 
    */
    'truncationReason'?: PreviewDefaultFilesInFolderResponseTruncationReasonEnum;

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
            "name": "files",
            "baseName": "files",
            "type": "Array<PortalFile>"
        },
        {
            "name": "isTruncated",
            "baseName": "isTruncated",
            "type": "boolean"
        },
        {
            "name": "truncationReason",
            "baseName": "truncationReason",
            "type": "PreviewDefaultFilesInFolderResponseTruncationReasonEnum"
        }    ];

    static getAttributeTypeMap() {
        return PreviewDefaultFilesInFolderResponse.attributeTypeMap;
    }
}


export type PreviewDefaultFilesInFolderResponseTruncationReasonEnum = 'too-many-results' | 'too-expensive-search';
export const PreviewDefaultFilesInFolderResponseTruncationReasonEnumValues: string[] = ['too-many-results', 'too-expensive-search'];