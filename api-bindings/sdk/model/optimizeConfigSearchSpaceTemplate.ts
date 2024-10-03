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


/**
* Search space template
*/
export class OptimizeConfigSearchSpaceTemplate {
    /**
    * Search space template identifier
    */
    'identifier': OptimizeConfigSearchSpaceTemplateIdentifierEnum;
    /**
    * Whether a classification block should be added to the search space
    */
    'classification'?: boolean;
    /**
    * Whether an anomaly block should be added to the search space
    */
    'anomaly'?: boolean;
    /**
    * Whether a regression block should be added to the search space
    */
    'regression'?: boolean;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "identifier",
            "baseName": "identifier",
            "type": "OptimizeConfigSearchSpaceTemplateIdentifierEnum"
        },
        {
            "name": "classification",
            "baseName": "classification",
            "type": "boolean"
        },
        {
            "name": "anomaly",
            "baseName": "anomaly",
            "type": "boolean"
        },
        {
            "name": "regression",
            "baseName": "regression",
            "type": "boolean"
        }    ];

    static getAttributeTypeMap() {
        return OptimizeConfigSearchSpaceTemplate.attributeTypeMap;
    }
}


export type OptimizeConfigSearchSpaceTemplateIdentifierEnum = 'speech_keyword' | 'speech_continuous' | 'audio_event' | 'audio_continuous' | 'visual' | 'motion_event' | 'motion_continuous' | 'audio_syntiant' | 'object_detection_bounding_boxes' | 'object_detection_centroids' | 'visual_ad';
export const OptimizeConfigSearchSpaceTemplateIdentifierEnumValues: string[] = ['speech_keyword', 'speech_continuous', 'audio_event', 'audio_continuous', 'visual', 'motion_event', 'motion_continuous', 'audio_syntiant', 'object_detection_bounding_boxes', 'object_detection_centroids', 'visual_ad'];