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

import { WindowSettings } from './windowSettings';

export class WindowSettingsResponseAllOf {
    'windowSettingsEvent': Array<WindowSettings>;
    'windowSettingsContinuous': Array<WindowSettings>;

    static discriminator: string | undefined = undefined;

    static attributeTypeMap: Array<{name: string, baseName: string, type: string}> = [
        {
            "name": "windowSettingsEvent",
            "baseName": "windowSettingsEvent",
            "type": "Array<WindowSettings>"
        },
        {
            "name": "windowSettingsContinuous",
            "baseName": "windowSettingsContinuous",
            "type": "Array<WindowSettings>"
        }    ];

    static getAttributeTypeMap() {
        return WindowSettingsResponseAllOf.attributeTypeMap;
    }
}
