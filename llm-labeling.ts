import fs from 'fs';
import Path from 'path';
import program from 'commander';
import { EdgeImpulseApi } from './api-bindings';
import * as models from './api-bindings/sdk/model/models';
import OpenAI from "openai";
import asyncPool from 'tiny-async-pool';
import sharp from 'sharp';

const packageVersion = (<{ version: string }>JSON.parse(fs.readFileSync(
    Path.join(__dirname, '..', 'package.json'), 'utf-8'))).version;

if (!process.env.EI_PROJECT_API_KEY) {
    console.log('Missing EI_PROJECT_API_KEY');
    process.exit(1);
}
if (!process.env.OPENAI_API_KEY) {
    console.log('Missing OPENAI_API_KEY');
    process.exit(1);
}

let API_URL = process.env.EI_API_ENDPOINT || 'https://studio.edgeimpulse.com/v1';
const API_KEY = process.env.EI_PROJECT_API_KEY;

API_URL = API_URL.replace('/v1', '');

program
    .description('Label bounding boxes ' + packageVersion)
    .version(packageVersion)
    .requiredOption('--which-labels <labels>',
        `List of bounding box labels to relabel (separate by comma)`)
    .requiredOption('--prompt <prompt>',
        `A prompt asking a question to the LLM. ` +
        `The answer should be a single label. ` +
        `E.g. "Is there a human in this picture, respond with only 'yes' or 'no'."`)
    .option('--disable-labels <labels>',
        `If a certain label is output, disable the data item. ` +
        `E.g. your prompt can be: "If the picture is blurry, respond with 'blurry'", ` +
        `and add "blurry" to the disabled labels. Multiple labels can be split by ",".`
    )
    .option('--concurrency <n>', `Concurrency (default: 1)`)
    .requiredOption('--data-ids-file <file>', 'File with IDs (as JSON)')
    .option('--propose-actions <job-id>', 'If this flag is passed in, only propose suggested actions')
    .option('--verbose', 'Enable debug logs')
    .allowUnknownOption(true)
    .parse(process.argv);

const api = new EdgeImpulseApi({ endpoint: API_URL });

const whichLabelsArgv = (<string[]>(<string>program.whichLabels || '').split(',')).map(x => x.trim().toLowerCase()).filter(x => !!x);
// the replacement looks weird; but if calling this from CLI like
// "--prompt 'test\nanother line'" we'll get this still escaped
// (you could use $'test\nanotherline' but we won't do that in the Edge Impulse backend)
const promptArgv = (<string>program.prompt).replace('\\n', '\n');
const disableLabelsArgv = (<string[]>(<string | undefined>program.disableLabels || '').split(',')).map(x => x.trim().toLowerCase()).filter(x => !!x);
const concurrencyArgv = program.concurrency ? Number(program.concurrency) : 1;
const dataIdsFile = <string>program.dataIdsFile;
const proposeActionsJobId = program.proposeActions ?
    Number(program.proposeActions) :
    undefined;

if (proposeActionsJobId && isNaN(proposeActionsJobId)) {
    console.log('--propose-actions should be numeric');
    process.exit(1);
}
let dataIds: number[] | undefined;
if (!fs.existsSync(dataIdsFile)) {
    console.log(`"${dataIdsFile}" does not exist (via --data-ids-file)`);
    process.exit(1);
}
try {
    dataIds = <number[]>JSON.parse(fs.readFileSync(dataIdsFile, 'utf-8'));
    if (!Array.isArray(dataIds)) {
        throw new Error('Content of the file is not an array');
    }
    for (let ix = 0; ix < dataIds.length; ix++) {
        if (isNaN(dataIds[ix])) {
            throw new Error('The value at index ' + ix + ' is not numeric');
        }
    }
}
catch (ex2) {
    console.log(`Failed to parse "${dataIdsFile}" (via --data-ids-file), should be a JSON array with numbers`, ex2);
    process.exit(1);
}

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    try {
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

        await api.authenticate({
            method: 'apiKey',
            apiKey: API_KEY,
        });

        // listProjects returns a single project if authenticated by API key
        const project = (await api.projects.listProjects()).projects[0];

        console.log(`Re-labeling bounding boxes for "${project.owner} / ${project.name}"`);
        console.log(`    Bounding boxes to relabel: "${whichLabelsArgv.join(', ')}"`);
        console.log(`    Prompt: "${promptArgv}"`);
        console.log(`    Remove bounding boxes with labels: ${disableLabelsArgv.length === 0 ? '-' : disableLabelsArgv.join(', ')}`);
        console.log(`    Concurrency: ${concurrencyArgv}`);
        if (dataIds.length < 6) {
            console.log(`    IDs: ${dataIds.join(', ')}`);
        }
        else {
            console.log(`    IDs: ${dataIds.slice(0, 5).join(', ')} and ${dataIds.length - 5} others`);
        }
        console.log(``);

        let samplesToProcess: models.Sample[];

        console.log(`Finding data by ID...`);
        samplesToProcess = await listDataByIds(project.id, dataIds);
        console.log(`Finding data by ID OK (found ${samplesToProcess.length} samples)`);
        console.log(``);

        samplesToProcess = samplesToProcess.sort((a, b) => a.id - b.id);

        const total = samplesToProcess.length;
        let gptQueryCount = 0;
        let activeGptQueries = 0;
        let processed = 0;
        let error = 0;

        const getSummary = () => {
            return `(query_count=${gptQueryCount}, error=${error})`;
        };

        let updateIv = setInterval(async () => {
            let currFile = (processed).toString().padStart(total.toString().length, ' ');
            console.log(`[${currFile}/${total}] Labeling samples... ` +
                getSummary());
        }, 3000);

        const labelSampleWithOpenAI = async (sample: models.Sample) => {
            try {
                const fullImgBuffer = await retryWithTimeout(async () => {
                    return await api.rawData.getSampleAsImage(project.id, sample.id, { });
                }, {
                    fnName: 'rawData.getSampleAsImage',
                    maxRetries: 3,
                    onWarning: (retriesLeft, ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] WARN: Failed to read image for ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}. Retries left=${retriesLeft}`);
                    },
                    onError: (ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] ERR: Failed to read image for ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}.`);
                    },
                    timeoutMs: 30000,
                });

                const imgMetadata = await sharp(fullImgBuffer).metadata();

                let newBbs: models.BoundingBox[] = [];

                await asyncPool(concurrencyArgv, sample.boundingBoxes, async (bb) => {
                    // don't label this
                    if (whichLabelsArgv.indexOf(bb.label.trim().toLowerCase()) === -1) {
                        newBbs.push(bb);
                        return;
                    }

                    if (bb.x < 0) bb.x = 0;
                    if (bb.y < 0) bb.y = 0;
                    if (bb.x + bb.width > imgMetadata.width!) bb.width = (imgMetadata.width! - bb.x);
                    if (bb.y + bb.height > imgMetadata.height!) bb.height = (imgMetadata.height! - bb.y);
                    if (bb.width <= 0) return;
                    if (bb.height <= 0) return;

                    const croppedBuffer = await sharp(fullImgBuffer).extract({
                        left: bb.x,
                        top: bb.y,
                        width: bb.width,
                        height: bb.height,
                    }).jpeg({ quality: 90 }).toBuffer();

                    // too many queries? wait...
                    while (activeGptQueries >= concurrencyArgv) {
                        await new Promise<void>(resolve => setTimeout(resolve, 100));
                    }

                    const newLabel = await retryWithTimeout(async () => {
                        try {
                            activeGptQueries++;

                            const resp = await openai.chat.completions.create({
                                model: 'gpt-4o-2024-08-06',
                                messages: [{
                                    role: 'user',
                                    content: [{
                                        type: 'text',
                                        text: promptArgv,
                                    }, {
                                        type: 'image_url',
                                        image_url: {
                                            url: 'data:image/jpeg;base64,' + (croppedBuffer.toString('base64')),
                                            detail: 'auto'
                                        }
                                    }]
                                }]
                            });

                            // console.log('resp', JSON.stringify(resp, null, 4));

                            if (resp.choices.length !== 1) {
                                throw new Error('Expected choices to have 1 item (' + JSON.stringify(resp) + ')');
                            }
                            if (resp.choices[0].message.role !== 'assistant') {
                                throw new Error('Expected choices[0].message.role to equal "assistant" (' + JSON.stringify(resp) + ')');
                            }
                            if (typeof resp.choices[0].message.content !== 'string') {
                                throw new Error('Expected choices[0].message.content to be a string (' + JSON.stringify(resp) + ')');
                            }

                            let label = resp.choices[0].message.content.toLowerCase();
                            if (label.endsWith('.')) {
                                label = label.replace(/(\.)+$/, '');
                            }
                            return label;
                        }
                        finally {
                            activeGptQueries--;
                        }
                    }, {
                        fnName: 'completions.create',
                        maxRetries: 3,
                        onWarning: (retriesLeft, ex) => {
                            let currFile = (processed).toString().padStart(total.toString().length, ' ');
                            console.log(`[${currFile}/${total}] WARN: Failed to label ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}. Retries left=${retriesLeft}`);
                        },
                        onError: (ex) => {
                            let currFile = (processed).toString().padStart(total.toString().length, ' ');
                            console.log(`[${currFile}/${total}] ERR: Failed to label ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}.`);
                        },
                        timeoutMs: 60000,
                    });

                    gptQueryCount++;

                    if (disableLabelsArgv.indexOf(newLabel) > -1) {
                        // remove bb
                        return;
                    }

                    bb.label = newLabel;
                    newBbs.push(bb);
                });

                await retryWithTimeout(async () => {
                    // dry-run, only propose?
                    if (proposeActionsJobId) {
                        await api.rawData.setSampleProposedChanges(project.id, sample.id, {
                            jobId: proposeActionsJobId,
                            proposedChanges: {
                                boundingBoxes: newBbs,
                            }
                        });
                    }
                    // actually perform actions
                    else {
                        await api.rawData.setSampleBoundingBoxes(project.id, sample.id, {
                            boundingBoxes: newBbs,
                        });
                    }
                }, {
                    fnName: 'edgeimpulse.api',
                    maxRetries: 3,
                    timeoutMs: 60000,
                    onWarning: (retriesLeft, ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] WARN: Failed to update bounding boxes for ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}. Retries left=${retriesLeft}`);
                    },
                    onError: (ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] ERR: Failed to update bounding boxes for ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}.`);
                    },
                });
            }
            catch (ex2) {
                let ex = <Error>ex2;
                let currFile = (processed + 1).toString().padStart(total.toString().length, ' ');
                console.log(`[${currFile}/${total}] Failed to label sample "${sample.filename}" (ID: ${sample.id}): ` +
                    (ex.message || ex.toString()));
                error++;
            }
            finally {
                processed++;
            }
        };

        try {
            console.log(`Labeling ${total.toLocaleString()} samples...`);

            await asyncPool(concurrencyArgv, samplesToProcess.slice(0, total), labelSampleWithOpenAI);

            clearInterval(updateIv);

            console.log(`[${total}/${total}] Labeling samples... ` + getSummary());
            console.log(`Done labeling samples, goodbye!`);
        }
        finally {
            clearInterval(updateIv);
        }
    }
    catch (ex2) {
        let ex = <Error>ex2;
        console.log('Failed to label data:', ex.message || ex.toString());
        process.exit(1);
    }

    process.exit(0);
})();

async function listDataByIds(projectId: number, ids: number[]) {
    const limit = 1000;
    let offset = 0;
    let allSamples: models.Sample[] = [];

    let iv = setInterval(() => {
        console.log(`Still finding data (found ${allSamples.length} samples)...`);
    }, 3000);

    try {
        while (1) {
            let ret = await api.rawData.listSamples(projectId, {
                category: 'training',
                labels: '',
                offset: offset,
                limit: limit,
            });
            if (ret.samples.length === 0) {
                break;
            }
            for (let s of ret.samples) {
                if (ids.indexOf(s.id) !== -1) {
                    allSamples.push(s);
                }
            }
            offset += limit;
        }

        offset = 0;
        while (1) {
            let ret = await api.rawData.listSamples(projectId, {
                category: 'testing',
                labels: '',
                offset: offset,
                limit: limit,
            });
            if (ret.samples.length === 0) {
                break;
            }
            for (let s of ret.samples) {
                if (ids.indexOf(s.id) !== -1) {
                    allSamples.push(s);
                }
            }
            offset += limit;
        }
    }
    finally {
        clearInterval(iv);
    }
    return allSamples;
}

async function listAllVideos(projectId: number) {
    const limit = 1000;
    let offset = 0;
    let allSamples: models.Sample[] = [];

    let iv = setInterval(() => {
        console.log(`Still listing videos (found ${allSamples.length} samples)...`);
    }, 3000);

    try {
        while (1) {
            let ret = await api.rawData.listSamples(projectId, {
                category: 'training',
                labels: '',
                offset: offset,
                limit: limit,
            });
            if (ret.samples.length === 0) {
                break;
            }
            for (let s of ret.samples) {
                if (s.chartType === 'video' && !s.isProcessing) {
                    allSamples.push(s);
                }
            }
            offset += limit;
        }
        while (1) {
            let ret = await api.rawData.listSamples(projectId, {
                category: 'testing',
                labels: '',
                offset: offset,
                limit: limit,
            });
            if (ret.samples.length === 0) {
                break;
            }
            for (let s of ret.samples) {
                if (s.chartType === 'video' && !s.isProcessing) {
                    allSamples.push(s);
                }
            }
            offset += limit;
        }
    }
    finally {
        clearInterval(iv);
    }
    return allSamples;
}


export async function retryWithTimeout<T>(fn: () => Promise<T>, opts: {
    fnName: string,
    timeoutMs: number,
    maxRetries: number,
    onWarning: (retriesLeft: number, ex: Error) => void,
    onError: (ex: Error) => void,
}) {
    const { timeoutMs, maxRetries, onWarning, onError } = opts;

    let retriesLeft = maxRetries;

    let ret: T;

    while (1) {
        try {
            ret = await new Promise<T>(async (resolve, reject) => {
                let timeout = setTimeout(() => {
                    reject(opts.fnName + ' did not return within ' + timeoutMs + 'ms.');
                }, timeoutMs);

                try {
                    const b = await fn();

                    resolve(b);
                }
                catch (ex) {
                    reject(ex);
                }
                finally {
                    clearTimeout(timeout);
                }
            });

            break;
        }
        catch (ex2) {
            let ex = <Error>ex2;

            retriesLeft = retriesLeft - 1;
            if (retriesLeft === 0) {
                onError(ex);
                throw ex2;
            }

            onWarning(retriesLeft, ex);
        }
    }

    return ret!;
}
