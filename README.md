# AI Actions block: Re-label bounding boxes with GPT4o

This is an Edge Impulse [AI Actions block](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/custom-blocks/transformation-blocks) that uses GPT4o to relabel or delete bounding boxes. This is super useful in combination with a [zero-shot object detection labeling block](https://github.com/edgeimpulse/zero-shot-object-detector-labeling-block). Zero-shot object detectors are great at finding objects, but not too great at labeling them accurately. Using AI Actions in Edge Impulse you can use a 2-stage approach. First, you greedily label bounding boxes using the zero-shot object detector; then use an LLM to accurately relabel them.

## Example: label chess pieces

Let's say you want to label individual chess pieces by their name (e.g. rook, bishop, king, etc.). OWL-ViT (a zero-shot object detector available in Edge Impulse) cannot distinguish a rook from a king accurately - but it _can_ find chess pieces. Here we can combine the zero-shot object detector with GPT4o to get highly accurate labels.

### Step 1: Use a zero-shot object detector to find the pieces

[Chess pieces](images/zero_shot1.png)

### Step 2: Using GPT4o to relabel the bounding boxes

[Labeled chess pieces](images/zero_shot2.png)

Here GPT4o relabeled all chess pieces with accurate names, and removed erronous ones (like the chess board).

## Use this from Edge Impulse (professional / enterprise)

If you just want to use GPT4o as an object detection re-labeling tool in your Edge Impulse project you don't need this repo. Just go to any project, select **Data acquisition > AI Actions**, choose **Re-label bounding boxes with GPT4o** (available for professional and enterprise projects only).

## Developing your own block

You can use this repository to develop your own block that uses GPT4o (or some other LLM) to help you re-label data, or add metadata.

### Running this block locally

1. Create a new Edge Impulse project, add some images, and add some bounding boxes to those images.
2. Create a file called `ids.json` and add the IDs of the samples you want to label. You can find the sample ID by clicking the 'expand' button on **Data acquisiton**.

    ![Finding IDs](images/find_ids.png)

    Add these IDs to the `ids.json` file as an array of numbers, e.g.:

    ```json
    [1299267659, 1299267609, 1299267606]
    ```

3. Load your API keys (both Edge Impulse and OpenAI):

    ```
    export OPENAI_API_KEY=sk-M...
    export EI_PROJECT_API_KEY=ei_44...
    ```

    > You can find your OpenAI API key on the [OpenAI API Keys](https://platform.openai.com/api-keys) page. You can find your Edge Impulse API Key via **Dashboard > Keys**.

4. Install Node.js 20.
5. Build and run this project to label your data:

    ```
    npm run build
    node build/llm-labeling.js \
        --which-labels piece \
        --prompt "What chess piece is this? Respond with only the name of the piece. If this is not a chess piece, say 'remove'." \
        --disable-labels "remove" \
        --concurrency 10 \
        --data-ids-file ids.json
    ```

6. Afterwards you'll have labeled data in your project.

### Pushing block to Edge Impulse (enterprise only)

If you've modified this block, you can push it back to Edge Impulse so it's available to everyone in your organization.

1. Update `parameters.json` to update the name and description of your block.
2. Initialize and push the block:

    ```
    $ edge-impulse-blocks init
    $ edge-impulse-blocks push
    ```

3. Afterwards, you can run your block through **Data acquisition > AI Actions** in any Edge Impulse project.

### Proposed changes

AI Actions blocks should be able to run in 'preview' mode (triggered when you click *Label preview data* in the Studio) - where changes are _staged_ but not directly applied. If this is the case `--propose-actions <job-id>` is passed into your block. When you see this flag you should not apply changes directly (e.g. via `api.rawData.setSampleBoundingBoxes`) but rather use the `setSampleProposedChanges` API. Search for this API in [llm-labeling.ts](llm-labeling.ts) to see how this should be used.
