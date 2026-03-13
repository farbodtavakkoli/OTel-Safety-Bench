import os
import json
import time
import scalarlm
import argparse
import datetime
import traceback

# export SCALARLM_API_URL=http://otel_llm_8b_safety.farbodopensource.org
# export SCALARLM_API_URL=https://tensorwave-api.scalarllm.com
# export SCALARLM_API_URL=http://fsdp_test.farbodopensource.org
def get_args():

    parser = argparse.ArgumentParser()

    # Sample data for inference
    parser.add_argument('--test_data_path', default = 'data/Inference_sample_test.jsonl')
    parser.add_argument('--max_tokens', default = 1024)
    parser.add_argument('--batch_size', default = 2)

    # Manual single-request mode
    parser.add_argument('--manual', action='store_true', help='Enable manual single-request mode instead of batch JSON processing')
    parser.add_argument('--context1', type=str, default='', help='Context passage 1')
    parser.add_argument('--context2', type=str, default='', help='Context passage 2')
    parser.add_argument('--context3', type=str, default='', help='Context passage 3')
    parser.add_argument('--context4', type=str, default='', help='Context passage 4')
    parser.add_argument('--context5', type=str, default='', help='Context passage 5')
    parser.add_argument('--question', type=str, default='', help='The question to answer based on the provided contexts')
    
    return parser.parse_args()

# Special tokens formatting for different model types
def format_conversation_inference(example):
    prompt = str(example.get('prompt', ''))

    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

def build_manual_prompt(config):
    """Build a single prompt from manually provided contexts and question."""
    contexts = [
        config.context1,
        config.context2,
        config.context3,
        config.context4,
        config.context5,
    ]

    context_block = ""
    for i, ctx in enumerate(contexts, 1):
        if ctx:
            context_block += f"CONTEXT {i}\n{ctx}\n\n"

    prompt_text = (
        f"CONTEXT:\n{context_block}"
        f"QUESTION:\n{config.question}\n\n"
        f"INSTRUCTIONS:\n"
        f"Provide a brief, direct answer from the CONTEXT above.\n"
        f"- Try to answer in 1-3 sentences\n"
        f"- Skip elaboration unless essential. Do not include fluff in your answer\n"
        f"- If no information available: \"Not found in context\"\n"
        f"- Be concise\n\n"
        f"Answer:"
    )

    return prompt_text


def get_dataset(config):

    dataset = []
    with open(config.test_data_path, 'r+') as f:
        test_data = json.load(f)
    
    count = len(test_data)

    for i in range(count):

        if 'prompt' in test_data[i]:
            prompt_text = test_data[i]['prompt']
        else:
            print(f"Warning: No prompt field found at index {i}: {test_data[i].keys()}")
            continue
        
        example = {
            'prompt': prompt_text,
            'reasoning': test_data[i].get('reasoning', '')
        }
        
        dataset.append(
            format_conversation_inference(example)
        )

    return dataset, test_data


if __name__ == '__main__':
    config = get_args()

    llm = scalarlm.SupermassiveIntelligence()

    if config.manual:
        # --- Manual single-request mode ---
        if not config.question:
            raise ValueError("--question is required in manual mode")

        prompt_text = build_manual_prompt(config)
        formatted_prompt = format_conversation_inference({'prompt': prompt_text})

        print("=" * 60)
        print("MANUAL MODE — Single Request")
        print("=" * 60)

        try:
            t0 = time.time()
            out = llm.generate(
                prompts=[formatted_prompt],
                max_tokens=config.max_tokens
            )
            t1 = time.time()
            generated = out[0] if out else ''
            print(f"\nGenerated Response:\n{generated}")
            print(f"\n(took {t1-t0:.2f} seconds)")
        except Exception as e:
            traceback.print_exc()
            print(e)

    else:
        # --- Batch JSON file mode (original behaviour) ---
        dataset, test_data = get_dataset(config)
        results = []
        processed_indices = []
        batch_size = int(config.batch_size)
        
        for i in range(0, 10, batch_size):
            batch = dataset[i: i+batch_size]
            try:
                t0 = time.time()

                out = llm.generate(
                        prompts=batch, 
                        max_tokens = config.max_tokens
                )
                results += out
                processed_indices += list(range(i, min(i+batch_size, len(dataset))))
                t1 = time.time()
                print(f'took {t1-t0} seconds to do batch of size {batch_size}')
            except Exception as e:
                traceback.print_exc()
                print(e)
                print('batch ', i, ' failed')
                continue
        
        output = []
        for idx, result_idx in enumerate(processed_indices):
            # Extract ground truth from various possible field names
            gt_answer = (
                test_data[result_idx].get('completion') or
                ''
            )
            output.append({
                'ground_truth_response': gt_answer,
                'input_prompt': dataset[result_idx],
                'generated_response': results[idx]
            })

        now = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        save_dir = f'scalarlm_training/runs/{now}'
        os.makedirs(save_dir, exist_ok = False)

        print(f"Saving inference results to {save_dir}/inference_results.json")
        
        with open(os.path.join(save_dir, 'inference_results.json'), 'w+') as f:
            json.dump(output, f, indent = 4)
        
 